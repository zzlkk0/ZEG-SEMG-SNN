# 06. 硬件友好算子替换 + 量化感知训练(QAT)

这一章是全套课程里工程含量最高的一章:我们要搭一套"伪量化"算子库,
替换掉 05 章列出的每一个"贵"的浮点算子,然后重新训练(热启动微调)找回精度。

## 6.1 伪量化的核心工具:round-STE

所有伪量化算子的地基都是同一个技巧(01 章讲过的替代梯度的一个特例):
**前向做真正的四舍五入(不可导),反向让梯度直接原样通过**。

```python
import torch

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad   # 直通:忽略 round 本身不可导这件事,梯度原样传回去

round_ste = _RoundSTE.apply
```

有了它,后面所有"量化到某个网格"的操作都是同一个套路:
`量化值 = round_ste(原始值 / 步长) * 步长`,只是"步长怎么算"和"限幅范围是多少"因场景而异。

## 6.2 权重量化:逐输出通道对称 INT(bits)

```python
def fake_quant_weight(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """逐输出通道(dim 0)对称量化。weight: [out_channels, ...]"""
    limit = 2 ** (bits - 1) - 1                                  # 例如 INT4 -> limit=7
    reduce_dims = tuple(range(1, weight.dim()))                   # 除了输出通道维,其余都参与求最大值
    amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    scale = amax / limit
    q = round_ste(weight / scale).clamp(-limit, limit)
    return q * scale
```

关键点:`amax` 是在"除了输出通道外的所有维度"上取的——也就是每一路输出神经元/
每一个卷积核,各自有一个独立的 scale。这比"整个权重矩阵共用一个 scale"精度损失小很多
(某一路输出如果数值天生偏小,不会被别的通道的大数值拖着一起用一个粗糙的量化步长)。

用它替换普通的 `nn.Linear` / `nn.Conv1d`:

```python
import torch.nn as nn
import torch.nn.functional as F

class QuantLinear(nn.Linear):
    def __init__(self, *args, bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.bits = bits

    def forward(self, x):
        weight = fake_quant_weight(self.weight, self.bits)
        return F.linear(x, weight, self.bias)
```

`self.bias` 故意没有量化——bias 通常用更宽的位宽保存(比如 INT16/INT32 累加器精度),
因为它不参与逐元素乘法,存储和加法成本都远低于权重矩阵,量化它带来的资源收益很小,
却可能引入不必要的精度损失。

## 6.3 激活量化:统一 Qm.n 网格

对应 05 章讲过的 Q8.8:

```python
def fake_quant_activation(x: torch.Tensor, frac_bits: int = 8, int_bits: int = 8) -> torch.Tensor:
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** (int_bits + frac_bits - 1) - 1
    q = round_ste(x / scale).clamp(-limit, limit)
    return q * scale
```

**在哪里插入这个函数**:每一次"线性/卷积输出 -> 下一层输入"之间,以及每一次膜电位更新
(`mem = beta * mem + current`)之后,都要过一次 `fake_quant_activation`——
这才能让训练时看到的数值精度,和硬件里寄存器实际能存的精度一致。
漏掉某一处不量化,训练出来的网络会"以为"那里有浮点精度,实际部署后行为会跑偏。

## 6.4 衰减系数量化:k/256 网格

膜电位衰减系数 `beta` 也要量化——硬件上通常希望它是"移位友好"的定点小数
(比如直接对应一个 8 位寄存器,`beta_hw = k/256`,`k` 是 1~256 的整数):

```python
def fake_quant_decay(beta: torch.Tensor, frac_bits: int = 8) -> torch.Tensor:
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** frac_bits
    q = round_ste(beta / scale).clamp(1, limit)   # 下限 clamp 到 1,避免衰减系数量化到 0(死神经元)
    return q * scale
```

## 6.5 替换 LayerNorm:为什么它不能直接量化,只能换掉

05 章提过 LayerNorm 的问题:均值、方差是**数据依赖**的,每个样本每次前向都要重新算,
没法预先烧到电路里,而且还带一个除法/开方。解决办法不是"把 LayerNorm 量化得更粗糙",
而是**换成一个完全不同、但效果相近的算子**:

```python
def fake_quant_tensor(weight: torch.Tensor, bits: int) -> torch.Tensor:
    """整个张量共享一个 scale 的对称量化(不是逐通道)。"""
    limit = 2 ** (bits - 1) - 1
    amax = weight.detach().abs().amax().clamp_min(1e-8)
    scale = amax / limit
    q = round_ste(weight / scale).clamp(-limit, limit)
    return q * scale


class HWAffine(nn.Module):
    """LayerNorm 的硬件友好替代:逐通道仿射(scale+bias),不算均值方差。"""
    def __init__(self, dim: int, bits: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.bits = bits

    def forward(self, x):
        weight = fake_quant_tensor(self.weight, self.bits)
        return x * weight + self.bias
```

`HWAffine` 就是一个纯粹的、**数据无关**的逐通道线性变换(`y = x*w + b`,`w`、`b` 是训练出来的
固定参数,不依赖当前输入的统计量)。它没有 LayerNorm 那种"对当前 batch/样本做归一化"的能力,
理论上表达能力更弱,但配合 QAT 微调,实践中足以找回大部分精度——因为前面几层的量化
本身已经把数值范围控制得比较稳定,后面不再需要 LayerNorm 那种强力的在线归一化。

**这个替换还有一个额外的、决定性的硬件收益:数据无关意味着它是线性的,可以在导出时
直接"折叠"进前一层的权重和 bias 里**(数学上 `Affine(Linear(x)) = Linear'(x)`,
两个线性变换可以合并成一个),LayerNorm 因为有均值方差,做不到这一点。折叠后,
硬件上这一层几乎不占用额外资源——07 章导出定点权重时会具体展开这个折叠怎么算。

## 6.6 替换 GELU:换成 ReLU6

`GELU` 依赖 `erf`,`ReLU6`(`clip(x, 0, 6)`)只是一个比较器+限幅,硬件成本几乎为零:

```python
def relu6(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 6.0)
```

QAT 微调时直接把模型定义里的 `nn.GELU()` 换成这个函数即可,不需要专门写伪量化版本
(它本身已经是分段线性,量化后的输入喂进去,输出自然落在量化网格上)。

## 6.7 折叠 BatchNorm

BatchNorm 训练时是数据依赖的(用 batch 统计量),但**推理时用的是训练完固定下来的
running_mean / running_var**,这就是数据无关的仿射变换,可以直接折叠进前面的卷积层:

```python
def fold_batchnorm(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    bn_scale = bn_weight / (bn_var + eps).sqrt()
    bn_shift = bn_bias - bn_mean * bn_scale
    folded_weight = conv_weight * bn_scale[:, None, None]   # 逐输出通道缩放卷积核
    folded_bias = conv_bias * bn_scale + bn_shift
    return folded_weight, folded_bias
```

折叠后硬件上根本不需要一个独立的 BatchNorm 模块,它已经"消失"在卷积权重里了。
训练阶段仍然保留标准的 `nn.BatchNorm1d`(它的在线统计量估计逻辑不需要重新发明),
只在**导出定点权重时**做一次折叠(07 章会具体展开)。

## 6.8 替换除法:用查找表(LUT)

如果模型里有"除法"(比如某种注意力机制里的归一化系数是 `intersection / union`,
`union` 是运行时才知道的整数计数),硬件上避免除法器的标准做法是**用查找表**——
因为 `union` 的取值范围通常是有限的(比如 0~某个时间步数上限),
可以把 `1/union` 对每个可能的 `union` 值预先算好,存进一张小表:

```python
def build_reciprocal_table(max_value: int, frac_bits: int = 12):
    values = np.arange(0, max_value + 1)
    denom = np.maximum(values, 1)          # 避免除以 0
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** frac_bits
    table = np.clip(np.round((1.0 / denom) / scale), 0, limit) * scale
    return table.astype(np.float32)        # table[i] ≈ 1/i,硬件上就是一张 BRAM 表
```

训练时用一个"伪量化的倒数"函数模拟这张表的精度损失:

```python
def fake_quant_reciprocal(value: torch.Tensor, frac_bits: int = 12) -> torch.Tensor:
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** frac_bits
    q = round_ste(value / scale).clamp(0, limit)
    return q * scale
```

推理时(硬件/07 章的 numpy 参考实现)直接用整数索引查表,**没有任何除法运算**。

## 6.9 QAT 训练循环:热启动 + 微调

有了所有伪量化算子,QAT 的训练循环和普通训练几乎一样,关键区别是三点:

1. **从已有的浮点(FP32)checkpoint 热启动**,不是从随机初始化开始——伪量化算子引入的
   噪声,让"从零训练"收敛明显更慢、更不稳定;热启动通常几个 epoch 就能找回大部分精度。
2. **学习率要调低**(比如浮点训练阶段的 1/10~1/5),因为热启动的起点已经接近一个好的解,
   只是需要"适应"量化噪声,不需要大幅度重新搜索参数空间。
3. **模型的 state_dict key 名字可能对不上**(比如把 `nn.LayerNorm` 换成了 `HWAffine`),
   热启动时需要写一个小的 key 重映射函数,把能对应上的权重(比如所有 Linear/Conv 层)
   直接拷贝过去,对不上的(比如新引入的 `HWAffine.weight/bias`)用默认初始化。

```python
def remap_state_dict(fp32_state: dict) -> dict:
    remapped = {}
    for key, value in fp32_state.items():
        new_key = key.replace("encoder.0.", "enc_linear.").replace("encoder.1.", "enc_affine.")
        remapped[new_key] = value
    return remapped

# 训练循环骨架(其余部分和 03 章几乎一样)
hw_model = HWTwoLayerSNN(...)
fp32_state = torch.load("fp32_checkpoint.pt")["model"]
hw_model.load_state_dict(remap_state_dict(fp32_state), strict=False)   # strict=False:允许新 key 缺省初始化

optimizer = torch.optim.AdamW(hw_model.parameters(), lr=2e-4)  # 比浮点训练时的学习率低一个量级
# ... 后续训练循环、早停、验证集选模型,和 03 章完全一样
```

## 6.10 一个真实踩过的坑:看起来在量化,实际上什么都没量化

这是一个非常值得记住的案例。`HWAffine` 最初的实现是这样写的(错误版本):

```python
# 错误版本:先 unsqueeze(-1) 再调用逐输出通道量化函数
weight = fake_quant_weight(self.weight.unsqueeze(-1), self.bits).squeeze(-1)
```

看起来合理——复用了 6.2 节的 `fake_quant_weight`。但仔细看 `fake_quant_weight` 的实现:
它在"除了 dim 0 外的所有维度"上取 `amax` 求 scale。`self.weight` 形状是 `[dim]`
(一维,每个通道一个数),`unsqueeze(-1)` 之后变成 `[dim, 1]`——**参与求 amax 的维度
(dim 1)大小是 1**,也就是说每个通道的 `amax` 就是它自己的绝对值。
于是 `scale = |w_i| / limit`,量化后的编码值**永远是 ±limit**(比如 INT8 就是 ±127),
`q * scale` 算出来正好精确等于原始的 `w_i`——**这是一个无损重建,完全没有发生任何量化**。

**这个 bug 是怎么被发现的**:不是靠代码 review 看出来的,而是在 07 章要讲的
"用 numpy 参考实现和 torch QAT 模型交叉验证"环节,发现两者在一小部分样本上的预测不一致
(不该有差异,因为两边理论上算的是同一套定点数值)。深挖之后才定位到这里。

**修复**:改成用一个真正"整个张量共享一个 scale"的量化函数(就是 6.5 节的
`fake_quant_tensor`),而不是错误地复用逐通道量化函数:

```python
# 正确版本
weight = fake_quant_tensor(self.weight, self.bits)
```

修完之后重新做 QAT 微调,精度**不降反升**(说明之前那个"伪量化"根本没有让网络
适应任何真实的精度损失,反而让训练在一个和最终硬件行为不一致的假设下进行)。

### 从这个案例里应该学到的通用检查方法

写完任何一个伪量化算子后,养成这个习惯:**打印一下量化后实际用到的编码值范围/
唯一值个数**。如果编码值集中在一两个极端值附近,或者唯一值个数和原始浮点值个数一样多,
大概率是 scale 算错了,量化根本没有生效:

```python
with torch.no_grad():
    weight = fake_quant_tensor(model.some_affine.weight, bits=8)
    codes = torch.round(weight / (weight.abs().max() / 127))
    print("distinct codes used:", codes.unique().numel(), "/ possible:", 255)
```

如果 `distinct codes used` 明显小于 `possible`,并且分布看起来合理(不是全部挤在
`±127`),说明量化大概率是生效的。

## 6.11 小结

到这一步,你已经有了:一套伪量化算子库(权重逐通道 INT、激活 Qm.n、衰减 k/2^n、
倒数查找表)、LayerNorm→HWAffine(可折叠)、GELU→ReLU6、BatchNorm 折叠公式、
以及热启动 QAT 的训练套路,外加一个关于"如何验证量化真的生效"的踩坑案例。

下一步要做的是:把训练好的 QAT 模型**真正导出成整数编码 + scale**(不再是"伪量化的
浮点数"),并写一份完全不依赖 PyTorch 的 numpy 参考实现来验证这份定点规格是自洽的。

下一步: [07-export-numpy-verify.md](07-export-numpy-verify.md)
