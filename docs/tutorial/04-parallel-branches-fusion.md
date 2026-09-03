# 04. 多分支并行模型与融合

## 4.1 为什么不用一个更大的模型就好了

把 03 章的两层 SNN 加宽加深,精度会有提升,但很快遇到收益递减:

- 单一模型只能看到"一种视角"的输入表示(比如只看统计特征,看不到原始时序细节)
- 单一模型的错误模式是相关的——加宽加深并不能修复"这个模型结构性看不到的信息"
- 在真实项目(sEMG 手势识别)里,最终采用的方案是**三个结构不同、看不同输入表示的分支
  各自训练,再做概率融合**,比任何单个分支都更准也更稳:某个分支在某一类手势上表现差,
  另一个分支可能刚好互补。

这是一个很通用的系统设计思路,不限于 SNN:**多个"专家"分支 + 后端融合**,
比"一个更大的模型"往往更容易调、更容易分析、更容易在资源受限的硬件上分别优化。

## 4.2 设计两个视角不同的分支(教学简化版)

延续 02~03 章的合成数据,我们人为构造"两种视角":

- **Context 分支**:看输入特征的全局统计视图(就是原始的 20 维特征本身)
- **Conv 分支**:假装原始信号是一段更细粒度的多通道时间序列,用 1D 卷积 + LIF 提取局部模式

为了让 demo 保持自包含,我们把 20 维特征重排成一个 `[time=10, channels=2]` 的伪时序输入
(真实项目里,这一步对应"原始肌电波形"这种天然带时间轴的信号,不需要人为重排):

```python
import torch
import torch.nn as nn

def to_pseudo_sequence(x, time=10, channels=2):
    # x: [batch, 20] -> [batch, time, channels]  (仅用于教学演示的重排)
    return x.view(x.shape[0], time, channels)

class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad):
        return grad

spike = SpikeSTE.apply

class ContextBranch(nn.Module):
    """看全局统计特征的分支,结构就是 03 章的两层 SNN。"""
    def __init__(self, in_features=20, hidden=64, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.out = nn.Linear(hidden, num_classes)
        self.beta = nn.Parameter(torch.zeros(hidden))

    def forward(self, x, steps=8):
        beta = torch.sigmoid(self.beta)
        mem = torch.zeros(x.shape[0], self.fc1.out_features)
        acc = torch.zeros(x.shape[0], self.out.out_features)
        current = self.fc1(x)
        for _ in range(steps):
            mem = beta * mem + current
            s = spike(mem - 1.0)
            mem = mem - s
            acc = acc + self.out(s)
        return acc / steps


class ConvBranch(nn.Module):
    """看伪时序输入的分支:1D 卷积提取局部模式,再接 LIF。"""
    def __init__(self, channels=2, conv_out=16, num_classes=5):
        super().__init__()
        self.conv = nn.Conv1d(channels, conv_out, kernel_size=3, padding=1)
        self.out = nn.Linear(conv_out, num_classes)
        self.beta = nn.Parameter(torch.zeros(conv_out))

    def forward(self, x_seq):
        # x_seq: [batch, time, channels] -> conv 需要 [batch, channels, time]
        current = self.conv(x_seq.transpose(1, 2)).transpose(1, 2)  # [batch, time, conv_out]
        beta = torch.sigmoid(self.beta)
        batch, time, _ = current.shape
        mem = torch.zeros(batch, self.conv.out_channels)
        acc = torch.zeros(batch, self.out.out_features)
        for t in range(time):
            mem = beta * mem + current[:, t]
            s = spike(mem - 1.0)
            mem = mem - s
            acc = acc + self.out(s)
        return acc / time
```

两个分支**分别独立训练**(各自的损失、各自的优化器、各自在验证集上早停),
就是重复 03 章的训练循环,分别喂 `x`(Context)和 `to_pseudo_sequence(x)`(Conv)。
这里不重复代码,直接进入融合部分。

## 4.3 融合:加权概率求和 + 验证集网格搜索权重

两个分支各自输出 logits,先各自转成概率(softmax),再做**加权求和**:

```
p_fused = w_context * softmax(logits_context) + w_conv * softmax(logits_conv)
pred = argmax(p_fused)
```

权重 `w_context, w_conv`(满足 `w_context + w_conv = 1`)怎么选?**在验证集上网格搜索**,
测试集从头到尾不参与这个搜索过程:

```python
import numpy as np
import torch.nn.functional as F

def fuse_and_search(logits_a_val, logits_b_val, y_val, logits_a_test, logits_b_test, y_test, resolution=20):
    prob_a_val = F.softmax(logits_a_val, dim=1).numpy()
    prob_b_val = F.softmax(logits_b_val, dim=1).numpy()
    y_val_np = y_val.numpy()

    best_w, best_acc = None, -1.0
    for w in np.linspace(0.0, 1.0, resolution + 1):
        fused = w * prob_a_val + (1 - w) * prob_b_val
        acc = (fused.argmax(1) == y_val_np).mean()
        if acc > best_acc:
            best_acc, best_w = acc, w

    print(f"validation-selected weight: w_context={best_w:.2f}, w_conv={1-best_w:.2f}, val_acc={best_acc:.4f}")

    prob_a_test = F.softmax(logits_a_test, dim=1).numpy()
    prob_b_test = F.softmax(logits_b_test, dim=1).numpy()
    fused_test = best_w * prob_a_test + (1 - best_w) * prob_b_test
    test_acc = (fused_test.argmax(1) == y_test.numpy()).mean()
    print(f"test_acc with validation-selected weight = {test_acc:.4f}")
    return best_w, test_acc
```

真实项目里分支数是 3(不是 2),网格搜索就从"一条线段上找一个点"变成
"在一个二维单纯形(simplex)上按某个分辨率打网格",思路完全一样,只是多循环一层。

## 4.4 类别不均衡时的偏置校准

如果某一类(比如"静息/背景"类)样本天然占比很高,模型会有系统性偏向,
融合之后加一个**只调"背景类" logit 偏置**的小校准步骤往往很有效:

```python
def calibrate_rest_bias(prob_val, y_val, rest_class=0, bias_range=np.linspace(-1.5, 0.5, 41)):
    best_bias, best_acc = 0.0, -1.0
    for bias in bias_range:
        adjusted = prob_val.copy()
        adjusted[:, rest_class] *= np.exp(bias)   # 等价于在 logit 空间给背景类加一个偏置
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        acc = (adjusted.argmax(1) == y_val).mean()
        if acc > best_acc:
            best_acc, best_bias = acc, bias
    return best_bias, best_acc
```

同样是**只在验证集上搜索**,搜到的 `best_bias` 再应用到测试集上做一次性报告。

## 4.5 真实项目里的数字(作为参考,不是本教程要复现的目标)

在真实的三分支 sEMG 手势识别项目里:单个分支的测试集准确率大约在 85%~89% 之间,
三分支(不同结构:统计特征分支、原始波形卷积分支、一个基于延迟编码的分支)融合 +
背景类偏置校准后,测试集准确率能到 91% 以上,**超过任何单个分支**。
这说明"结构不同、犯错方式不相关的多个专家 + 融合"是一个成本不高但很有效的精度提升手段,
尤其是在你已经把单个模型调得差不多、边际收益变小的阶段。

## 4.6 小结

- 多分支设计的核心价值不是"更大的模型",而是"结构不同、错误不相关的多个视角"
- 融合权重和任何校准参数,永远只在验证集上搜索,测试集只用来做最后一次报告
- 到这里为止,你已经有了一个精度不错的多分支 SNN 系统。接下来的章节要解决完全不同的问题:
  **这样一个浮点模型,能不能塞进一块小 FPGA?**

下一步: [05-quantization-motivation.md](05-quantization-motivation.md)
