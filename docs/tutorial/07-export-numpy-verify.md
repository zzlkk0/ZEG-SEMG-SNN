# 07. 导出定点权重 + 纯 numpy 参考实现

06 章的 QAT 模型训练时用的是"伪量化"——权重和激活在前向计算里被 round 到定点网格,
但**存储和计算全程仍然是 float32**,只是数值恰好落在量化网格上。这对训练是对的
(反向传播需要浮点梯度),但不是硬件真正跑的东西。这一章要做的是:

1. 把权重**真正导出成整数编码(int8) + 浮点缩放系数(scale)**,不再是"看起来量化了的浮点数"
2. 写一份**完全不 import torch**、纯 numpy 的推理实现,只用整数编码和 scale 重新计算
3. 验证这份 numpy 实现和 QAT 训练时的 torch 前向,在同样输入下给出一致的预测

为什么需要独立于 torch 的纯 numpy 实现?因为下一步(HLS/RTL 开发)是 C++/Verilog 的世界,
不会有人在 FPGA 里跑 PyTorch。numpy 参考实现扮演的角色是"一份可执行的、无歧义的规格书"——
硬件工程师写 RTL 时,应该能用它逐位对照验证自己的实现对不对。

## 7.1 权重打包格式

约定一个简单一致的命名规则:每一层导出三个数组

```
<层名>_codes   int8,量化后的整数编码(权重是 [out, in, ...] 形状,逐输出通道)
<层名>_scale   float32,每个输出通道一个缩放系数(仿射层是单个标量)
<层名>_bias    float32,偏置(不量化,原始精度保留)
```

真实值的重建公式永远是同一个:`真实权重 = codes.astype(float32) * scale`。

```python
import numpy as np
import torch

def quantize_weight(weight: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """对应 06 章 fake_quant_weight 的“真实导出”版本:算出 codes 和 scale,而不是伪量化浮点数。"""
    limit = 2 ** (bits - 1) - 1
    reduce_axes = tuple(range(1, weight.ndim))
    amax = np.abs(weight).max(axis=reduce_axes, keepdims=True) if reduce_axes else np.abs(weight).max()
    amax = np.maximum(amax, 1e-8)
    scale = amax / limit
    codes = np.clip(np.round(weight / scale), -limit, limit).astype(np.int8)
    return codes, scale.reshape(-1).astype(np.float32)


def quantize_affine(weight: np.ndarray, bits: int = 8) -> tuple[np.ndarray, np.float32]:
    """HWAffine 权重:整个张量共享一个 scale(对应 06 章的 fake_quant_tensor)。"""
    limit = 2 ** (bits - 1) - 1
    amax = float(np.maximum(np.abs(weight).max(), 1e-8))
    scale = amax / limit
    codes = np.clip(np.round(weight / scale), -limit, limit).astype(np.int8)
    return codes, np.float32(scale)
```

## 7.2 折叠 BatchNorm 后导出

06 章推导过折叠公式,导出时直接套用,把 `running_mean/running_var/weight/bias`
变成两个数组 `scale`、`shift`(不再需要单独的 BatchNorm 模块):

```python
def export_folded_batchnorm(bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    bn_scale = bn_weight / np.sqrt(bn_var + eps)
    bn_shift = bn_bias - bn_mean * bn_scale
    return bn_scale.astype(np.float32), bn_shift.astype(np.float32)
```

## 7.3 导出脚本骨架

```python
def export_model(state: dict, weight_bits: int, out_path: str) -> None:
    arrays: dict[str, np.ndarray] = {}

    def add_linear(prefix, weight_key, bias_key, bits):
        weight = state[weight_key].detach().cpu().numpy()
        bias = state[bias_key].detach().cpu().numpy().astype(np.float32)
        codes, scale = quantize_weight(weight, bits)
        arrays[f"{prefix}_codes"] = codes
        arrays[f"{prefix}_scale"] = scale
        arrays[f"{prefix}_bias"] = bias

    def add_affine(prefix, weight_key, bias_key):
        weight = state[weight_key].detach().cpu().numpy()
        bias = state[bias_key].detach().cpu().numpy().astype(np.float32)
        codes, scale = quantize_affine(weight)
        arrays[f"{prefix}_codes"] = codes
        arrays[f"{prefix}_scale"] = scale
        arrays[f"{prefix}_bias"] = bias

    add_linear("fc1", "fc1.weight", "fc1.bias", weight_bits)
    add_affine("affine1", "affine1.weight", "affine1.bias")
    add_linear("out", "out.weight", "out.bias", weight_bits)

    # 衰减系数、激活格式等标量元数据一并存进同一个 npz
    arrays["beta"] = np.float32(quantize_decay_value(float(torch.sigmoid(state["beta_raw"]))))
    arrays["act_frac_bits"] = np.int32(8)
    arrays["act_int_bits"] = np.int32(8)

    np.savez(out_path, **arrays)
    print(f"wrote {out_path}, total bytes = {sum(a.nbytes for a in arrays.values())}")
```

## 7.4 纯 numpy 推理实现

**这个文件里绝对不能出现 `import torch`**——这是它作为"独立规格书"的意义所在。
所有算子都用 numpy 手写,和 06 章的伪量化算子做的事情完全一样,只是这次操作的是
真实的整数编码 + scale,而不是伪量化浮点数:

```python
# hw_fixed_reference.py -- 只 import numpy,不 import torch

import numpy as np

def fake_quant_activation(x: np.ndarray, frac_bits: int, int_bits: int) -> np.ndarray:
    scale = 2.0 ** (-frac_bits)
    limit = 2 ** (int_bits + frac_bits - 1) - 1
    q = np.clip(np.round(x / scale), -limit, limit)
    return (q * scale).astype(np.float32)

def spike(x: np.ndarray) -> np.ndarray:
    return (x >= 0).astype(np.float32)

def linear(x: np.ndarray, codes: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    weight = codes.astype(np.float32) * scale[:, None]   # 逐输出通道反量化
    return x @ weight.T + bias

def affine(x: np.ndarray, codes: np.ndarray, scale: np.float32, bias: np.ndarray) -> np.ndarray:
    weight = codes.astype(np.float32) * scale             # 单一 scale 反量化
    return x * weight + bias


class HWFixedModel:
    def __init__(self, npz_path: str):
        with np.load(npz_path) as archive:
            self.w = {k: archive[k] for k in archive.files}
        self.act_frac_bits = int(self.w["act_frac_bits"])
        self.act_int_bits = int(self.w["act_int_bits"])

    def _q(self, x):
        return fake_quant_activation(x, self.act_frac_bits, self.act_int_bits)

    def infer(self, x_seq: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
        w = self.w
        batch = x_seq.shape[0]
        mem = np.zeros((batch, w["fc1_bias"].shape[0]), dtype=np.float32)
        beta = float(w["beta"])
        logits = np.zeros((batch, w["out_bias"].shape[0]), dtype=np.float32)
        current = affine(
            linear(x_seq, w["fc1_codes"], w["fc1_scale"], w["fc1_bias"]),
            w["affine1_codes"], w["affine1_scale"], w["affine1_bias"],
        )
        current = self._q(current)
        for _ in range(steps):
            mem = self._q(beta * mem + current)
            s = spike(mem - 1.0)
            mem = mem - s
            logits = logits + linear(s, w["out_codes"], w["out_scale"], w["out_bias"])
        logits = logits / steps
        return logits, logits.argmax(axis=1)
```

## 7.5 验证:numpy 参考和 torch QAT 模型是否一致

这是全流程里**最重要的一步测试**——它是唯一能证明"你导出的定点规格真的对应训练时
模型行为"的检查手段。方法:同一批验证集样本,分别喂给两边,比较 argmax 是否完全一致、
logits 差异是否在浮点求和顺序误差的量级(而不是逻辑错误的量级):

```python
def cross_validate(torch_model, numpy_model, X_val, steps, atol=1e-2):
    torch_model.eval()
    with torch.no_grad():
        torch_logits = torch_model(constant_encode(X_val, steps)).numpy()

    numpy_logits, numpy_argmax = numpy_model.infer(X_val.numpy(), steps)

    torch_argmax = torch_logits.argmax(axis=1)
    match_rate = (torch_argmax == numpy_argmax).mean()
    max_logit_diff = np.abs(torch_logits - numpy_logits).max()

    print(f"argmax match rate: {match_rate:.4f}")
    print(f"max logit diff: {max_logit_diff:.6f}")
    assert match_rate == 1.0, "numpy reference disagrees with torch QAT model on at least one prediction"
```

**预期结果**:`argmax match rate` 应该是 100%。`max logit diff` 通常不会是精确的 0——
浮点数加法不满足结合律,torch 和 numpy 内部求和的顺序、批处理方式不同,会有量级在
`1e-3`~`1e-1` 的微小残差,这是正常的、可以接受的数值噪声,**不是 bug**。
真正的 bug 信号是 argmax 不匹配,或者残差大到肉眼可见(比如 > 1.0)。

如果发现不匹配,回头检查:是不是某个算子在 06 章的伪量化版本和这里的"真实反量化"版本
之间,scale 的计算方式对不上(就像 06.10 节那个案例)。

## 7.6 黄金测试向量(Golden Test Vectors)

有了验证过的 numpy 参考实现,下一步是给 RTL/HLS 工程师准备一份**带中间值的标准答案**——
不只是"输入 -> 最终输出",还要包含关键的中间检查点(第一层输出、每层最终膜电位、
发放计数、最终 logits),这样硬件仿真如果在某一步就跑偏了,能立刻定位到具体是哪一层,
而不是只知道"最终结果不对,但不知道哪里错的"。

```python
def export_golden_vectors(numpy_model, X_samples, y_labels, steps, out_path):
    w = numpy_model.w
    batch = X_samples.shape[0]
    mem = np.zeros((batch, w["fc1_bias"].shape[0]), dtype=np.float32)
    beta = float(w["beta"])
    logits = np.zeros((batch, w["out_bias"].shape[0]), dtype=np.float32)

    current = affine(
        linear(X_samples, w["fc1_codes"], w["fc1_scale"], w["fc1_bias"]),
        w["affine1_codes"], w["affine1_scale"], w["affine1_bias"],
    )
    current = numpy_model._q(current)
    first_layer_output = current.copy()   # 检查点 1:第一层量化后的输出
    spike_count = np.zeros(batch)

    for _ in range(steps):
        mem = numpy_model._q(beta * mem + current)
        s = spike(mem - 1.0)
        mem = mem - s
        spike_count += s.sum(axis=1)
        logits = logits + linear(s, w["out_codes"], w["out_scale"], w["out_bias"])
    logits = logits / steps

    np.savez(
        out_path,
        inputs=X_samples, truth=y_labels,
        first_layer_output=first_layer_output,
        final_membrane=mem,                 # 检查点 2:最终膜电位
        spike_count=spike_count,             # 检查点 3:总发放次数
        logits=logits, argmax=logits.argmax(1),  # 检查点 4:最终输出
    )
```

选样本时,通常每个类别至少选 1 个代表样本,再加几个连续/相似样本做冗余检查,
凑够十几到二十个样本就够——黄金向量不是用来做统计意义上的精度评估的(那是 07.5 节
交叉验证 + 08 章全量测试集评估的事),它的作用是给硬件仿真提供一个**可以逐行断言比对**
的小规模确定性测试集。

## 7.7 小结

现在你有了:一份真实的整数权重导出(codes + scale,不再是伪量化的浮点数)、
一份完全不依赖 torch 的 numpy 定点参考实现、一套验证它和 QAT 训练结果一致的交叉检验方法、
以及一份供硬件仿真使用的黄金测试向量。这三样东西合在一起,才是"可以真正交给硬件团队"
的完整规格。最后一章我们来看:这个模型现在到底能不能放进目标 FPGA,以及怎么诚实地
回答这个问题。

下一步: [08-resource-estimate-deployment.md](08-resource-estimate-deployment.md)
