# 01. 脉冲神经元(LIF)与替代梯度

## 1.1 ANN 神经元 vs SNN 神经元

一个普通 ANN 神经元做的事情很简单:

```
y = activation(W x + b)
```

一次前向,一个输出,没有"时间"的概念。

SNN 神经元不一样。它有一个**内部状态**(膜电位,membrane potential),
随时间步(timestep)累积输入电流,电位超过阈值就"发放一个脉冲"(spike,值为 0 或 1),
发放后电位回落。这是在模拟真实神经元的行为,也正是它天然适合硬件的原因:
**脉冲是二值信号,不需要乘法器传递,大量运算可以退化成"门控加法"。**

最常用的模型叫 **LIF(Leaky Integrate-and-Fire,漏电积分发放)**:

```
mem[t] = beta * mem[t-1] + I[t]        # 漏电积分:上一步电位打折(漏电)+ 这一步输入电流
spike[t] = 1 if mem[t] >= threshold else 0   # 发放
mem[t] = mem[t] - spike[t] * threshold        # 发放后电位减去阈值("软复位"/减法复位)
```

- `beta`(衰减系数,∈(0,1))控制"记忆"能维持多久。beta 越接近 1,记忆越长。
- `threshold` 一般归一化成 1.0,让 `mem` 和 `spike` 都在一个干净的数值范围里。
- "软复位"(减去阈值)比"硬复位"(直接置零)更常用,因为它保留了超出阈值的那部分电位,
  训练时梯度信号更连续,不会突然丢一大块信息。

## 1.2 最小可运行例子:仿真单个神经元

```python
import numpy as np

def lif_trace(input_current, beta=0.8, threshold=1.0, steps=20):
    mem = 0.0
    mem_trace, spike_trace = [], []
    for t in range(steps):
        mem = beta * mem + input_current[t]
        spike = 1.0 if mem >= threshold else 0.0
        mem = mem - spike * threshold
        mem_trace.append(mem)
        spike_trace.append(spike)
    return np.array(mem_trace), np.array(spike_trace)

# 恒定输入电流 0.3,看看多久发放一次脉冲
current = np.full(20, 0.3)
mem, spikes = lif_trace(current)
print("membrane:", np.round(mem, 2))
print("spikes:  ", spikes.astype(int))
```

跑一下会看到膜电位阶梯式上升,每次到达 1.0 附近就发放一个脉冲然后回落——
这就是"积分-发放-复位"的循环。**输入电流越大,发放频率越高**,这是 SNN 编码信息的基本方式之一
(速率编码,rate coding,02 章会用到)。

## 1.3 为什么不能直接反向传播

`spike = 1 if mem >= threshold else 0` 是一个阶跃函数(Heaviside step function)。
它的导数几乎处处为零,只有在阈值那一点是无穷大——直接用它做反向传播,
梯度要么是 0(学不动)要么是无穷大(数值爆炸)。这是训练 SNN 最核心的技术难点。

### 替代梯度(Surrogate Gradient)/ 直通估计器(STE)

解决办法很朴素:**前向该怎么算怎么算(离散的 0/1 脉冲不变),
反向传播时假装这个函数是一个"看起来差不多"的连续函数**,比如 sigmoid 的导数、
或者干脆用"直通"(gradient 原样传回去,即 STE, Straight-Through Estimator)。

在 PyTorch 里用 `torch.autograd.Function` 手写:

```python
import torch

class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mem_minus_threshold):
        ctx.save_for_backward(mem_minus_threshold)
        return (mem_minus_threshold >= 0).float()   # 前向:真正的阶跃函数

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # 替代梯度:用一个"伪导数"近似阶跃函数的导数,
        # 比如以阈值为中心的窄三角形/sigmoid 导数,这里用简单的分段线性
        surrogate = torch.clamp(1.0 - x.abs(), min=0.0)
        return grad_output * surrogate

spike = SpikeFunction.apply
```

前向调用 `spike(mem - threshold)`,返回值仍然是干净的 0/1(硬件要的就是这个);
反向传播时梯度会顺着 `surrogate` 这个连续函数流回去,让网络能学习。
**这一个技巧是所有能训练的 SNN 的共同基础**,后面所有章节都在用它。

> 提示:本课程后续章节为了简化,会直接用 `(x >= 0).float()` 配合更简化的 STE
> (反向传播时梯度原样通过,不做近似整形)。效果和上面的分段线性替代梯度接近,
> 实现更简单,收敛也够用。两种写法你都会在开源 SNN 代码里见到。

## 1.4 小结

- LIF 神经元 = 漏电积分(`beta * mem + I`)+ 阈值发放(`spike`)+ 复位(减去阈值)
- 脉冲是二值的,这是 SNN 相比 ANN 在硬件上的核心优势
- 阶跃函数不可导,需要"替代梯度"技巧才能训练
- 下一章我们会用这个神经元真正训练一个分类器

下一步: [02-first-training.md](02-first-training.md)
