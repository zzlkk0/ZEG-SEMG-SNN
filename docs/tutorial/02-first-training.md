# 02. 训练你的第一个 SNN 分类器

这一章的目标:用 01 章的 LIF 神经元,在一份合成数据上,从零训练出一个能跑到
90%+ 准确率的小分类器。全程不依赖任何外部数据集。

## 2.1 数据:合成分类任务

用 `sklearn.datasets.make_classification` 造一份多类别、有一定难度的表格数据,
把它当成"传感器特征向量"(在真实项目里,这一步的输入可能是肌电信号窗口的统计特征、
振动信号的频谱特征等等——SNN 分类器不关心特征的物理含义,只关心数值)。

```python
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(
    n_samples=4000, n_features=20, n_informative=14,
    n_classes=5, n_clusters_per_class=2, random_state=0,
)
X = (X - X.mean(0)) / X.std(0)   # 归一化到接近 N(0,1),后面编码成脉冲需要这一步

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=0, stratify=y_tmp)

print(X_train.shape, X_val.shape, X_test.shape)
```

## 2.2 编码:把连续特征变成脉冲序列(速率编码)

SNN 的输入通常也得是脉冲(或者至少是"随时间变化的电流")。最简单的编码方式是
**速率编码(rate coding)**:把每个特征值映射到 [0,1] 的一个发放概率,
每个时间步独立采样一次伯努利分布,时间步越多,平均发放率越接近真实概率——
用发放的"密度"编码数值大小。

```python
def rate_encode(x, steps):
    """x: [batch, features], 值域大致在 [-3,3](已标准化). 返回 [steps, batch, features] 的 0/1 脉冲."""
    prob = torch.sigmoid(torch.as_tensor(x, dtype=torch.float32))   # 映射到 (0,1) 概率
    prob = prob.unsqueeze(0).expand(steps, *prob.shape)
    return torch.bernoulli(prob)
```

> 这不是唯一的编码方式。真实项目里常见的是直接把连续特征当作**恒定输入电流**喂给第一层
> (不做逐步伯努利采样),因为伯努利采样引入的噪声对小数据集不一定有利。
> 本教程用伯努利速率编码是因为它最直观地展示"脉冲是怎么携带信息的";
> 04 章开始我们会切换成更简单可靠的"恒定电流"输入方式。

## 2.3 模型:单层 LIF 分类器

```python
import torch.nn as nn

class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad):
        return grad   # 直通估计器:梯度原样传回去

spike = SpikeSTE.apply

class FirstSNN(nn.Module):
    def __init__(self, in_features, hidden, num_classes, beta=0.85):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.beta = beta

    def forward(self, spike_input):
        # spike_input: [steps, batch, in_features]
        steps, batch, _ = spike_input.shape
        mem = torch.zeros(batch, self.fc1.out_features)
        output_sum = torch.zeros(batch, self.fc2.out_features)
        for t in range(steps):
            current = self.fc1(spike_input[t])
            mem = self.beta * mem + current
            s = spike(mem - 1.0)          # 阈值固定为 1.0
            mem = mem - s                  # 软复位
            output_sum = output_sum + self.fc2(s)
        return output_sum / steps          # 对所有时间步的输出取平均,当作 logits
```

要点:

- `mem` 每个时间步更新一次,`fc1` 的权重在所有时间步之间**共享**(这是 SNN 的常规做法,
  不是每个时间步一套权重)。
- 最后一层 `fc2` 输出不发放脉冲,直接把每一步的线性输出累加平均,当作分类 logits——
  这样可以用普通的交叉熵损失训练,不需要在输出层也做脉冲离散化。

## 2.4 训练循环

```python
model = FirstSNN(in_features=20, hidden=64, num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

STEPS = 15
for epoch in range(30):
    model.train()
    perm = torch.randperm(len(X_train_t))
    total_loss = 0.0
    for i in range(0, len(perm), 128):
        idx = perm[i:i+128]
        batch_x = rate_encode(X_train_t[idx].numpy(), STEPS)
        logits = model(batch_x)
        loss = criterion(logits, y_train_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)

    model.eval()
    with torch.no_grad():
        val_logits = model(rate_encode(X_val_t.numpy(), STEPS))
        val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()
    print(f"epoch {epoch:02d}  train_loss={total_loss/len(perm):.4f}  val_acc={val_acc:.4f}")
```

跑起来大概 20~30 个 epoch 就能到 90% 左右的验证集准确率(合成数据比较简单)。

## 2.5 常见坑

1. **忘记替代梯度** —— 如果直接用 `(mem - 1.0 >= 0).float()` 而不经过自定义 `Function`,
   PyTorch 会报"这个操作不可导"或者梯度直接是 0,网络完全学不动。
2. **阈值和输入幅度不匹配** —— 如果输入电流普遍很大(比如没有归一化),
   膜电位一两步就冲到很高,几乎每步都发放,退化成"总是 1";反过来如果输入太小,
   膜电位永远到不了阈值,输出全是 0。**归一化输入 + 固定阈值=1.0 是最简单的搭配。**
3. **时间步数太少** —— 少于 5~10 步,速率编码的随机噪声会淹没信号,验证集准确率会很不稳定。

## 2.6 小结

你现在有了一个能训练、能收敛的最简单 SNN。下一章我们会把它做"深"、做"多步",
并引入可学习的衰减系数——这是通向实用精度的关键一步。

下一步: [03-deeper-multistep.md](03-deeper-multistep.md)
