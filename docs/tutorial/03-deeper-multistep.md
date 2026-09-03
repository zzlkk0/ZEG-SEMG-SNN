# 03. 做深、做多步:两层 LIF + 可学习衰减 + 训练工程细节

02 章的单层模型能跑,但精度天花板不高。这一章把它升级成一个更接近真实项目里
"能打"的版本:两层 LIF、可学习的衰减系数、更稳健的训练循环(梯度裁剪、早停、验证集选模型)。

## 3.1 换一种更稳的输入方式:恒定电流而不是伯努利采样

02 章用伯努利采样做速率编码,噪声较大。真实项目里更常见、更稳的做法是:
**把同一个(归一化后的)特征向量在每个时间步原样喂给网络,让 LIF 自己的时间动态
(积分、发放、复位)产生随时间变化的内部状态**,而不是在输入端就随机化。
也就是说,"时间"这个维度主要用来给网络做"多次积分、多次判断"的机会,
不一定非要靠输入端的随机采样来制造时间变化。

```python
def constant_encode(x, steps):
    """x: [batch, features] -> [steps, batch, features],每一步原样重复."""
    x = torch.as_tensor(x, dtype=torch.float32)
    return x.unsqueeze(0).expand(steps, *x.shape).contiguous()
```

## 3.2 可学习衰减系数

02 章的 `beta` 是一个固定超参数。让它可学习,网络能自己找到"这一层该记多久"的最优值。
用 `sigmoid` 把一个无约束的实数参数映射到 (0,1) 区间,是最简单的参数化方式:

```python
import torch.nn as nn

class LearnableBeta(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(num_channels))  # 初值 sigmoid(0)=0.5

    def value(self):
        return torch.sigmoid(self.raw)
```

如果是"每个神经元一个衰减值"(而不是全层共享一个标量),`num_channels` 就设成该层的宽度,
训练会自动让一部分神经元"记得快"、一部分"记得慢"——这在时序任务上通常比全局共享一个
`beta` 效果更好,代价是多了一点点参数量(硬件上对应每个神经元一个独立的衰减寄存器)。

## 3.3 两层网络

```python
class SpikeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x >= 0).float()
    @staticmethod
    def backward(ctx, grad):
        return grad

spike = SpikeSTE.apply

class TwoLayerSNN(nn.Module):
    def __init__(self, in_features, hidden1, hidden2, num_classes, substeps=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, num_classes)
        self.beta1 = LearnableBeta(hidden1)
        self.beta2 = LearnableBeta(hidden2)
        self.substeps = substeps   # 每个时间步内部再做几次积分-发放

    def forward(self, x_seq):
        # x_seq: [steps, batch, in_features]
        steps, batch, _ = x_seq.shape
        b1, b2 = self.beta1.value(), self.beta2.value()
        m1 = torch.zeros(batch, self.fc1.out_features)
        m2 = torch.zeros(batch, self.fc2.out_features)
        output_sum = torch.zeros(batch, self.out.out_features)
        for t in range(steps):
            current1 = self.fc1(x_seq[t])
            for _ in range(self.substeps):
                m1 = b1 * m1 + current1
                s1 = spike(m1 - 1.0)
                m1 = m1 - s1
                current2 = self.fc2(s1)
                m2 = b2 * m2 + current2
                s2 = spike(m2 - 1.0)
                m2 = m2 - s2
                output_sum = output_sum + self.out(s2)
        return output_sum / (steps * self.substeps)
```

`substeps`("每个外部时间步内部再积分几次")是一个很实用的技巧:它让网络在不增加
输入编码步数的情况下获得更多"思考"机会,类似给每一帧输入更多时间做递归处理。
真实项目里,这个参数经常和输入的物理采样窗口大小配合调优。

## 3.4 训练工程:梯度裁剪 + 早停 + 验证集选模型

深一点的 SNN 训练不稳定性会上升(脉冲在多层间传递,容易出现"某一层几乎不发放"或
"发放过于密集"的情况),几个简单但很关键的工程手段:

```python
def evaluate(model, X, y, steps):
    model.eval()
    with torch.no_grad():
        logits = model(constant_encode(X, steps))
        acc = (logits.argmax(1) == y).float().mean().item()
    return acc

model = TwoLayerSNN(in_features=20, hidden1=64, hidden2=32, num_classes=5)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
criterion = nn.CrossEntropyLoss()

best_val_acc, best_state, patience, bad_epochs = 0.0, None, 8, 0
STEPS = 8

for epoch in range(40):
    model.train()
    perm = torch.randperm(len(X_train_t))
    for i in range(0, len(perm), 128):
        idx = perm[i:i+128]
        logits = model(constant_encode(X_train_t[idx], STEPS))
        loss = criterion(logits, y_train_t[idx])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 防止脉冲反传梯度爆炸
        optimizer.step()
    scheduler.step()

    val_acc = evaluate(model, X_val_t, y_val_t, STEPS)
    print(f"epoch {epoch:02d}  val_acc={val_acc:.4f}")

    # 早停 + 只在验证集上选模型(测试集全程不参与训练决策)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        bad_epochs = 0
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("early stopping")
            break

model.load_state_dict(best_state)
test_acc = evaluate(model, X_test_t, y_test_t, STEPS)
print(f"final test_acc={test_acc:.4f}  (selected by val_acc={best_val_acc:.4f})")
```

**为什么强调"只用验证集选模型"**:如果直接跟着测试集准确率调超参 / 选 checkpoint,
测试集准确率就不再是"未见数据上的诚实估计",而是被间接过拟合了。这在后面的量化章节
(06~07)会再次很重要——量化后的模型也必须用同一套验证集流程选择,而不是挑一个
测试集上表现最好的量化配置。

## 3.5 软复位 vs 硬复位

- **软复位**(减去阈值,`mem = mem - spike * threshold`):保留了超出阈值的多余电位,
  下一步会带着这部分继续积分。梯度更连续,通常更好训练。本教程全程用这个。
- **硬复位**(直接归零,`mem = mem * (1 - spike)`):电位归零,丢失多余信息,但硬件实现
  更简单(不需要减法器,只需要一个多路选择器)。真实项目做硬件设计时,如果发现软复位的
  额外精度收益很小,可能会换成硬复位以省资源——这是一个"训练时想清楚,硬件实现时权衡"
  的典型例子。

## 3.6 小结

现在你有了一个训练工程比较完整的两层 SNN:可学习衰减、多子步积分、梯度裁剪、
早停+验证集选模型。下一章开始进入"系统设计"层面——为什么真实项目通常不会只依赖
一个模型,而是搭多个并行分支再做融合。

下一步: [04-parallel-branches-fusion.md](04-parallel-branches-fusion.md)
