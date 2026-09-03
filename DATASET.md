# 数据与模型文件说明

## 为什么仓库不包含数据和 checkpoint

NinaPro DB5 数据受其发布方条款约束，文件体积也不适合直接提交到 Git。
训练 checkpoint、处理后的窗口、导出权重和 bitstream 都是可重新生成的产物，
因此本仓库只发布源代码、教学文档和不含个人样本的结果说明。

请从 NinaPro 官方渠道获取 DB5，并遵守其引用和使用条件。不要把原始受试者数据
或重新打包的数据集提交到本仓库。

## 推荐目录

```text
training/semg_snn_fpga_reproduction/
  data/
    extracted/        DB5 解压后的 MATLAB 文件
    processed/        prepare_db5.py 生成的 train/val/test.npz
  runs/               checkpoint 与指标

training/semg_snn_90_loop/
  data/               prepare.py 生成的特征数据
  runs/               Context、Hybrid 和 QAT checkpoint/指标
  weights_hw/         定点导出包与黄金向量
```

这些目录均已被 `.gitignore` 排除。

## 固定划分

| Split | Repetitions | 窗口数 |
|---|---|---:|
| Train | 1, 2, 4, 6 | 44,630 |
| Validation | 3 | 11,060 |
| Test | 5 | 11,276 |

窗口为 100 点，shift 为 20 点，从每段记录的第 400 点以后开始。动作窗要求至少
80% 标签一致，Rest 窗要求全部为 0。

## 准备数据

```bash
cd training/semg_snn_fpga_reproduction
python prepare_db5.py \
  --input-dir data/extracted \
  --output-dir data/processed

cd ../semg_snn_90_loop
python prepare.py
python prepare_continuous_context.py
```

部分优化脚本会读取相邻的 `semg_snn_fpga_reproduction/data/processed`。保持
仓库默认目录结构即可，无需修改源码。

## 隐私要求

若使用自行采集的 sEMG：

- 不提交原始受试者记录、姓名、编号映射、同意书或设备序列号；
- 在发布统计前确认伦理审批与受试者授权范围；
- 公开示例优先使用合成数据或经授权的匿名小样本；
- 日志中不要记录本机绝对路径、用户名或访问令牌。
