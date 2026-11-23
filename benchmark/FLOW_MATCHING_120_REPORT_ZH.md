# Flow Matching 扩展实验

## 结论

建议把 Flow Matching 加入研究对照。它提供了另一种训练条件生成模型的方式，但是否用于简历主结果，
应由实测的质量、稳定性和成本决定。本次没有为了使新方法胜出而更改既有测试划分。

在这组设置和共享测试队列上，Flow Matching 的平均 NLL 更低。

| 模型 | 测试 NLL：均值 ± 标准差 |
|---|---:|
| 条件 Flow Matching | 3.2431 ± 0.0212 |
| 条件 Gaussian MLP | 5.6596 ± 0.0130 |
| 条件 spline flow | 3.5517 ± 0.1381 |
| 无条件 Gaussian | 6.1184 |
| 无条件 GMM | 4.8605 |
| 无条件 spline flow | 4.6590 ± 0.0228 |

神经模型运行三个固定初始化种子 11、22、33；标准差反映种子波动。主要 NLL 在
1,872 个 simulation、5,372 个原子记录上逐 simulation 等权，表示空间与上轮 spline benchmark 完全一致。
**Spline 减 FM 的平均 NLL 差为 0.3086 nats，配对 simulation bootstrap 的 95% 区间为
[0.2682, 0.3520]**。正值有利于 FM，负值有利于 spline。区间基于种子平均后的逐组损失，
只反映测试 simulation 抽样不确定性；不是全面的实验误差界。

## 为什么值得加入

既有模型是离散的 conditional spline normalizing flow；Flow Matching 在这里训练一个连续时间速度场。
二者都建模给定九个实验控制参数的五维原子状态分布。训练 Flow Matching 时不需要求解生成 ODE，
但生成与 likelihood 评估需要数值积分。它是训练和采样机制的对照，而不是自动更优的模型。
术语中 Continuous Normalizing Flow 也常被缩写为 CNF，容易与 Conditional Normalizing Flow 混淆；
本项目明确使用 `conditional_spline` 和 `conditional_flow_matching` 命名。

本次实现的是**独立噪声/数据配对、线性插值路径的 conditional flow matching**：

`z ~ N(0,I), t ~ Uniform(0,1), x_t=(1-t)z+t*x, target_velocity=x-z`。

九维实验参数只作为网络条件输入，不参与输运。没有实现 minibatch optimal transport，不能称为 OT-CFM。
速度 MLP 有 4 个 224 单元隐藏层、SiLU 激活和时间 Fourier 特征，共 157,701 个参数；
spline baseline 有 174,728 个参数。FM 最多 120 epochs，spline baseline 最多 30 epochs，batch 512，AdamW、学习率 0.001、weight decay 0.0001，
余弦衰减、梯度裁剪、patience=6。网络和预算不是完全相同，结论适用于这些已运行的设置。

## 如何保证比较有意义

- 对清洗后的 33,424 条记录，逐字节校验新旧数据划分清单和训练集标准化参数一致。
- 训练及主要指标都对 simulation 等权；没有按行重新划分或按测试指标选择种子。
- FM checkpoint 按验证集固定 4 组 noise/time draws 的 CFM MSE 选择。Spline 按验证 NLL 选择，
  因而 checkpoint 选择规则不同。这是各自常见目标下的固定预算比较，不是严格控制全部变量的因果实验。
- FM 的 MSE 不被当作 NLL。最终 NLL 用反向 ODE 和五维精确 divergence 计算，RK4 128 步。
  “精确”指空间 Jacobian trace，ODE 仍是数值近似。
- 在测试 NLL 之前，对固定 256 个验证 simulation 检查 64/128 步的 NLL 差异，记录在 convergence.json。
  均值差阈值为 0.005 nats，最大差阈值为 0.1，超标会停止报告，而不是保留不收敛结果。
- 已用具有解析解的平移与缩放速度场测试反向积分方向、密度符号和体积修正项。
- 默认采样配置预先设定为 RK4 32 步。Euler 8/16/32/64、RK4 8/16/32 是描述性 solver sweep，
  使用相同初始噪声，不据测试表现重选 checkpoint 或宣称某个配置普遍最优。

## 生成质量

| 模型 | 边缘 W1 | 共同 bin JS | 五维联合 SWD | 条件+目标联合 SWD |
|---|---:|---:|---:|---:|
| 条件 Flow Matching | 0.0367 | 0.00443 | 0.0363 | 0.0245 |
| 条件 Gaussian MLP | 0.0503 | 0.00723 | 0.1353 | 0.0472 |
| 条件 spline flow | 0.0356 | 0.00416 | 0.0359 | 0.0259 |

所有值越低越好。W1 是五个训练标准化坐标的平均 Wasserstein 距离。
JS 使用三种模型共同的训练集定义 bin edges，并包含上下 overflow bins；
这组 JS 是新计算的，不能与旧报告各模型单独范围的 JS 直接混用。
五维 SWD 用固定 64 个随机单位方向，反映目标联合分布；14 维 SWD 对标准化条件与目标一起投影，
提供条件关联的额外诊断，但不能证明每个条件下的完整分布正确，也不等价于条件似然。

FM 的 90% 边缘预测区间覆盖率（固定 256 个测试 simulation，每个 128 个生成样本；种子平均），
依 X、Y、Vx、Vy、log1p(Vz) 顺序为：85.7%, 87.0%, 87.4%, 89.7%, 88.9%。
这是有限样本诊断，不是联合预测区间覆盖率。物理空间超界和负 Vz 比例记录在 common_quality_metrics.csv；
没有通过裁剪或拒绝采样改善这些数值。

## 速度与数值积分成本

| 方法 | Solver | 步数 | 512 个样本 batch 的 p50 用时，三种子平均 |
|---|---|---:|---:|
| conditional_spline | direct | 0 | 20.75 ms |
| flow_matching | euler | 8 | 11.39 ms |
| flow_matching | euler | 16 | 21.34 ms |
| flow_matching | euler | 32 | 40.81 ms |
| flow_matching | euler | 64 | 81.45 ms |
| flow_matching | rk4 | 8 | 39.83 ms |
| flow_matching | rk4 | 16 | 77.14 ms |
| flow_matching | rk4 | 32 | 146.99 ms |

同一进程、CPU、2 个线程、batch 512，3 次预热后计时 10 次；推断时间包括模型采样路径，
FM 使用预先生成的起始噪声，spline sample 包含起始噪声生成。因此这是近似可比的离线批计算计时，
不能解释为线上请求延迟，也不能宣称相对物理模拟器的加速。
一次 RK4 步需要 4 次速度网络求值；训练不用积分不代表推断不用积分。
FM 平均训练用时 34.11 秒，之前 spline 平均 80.75 秒，
记录在各自日志中；两次不同会话的 wall time 只作参考，不能将该比值视为严格训练加速实验。

## 限制与下一步

本次仍只使用参数映射无歧义的约 13% 去重记录；唯一键不替代原始 batch/run provenance 确认。
测试集此前已经用于 spline 实验，本次是共享测试集上的探索性扩展，不是新的盲测。
如据结果继续选择模型、路径、网络或 solver，应使用验证集，并保留新的独立测试队列作最终确认。
本次固定预算和独立配对 linear CFM 不能代表所有 Flow Matching 方法的最优能力。
若增加 OT pairing，必须控制参数条件与配对方式，避免把不同实验条件的原子错误混配。

## 复现与来源

```bash
python src/flow_matching.py --data-dir /path/to/data/compiled --baseline-results results --output-dir results_flow_matching_120 --epochs 120 --seeds 11 22 33
python src/report_flow_matching.py --data-dir /path/to/data/compiled --root . --fm-dir results_flow_matching_120 --report-name FLOW_MATCHING_120_REPORT_ZH.md
python src/predict.py --model flow_matching --results results --fm-results results_flow_matching_120 --context-json example_context.json --samples 1000 --output fm_samples.csv
```

- Lipman et al., Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- 官方 ODE solver / likelihood 文档：https://facebookresearch.github.io/flow_matching/generated/flow_matching.solver.ODESolver.html

本实现依照上述数学定义用 PyTorch 独立实现，并非调用官方 solver。
