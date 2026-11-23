"""Render a factual report from the saved benchmark results; never invent metrics."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=Path('results'))
    args=ap.parse_args();r=args.results
    audit=json.loads((r/'data_audit.json').read_text());metrics=pd.read_csv(r/'metrics.csv')
    comparisons=json.loads((r/'comparisons.json').read_text())
    summary=metrics.groupby('model').test_group_nll.agg(['mean','std','count'])
    names={'unconditional_gaussian':'无条件全协方差 Gaussian', 'unconditional_gmm':'无条件 GMM',
           'conditional_gaussian':'条件 Gaussian MLP（对角协方差）',
           'unconditional_spline':'无条件 spline flow', 'conditional_spline':'条件 spline flow'}
    table=['| 模型 | 测试 NLL：simulation 等权 | 随机种子数 |','|---|---:|---:|']
    for name,row in summary.iterrows():
        value=f"{row['mean']:.4f}"
        if row['count']>1:value+=f" ± {row['std']:.4f}"
        table.append(f"| {names[name]} | {value} | {int(row['count'])} |")
    base=summary.loc['unconditional_spline','mean'];candidate=summary.loc['conditional_spline','mean']
    delta=base-candidate;relative=100*delta/base
    ci=comparisons['unconditional_spline']['paired_simulation_bootstrap_95_ci']
    cnf=metrics[metrics.model=='conditional_spline']
    perm=cnf.context_permutation_nll_penalty.mean()
    fidelity=metrics.groupby('model')[['mean_wasserstein_group_weighted','mean_histogram_js_group_weighted']].mean()
    ftable=['| 模型 | 平均 Wasserstein 距离 | 平均边缘直方图 JS |','|---|---:|---:|']
    for name,row in fidelity.iterrows():
        ftable.append(f"| {names[name]} | {row.iloc[0]:.4f} | {row.iloc[1]:.5f} |")
    diagnosis=json.loads((r/'sampling_diagnostics.json').read_text())
    models=diagnosis['models'];flow_diag=[v for v in models if v['model']=='conditional_spline']
    coverage=np.mean([v['marginal_coverage_90'] for v in flow_diag],axis=0)
    outside=np.mean([v['group_weighted_outside_spatial_cut_rate'] for v in flow_diag])
    batch_ms=np.mean([v['cpu_512_sample_batch_ms_p50'] for v in flow_diag])
    coverage_table=['| 目标 | 90% 条件边缘预测区间的观测覆盖率 |','|---|---:|']
    for label,value in zip(['X','Y','Vx','Vy','log1p(Vz)'],coverage):
        coverage_table.append(f'| {label} | {100*value:.1f}% |')
    report=f'''# 冷原子生成建模：实验优化与复现报告

本报告记录本次新增的回顾性复现实验。代码与结果不代表在 2024 年访问研究期间已经完成这些扩展。

## 核心结果

在可唯一关联参数的 {audit['final_rows']:,} 条原子记录、{audit['final_simulations']:,} 个 simulation 上，
建立了 simulation 分组且互不重叠的训练/验证/测试集。测试集包含
{audit['test']['simulations']:,} 个未出现在训练集的 simulation、{audit['test']['rows']:,} 条记录。
预先指定的主要比较是条件 spline flow 对无条件 spline flow，两个模型使用相同的 flow 深度和隐藏层设置。

{chr(10).join(table)}

均值 ± 样本标准差来自三个初始化种子；Gaussian/GMM 各为一个拟合结果。主要指标是先对每个
simulation 内的原子 NLL 求平均，再对 simulation 求平均。更低表示在本次统一表示和队列上密度估计更好。

条件 flow 相比无条件 flow 的平均 NLL 差为 **{delta:.4f} nats/五维观测**，数值相对下降
**{relative:.1f}%**。对测试 simulation 配对 bootstrap 的 95% 区间为 **[{ci[0]:.4f}, {ci[1]:.4f}]**，
正值有利于条件 flow。区间基于先对种子求平均的逐组损失，只反映 simulation 抽样不确定性，
不涵盖全部训练或数据来源不确定性。相对百分比依赖坐标表示，优先解释绝对 NLL 差。

打乱完整九维条件向量后，条件 flow 的测试 NLL 平均增加 **{perm:.4f} nats**。
这是条件信息相关性的诊断；打乱后的条件/原子组合可能在物理上不匹配，不能解释为现实中的退化幅度。

## 为什么选择这个模型

研究目标是 p(X,Y,Vx,Vy,Vz | 九个激光束和磁场参数)，而不是只预测一个均值。
Spline flow 支持条件采样以及显式条件 likelihood 计算，并允许非高斯分布。
这里用无条件 flow 消融检验条件信息，用条件对角 Gaussian MLP 比较分布表达能力，
用全协方差 Gaussian 和 GMM 提供传统统计基准。结论仅适用于这些已运行的模型和设置。

主 flow 使用 8 个 rational-quadratic spline coupling blocks，每个隐藏网络 2 层、64 单元，
配合 LU linear permutations。优化器为 AdamW，初始学习率 0.001，weight decay 0.0001，
batch size 512，最多 30 epochs，余弦学习率衰减、梯度裁剪和 patience=6 的提前停止规则。
checkpoint 只按验证集逐 simulation 等权 NLL 选择。三种子为 11、22、33。
这些不是对旧模型架构优化幅度的对照实验；未补跑旧的有问题评估流程来制造提升。

## 数据修复与适用范围

- 原始原子文件 {audit['raw_atom_rows']:,} 行，其中 {audit['exact_duplicate_atom_rows']:,} 行为完全重复。
- 去重后 {audit['deduplicated_atom_rows']:,} 行；空间筛选后 {audit['after_spatial_cut_rows']:,} 行。
- 参数表有一次 simulation 编号重置；全表 {audit['ambiguous_parameter_ids']:,} 个编号对应多个不同条件。
- 空间筛选后的 {audit['atom_rows_with_ambiguous_ids']:,} 条记录因条件映射有歧义而排除，
  {audit['atom_rows_with_missing_ids']:,} 条因找不到参数而排除。
- 最终保留 {audit['final_rows']:,} 行，约占去重原子记录的 {100*audit['retained_fraction_of_unique_atoms']:.1f}%。
  此子集存在选择偏差，结果不能推广到所有原始模拟。
- 保留队列中没有不同 simulation ID 对应完全相同的九维条件配置，已额外核查配置级跨集合泄漏。
  唯一键仍不能独立证明来源，需由原始 batch/run 标识确认参数表的物理对应关系。
- 标准化仅用训练集；X/Y 换算为 mm，Vz 使用 log1p。数据预处理与训练对每个 simulation 等权，
  避免产出原子较多的 simulation 主导目标。该模型不预测原子产量或零原子产出概率。

| 划分 | 原子记录 | simulation |
|---|---:|---:|
| 训练 | {audit['train']['rows']:,} | {audit['train']['simulations']:,} |
| 验证 | {audit['validation']['rows']:,} | {audit['validation']['simulations']:,} |
| 测试 | {audit['test']['rows']:,} | {audit['test']['simulations']:,} |

## 生成质量与限制

{chr(10).join(ftable)}

这些边缘指标并不总与条件 NLL 排名一致。无条件模型可以在总体直方图上表现较好，
但无法表达随实验参数变化的条件分布。不能据此宣称条件 flow 在所有生成质量指标上全面最好。
边缘分布对比使用共同 bin edges，移除了旧版不可比的五维直方图 KL。
没有将 chi-square p-value 当作模型正确性的证明。

固定选取 256 个测试 simulation，每组随机取一个观测原子，并生成 128 个条件样本，
检查边缘预测区间的覆盖率。下表对三个种子求平均，是小规模诊断，不是联合区间覆盖率，也不证明完全校准。

{chr(10).join(coverage_table)}

条件 flow 生成样本中，simulation 等权后约 **{100*outside:.2f}%** 的 X/Y 超出训练使用的 ±10 mm 空间范围。
没有对生成样本裁剪或拒绝采样。应考虑显式有界变换或物理约束作为后续独立实验，
并重新评估似然和区间覆盖。三种子平均 CPU 512 样本 batch 中位用时约 **{batch_ms:.2f} ms**；
这是离线批采样计时，不是服务 p95 延迟，也不是相对物理模拟器的加速比。

## 历史结果如何处理

历史结果截图能核实：2-epoch CNF 的 test NLL 为 3.6118，8-epoch 为 3.1062。
后者在同一次训练中 validation NLL 从 3.8823 降至 3.0972。
但旧数据合并、标准化及按行划分存在问题，不能作为可信的泛化性能对照。
本次数据队列、权重、预处理和测试划分改变，**不能把新 NLL 与旧 3.11 做提升比较**。
现有 Simformer 截图是三变量 toy example，也不是本任务的可比 baseline。

## 后续最有价值的实验

1. 恢复 source batch/run key，正确关联被排除的数据，再预先固定新的独立测试队列。
2. 增加 batch 或参数区间外推 holdout；本次只测试同一保留队列中的新 simulation。
3. 对固定参数运行更多原子采样，测试多峰结构、尾部、跨变量相关性和条件校准。
4. 在相同硬件、相同批量和相同产出要求下，对比原模拟器与代理模型的耗时与物理误差。
5. 如果根据本报告测试结果继续选择模型，后续最终结论应使用新保留测试集。


'''
    (r.parent/'EXPERIMENT_REPORT_ZH.md').write_text(report)
    print('Wrote',r.parent/'EXPERIMENT_REPORT_ZH.md')


if __name__=='__main__':main()
