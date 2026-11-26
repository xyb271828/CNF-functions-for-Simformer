"""Combine recorded cold-atom Simformer, spline and FM results without selection."""
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import group_bootstrap_delta,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));args=ap.parse_args()
    root=args.root
    configurations=[
        ('Conditional spline | max 30','results','conditional_spline',30),
        ('Flow Matching | max 30','results_flow_matching','conditional_flow_matching',30),
        ('Flow Matching | max 120','results_flow_matching_120','conditional_flow_matching',120),
        ('Simformer adaptation | max 30','results_simformer_30','simformer_likelihood_adaptation',30),
        ('Simformer adaptation | max 120','results_simformer_120','simformer_likelihood_adaptation',120)]
    all_rows=[];raw={};sampler_rows=[];samples={}
    for setting,folder,model,budget in configurations:
        m=pd.read_csv(root/folder/'metrics.csv');m=m[m.model==model].copy()
        m['setting']=setting;m['max_epochs']=budget;all_rows.append(m)
        frame=pd.read_csv(root/folder/'per_simulation_nll.csv')
        if 'model' in frame:frame=frame[frame.model==model]
        raw[setting]=frame.groupby('group').nll.mean()
        if folder.startswith('results_simformer'):
            sampler=pd.read_csv(root/folder/'sampling_timing.csv');sampler['setting']=setting;sampler_rows.append(sampler)
        if model=='conditional_spline':fname='conditional_spline_seed11_generated.npz'
        elif model=='conditional_flow_matching':fname='flow_matching_seed11_generated.npz'
        else:fname='simformer_seed11_generated.npz'
        saved=np.load(root/folder/fname);samples[setting]=saved['generated_standardized']
        if len(samples)==1:y=saved['target_standardized'];groups=saved['simulation_ids']
        else:
            np.testing.assert_array_equal(y,saved['target_standardized'])
            np.testing.assert_array_equal(groups,saved['simulation_ids'])
    all_rows=pd.concat(all_rows,ignore_index=True);all_rows.to_csv(root/'simformer_comparison.csv',index=False)
    summary=all_rows.groupby('setting').agg(mean_nll=('test_group_nll','mean'),sd_nll=('test_group_nll','std'),
        seeds=('seed','count'),mean_best_epoch=('best_epoch','mean'),mean_train_seconds=('training_seconds','mean'))
    summary.to_csv(root/'simformer_comparison_summary.csv')
    comparisons={}
    for s,f in [('Simformer adaptation | max 30','Conditional spline | max 30'),
                ('Simformer adaptation | max 30','Flow Matching | max 30'),
                ('Simformer adaptation | max 120','Flow Matching | max 120'),
                ('Simformer adaptation | max 120','Conditional spline | max 30')]:
        comparisons[s+' minus '+f]=group_bootstrap_delta(raw[s],raw[f])
    write_json(root/'simformer_pairwise_comparisons.json',comparisons)
    pd.concat(sampler_rows,ignore_index=True).to_csv(root/'simformer_sampling_timing.csv',index=False)
    table=['| 模型设置 | 平均测试 NLL ± 种子标准差 | 种子数 |','|---|---:|---:|']
    for setting,_,_,_ in configurations:
        row=summary.loc[setting];table.append(f'| {setting} | {row.mean_nll:.4f} ± {row.sd_nll:.4f} | {int(row.seeds)} |')
    comparison_table=['| 比较（Simformer 减去另一个模型） | 平均 NLL 差 | 配对 bootstrap 95% 区间 |','|---|---:|---:|']
    for name,v in comparisons.items():
        lo,hi=v['paired_simulation_bootstrap_95_ci']
        comparison_table.append(f"| {name} | {v['baseline_minus_candidate_group_nll']:.4f} | [{lo:.4f}, {hi:.4f}] |")
    p30=json.loads((root/'results_simformer_30/protocol.json').read_text())
    frames=all_rows[all_rows.model=='simformer_likelihood_adaptation']
    con=[]
    for folder in ['results_simformer_30','results_simformer_120']:
        con.extend(json.loads((root/folder/'convergence.json').read_text()))
    dsmsteps=frames.groupby('max_epochs').optimizer_steps.agg(['min','max']).to_dict()
    param=int(frames.parameters.iloc[0])
    epoch_table=['| 训练上限 | 种子 | 实际 epochs | 选中 epoch | 实际 updates |','|---|---:|---:|---:|---:|']
    for row in frames.itertuples():
        epoch_table.append(f'| {row.max_epochs} | {row.seed} | {int(row.epochs_run)} | {int(row.best_epoch)} | {int(row.optimizer_steps)} |')
    sim30=summary.loc['Simformer adaptation | max 30','mean_nll']
    spline30=summary.loc['Conditional spline | max 30','mean_nll']
    fm30=summary.loc['Flow Matching | max 30','mean_nll']
    sim120=summary.loc['Simformer adaptation | max 120','mean_nll']
    fm120=summary.loc['Flow Matching | max 120','mean_nll']
    report=f'''# 冷原子数据上的 Simformer 对照实验

## 已完成的工作

补跑了基于官方 JAX/Haiku 最小示例的 Simformer 冷原子适配模型。
其 Transformer 与原始截图一致：2 层、2 heads、attention key size 10、widening factor 3，
value/node-ID/condition embedding 为 20/20/10，时间 Fourier embedding 为 64。
14 个 scalar tokens 对应九个实验控制参数和五个原子状态，共 {param:,} 个参数。

这是**正向 likelihood 任务专用的 Simformer 适配版本**：九个控制参数始终为已观测条件，
五个状态变量被加噪并预测 score；没有训练任意随机 conditioning masks，也没有声称完成
论文全部 posterior、missing-data 或 arbitrary-conditioning 实验。官方上游 commit 为
`{p30['upstream_commit']}`，选取的源码、许可证与提取方式在 vendor_simformer 中保留。

## 是否与 normalizing flow / FM 使用同一数据

是。使用完全相同的原始两个 CSV、去重/空间筛选/无歧义参数关联、训练集标准化和 simulation 分组划分。
新输出的 `split_manifest.csv` 和 `preprocessing.json` 已与既有 benchmark 逐字节校验一致。

| 划分 | 原子记录 | simulation |
|---|---:|---:|
| 训练 | 23,195 | 8,734 |
| 验证 | 4,857 | 1,872 |
| 测试 | 5,372 | 1,872 |

总计 33,424 条记录、12,478 个 simulation，是原始去重记录中约 13% 的无歧义映射子集。
本次没有恢复有歧义的 source batch/run 信息，不能推广到所有原始数据。

## 测试结果

{chr(10).join(table)}

NLL 越低越好；对每个 simulation 内原子 NLL 求平均，再对 simulation 等权平均。
三个初始化种子为 11、22、33；标准差是种子间样本标准差。
坐标为训练集标准化的 X_mm、Y_mm、Vx、Vy、log1p(Vz)，与既有两类 flow 实验一致。
本次新 Simformer 数字可以用于这组对照，不能与旧 toy loss 或历史 NLL=3.11 混用。

{chr(10).join(comparison_table)}

差值正数有利于另一个 flow 模型，负数有利于 Simformer；区间跨零时不能据该区间认定稳定优势。
区间对种子平均后的逐 simulation 损失做配对 bootstrap，仅覆盖测试组抽样不确定性，
不是完整的模型选择、数据来源或优化不确定性。

## 训练设置与公平性边界

- 同样的两档训练上限：30 和 120 epochs，batch size 512；每轮 46 次更新。
- 在这份固定数据上，30 轮上限为 1,380 updates，120 轮上限为 5,520 updates；
  提前停止会使实际步数更少。历史 toy 截图的 75,000 updates 并未在本次冷原子任务重放。
- MLE-weighted denoising score matching；九个条件不加噪，只计算五个生成状态的 score loss。
  每个 simulation 在训练和验证中等权。
- AdamW、学习率从 .001 余弦衰减到 .0001、weight decay .0001、梯度裁剪 5、patience=6。
  这些优化设置用于当前固定预算对照，不是历史 toy 运行的全部超参数逐项复现。
- checkpoint 按四组固定 noise/time draws 的验证 DSM 选择；FM 按验证 CFM MSE，spline 按验证 NLL。
  三种网络参数量、训练目标、框架和选择准则不同，因此不是只改变一种因素的因果实验，
  也不是对各方法充分调参后最优能力的排名。

{chr(10).join(epoch_table)}

## Simformer 的 likelihood 如何计算

使用原最小示例的 VE noise schedule：sigma_min=.001，sigma_max=15，最小时间 .01，
score 输出缩放下限 .01。终点初始化分布使用训练集组等权矩加 VE 终点噪声的对角 Gaussian 近似。

训练 loss 是 DSM，不是 NLL。测试 NLL 来自 learned score 的 probability-flow ODE，
积分只对五个生成变量计算精确 divergence，不把九个观测参数的密度加入 NLL。
ODE 仍为数值近似；先在固定验证子集做 64/128 步精度比较，必要时使用 256 步，
只有通过收敛阈值后才计算测试 NLL。六次运行的最大验证均值差为
{max(v['mean_abs_nll_difference'] for v in con):.6g} nats，最大单点差为
{max(v['max_abs_nll_difference'] for v in con):.6g} nats。

新增解析 Gaussian probability-flow 与线性变换密度测试，验证积分方向和五维体积修正。
模型保存在最小正时间截断处；不能把数值 ODE 的 likelihood 描述成原始数据真实分布的精确概率。

## 采样与耗时

同时保存 probability-flow ODE 的生成分布诊断和反向 SDE 500 步诊断。
后者对应原始示例中的迭代式采样路线。有限步 SDE 和数值 ODE 的生成结果并非逐点相同，
不能把 ODE NLL 当作有限步 SDE 离散 sampler 的精确密度。

`simformer_sampling_timing.csv` 记录每个种子的 512 样本 batch p50/p95（3 次预热、5 次计时），
包括 ODE 32/128 步和反向 SDE 100/500 步，JIT 编译不计入这些 warm batch timing。
这是 CPU 离线批采样，不是线上响应延迟；与历史另一会话的 wall time 比较需谨慎。
训练初次 update 的编译用时另行记录。

## 如何向老师描述

现在已经有同一冷原子数据、同一分组测试划分下的 Simformer 适配 baseline，
可以引用表中**注明预算和模型配置**的差异。若某个差值不利于 flow，也应保留，
不能预设加入 normalizing flow 或 FM 一定更好。

可直接引用的英文实验描述：

> I evaluated a cold-atom adaptation of the official minimal-example Simformer
> architecture on the same preprocessed data and simulation-level test split as
> the flow models. Across three initialization seeds, mean test NLL was
> {sim30:.3f} for Simformer, {spline30:.3f} for the conditional spline flow, and
> {fm30:.3f} for Flow Matching under a maximum 30-epoch budget. Under separate
> maximum 120-epoch settings with validation-based early stopping, Simformer
> reached {sim120:.3f} and Flow Matching reached {fm120:.3f}. These results describe
> the tested architectures and optimization settings, rather than the best
> achievable performance of each method.

这些依然是独立模型的对照，尚未将 conditional flow 或 FM 集成进 Simformer。
本次也是新做的回顾性扩展，不应倒写成 2024 年已完成这些实验。

## 复现

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
SIMFORMER_CPU_OFFSET=0 python src/simformer_cold_atom.py --data-dir /path/to/data/compiled --baseline-dir results --out results_simformer_30 --epochs 30 --seeds 11 22 33
SIMFORMER_CPU_OFFSET=0 python src/simformer_cold_atom.py --data-dir /path/to/data/compiled --baseline-dir results --out results_simformer_120 --epochs 120 --seeds 11 22 33
python src/report_simformer.py --root .
```

源数据不包含在本包内。所有已运行配置、日志、权重、样本和逐 simulation 损失均保留在结果目录。
来源：https://github.com/mackelab/simformer ，MIT 许可证随 vendor 源码保留。
'''
    (root/'SIMFORMER_COMPARISON_ZH.md').write_text(report)
    fig,ax=plt.subplots(figsize=(11,4.5))
    for i,(setting,_,_,_) in enumerate(configurations):
        vals=all_rows[all_rows.setting==setting].test_group_nll
        ax.scatter([i]*len(vals),vals,s=35)
    ax.set_xticks(range(len(configurations)),['Spline\nmax30','FM\nmax30','FM\nmax120','Simformer\nmax30','Simformer\nmax120'])
    ax.set_ylabel('Held-out simulation-weighted NLL (lower is better)')
    ax.set_title('Same cold-atom data and split; different architectures / declared budgets')
    fig.tight_layout();fig.savefig(root/'simformer_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.8))
    for j,folder in enumerate(['results_simformer_30','results_simformer_120']):
        for seed in [11,22,33]:
            h=pd.read_csv(root/folder/f'simformer_seed{seed}_history.csv')
            axes[j].plot(h.epoch,h.validation_fixed_mc_group_dsm,label=f'seed {seed}')
        axes[j].set_title(folder.replace('results_',''));axes[j].set_xlabel('Epoch');axes[j].set_ylabel('Validation DSM (not NLL)');axes[j].legend()
    fig.tight_layout();fig.savefig(root/'simformer_training_curves.png',dpi=180);plt.close(fig)
    print(summary.to_string());print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
