"""Summarize all completed budgets without selecting favorable seeds or solvers."""
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark import group_bootstrap_delta,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));args=ap.parse_args()
    root=args.root;parts=[];timing=[];quality=[]
    baseline=pd.read_csv(root/'results/metrics.csv');baseline=baseline[baseline.model=='conditional_spline'].copy()
    baseline['setting']='Spline, max 30 epochs';parts.append(baseline)
    for folder,label in [('results_flow_matching','FM, max 30 epochs'),('results_flow_matching_120','FM, max 120 epochs')]:
        m=pd.read_csv(root/folder/'metrics.csv');m['setting']=label;parts.append(m)
        t=pd.read_csv(root/folder/'cpu_sampling_timing.csv')
        t=t[(t.model=='flow_matching')&(t.solver=='rk4')&(t.steps==32)].copy();t['setting']=label;timing.append(t)
        q=pd.read_csv(root/folder/'common_quality_metrics.csv')
        q=q[q.model=='conditional_flow_matching'].copy();q['setting']=label;quality.append(q)
    t=pd.read_csv(root/'results_flow_matching/cpu_sampling_timing.csv')
    t=t[t.model=='conditional_spline'].copy();t['setting']='Spline, max 30 epochs';timing.append(t)
    q=pd.read_csv(root/'results_flow_matching/common_quality_metrics.csv')
    q=q[q.model=='conditional_spline'].copy();q['setting']='Spline, max 30 epochs';quality.append(q)
    m=pd.concat(parts,ignore_index=True);m.to_csv(root/'budget_comparison.csv',index=False)
    summary=m.groupby('setting').agg(nll_mean=('test_group_nll','mean'),nll_sd=('test_group_nll','std'),
        train_seconds_mean=('training_seconds','mean'),best_epoch_mean=('best_epoch','mean'))
    ts=pd.concat(timing).groupby('setting').p50_ms.mean()
    qs=pd.concat(quality).groupby('setting')[['mean_marginal_wasserstein','joint_context_target_sliced_wasserstein_64']].mean()
    old=pd.read_csv(root/'results/per_simulation_nll.csv');s=old[old.model=='conditional_spline'].groupby('group').nll.mean()
    f30=pd.read_csv(root/'results_flow_matching/per_simulation_nll.csv').groupby('group').nll.mean()
    f120=pd.read_csv(root/'results_flow_matching_120/per_simulation_nll.csv').groupby('group').nll.mean()
    compare={'spline_minus_fm30':group_bootstrap_delta(s,f30),
             'spline_minus_fm120':group_bootstrap_delta(s,f120),
             'fm30_minus_fm120':group_bootstrap_delta(f30,f120)}
    write_json(root/'budget_comparisons.json',compare)
    rows=['| 设置 | 测试 NLL，均值 ± SD | 训练用时均值（参考） | 512 样本 p50 用时 | 边缘 W1 |',
          '|---|---:|---:|---:|---:|']
    for name in ['Spline, max 30 epochs','FM, max 30 epochs','FM, max 120 epochs']:
        row=summary.loc[name]
        rows.append(f'| {name} | {row.nll_mean:.4f} ± {row.nll_sd:.4f} | {row.train_seconds_mean:.2f} s | {ts[name]:.2f} ms | {qs.loc[name,"mean_marginal_wasserstein"]:.4f} |')
    ext=compare['spline_minus_fm120'];ci=ext['paired_simulation_bootstrap_95_ci'];d=ext['baseline_minus_candidate_group_nll']
    if ci[0]>0:
        verdict='延长 FM 训练预算后，在这份共享 holdout 上获得了更低的平均 NLL；但采样成本和预算差异仍需一并报告。'
    elif ci[1]<0:
        verdict='在本次两个 FM 预算下，spline flow 的平均 NLL 仍更低，建议保留 spline 作为当前密度建模主结果。'
    else:
        verdict='扩展预算的 FM 与 spline 的配对区间跨零，没有足够证据认定稳定的 NLL 优势；应保留两个模型和完整预算信息。'
    report=f'''# 是否加入 Flow Matching：完整预算对照

**建议加入实验。** 它让项目从“实现一种 flow”升级为有方法选择依据的条件生成建模比较。
是否把它写为性能提升，必须以具体指标、训练预算和采样成本为前提。

{verdict}

{chr(10).join(rows)}

三个设置均使用种子 11/22/33、相同 simulation-disjoint 数据划分和训练集标准化。
NLL 对 1,872 个测试 simulation 等权，越低越好。FM 采样用固定 RK4 32 步，spline 是直接可逆变换采样。
训练 wall time 来自不同运行且网络结构不同，只供参考，不能宣称严格训练加速。

Spline 减扩展预算 FM 的平均测试 NLL 差为 **{d:.4f} nats**，配对 simulation bootstrap 的 95% 区间为
**[{ci[0]:.4f}, {ci[1]:.4f}]**；正数有利于 FM。SD 与 bootstrap 区间描述不同的不确定性，不能混用。
30 与 120 轮是分别训练的模型，学习率 schedule 长度也随预算改变；不能解释为完全相同轨迹的续训。
提前停止可能使实际训练轮数低于上限，具体轮数和 checkpoint 已记录。

## 重要边界

- 这是本次新做的回顾性扩展，不能倒写为 2024 年在 Imperial 已取得这些结果。
- 最大 120 轮预算是在查看 30 轮结果后增加的探索性分析，所有预算和种子完整报告。
- 共享测试集此前已用于 spline；进一步根据本次结果调参时，应另留独立确认集。
- 目前仅使用原始去重数据中约 13% 的无歧义映射子集，仍需恢复原始 batch/run 标识。
- 这不是所有 Flow Matching 路径、OT pairing、网络和优化设置的能力上限。

详细方法、数值精度、覆盖率和速度曲线见两个 FLOW_MATCHING_REPORT 文档及对应结果目录。
'''
    (root/'FM_DECISION_ZH.md').write_text(report)
    settings=['Spline, max 30 epochs','FM, max 30 epochs','FM, max 120 epochs']
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for i,name in enumerate(settings):
        axes[0].scatter([i]*3,m[m.setting==name].test_group_nll,s=40)
    ticklabels=['Spline\nmax 30 epochs','FM\nmax 30 epochs','FM\nmax 120 epochs']
    axes[0].set_xticks(range(3),ticklabels);axes[0].set_ylabel('Test NLL per simulation (lower is better)')
    axes[0].set_title('All three initialization seeds')
    axes[1].bar(range(3),[ts[n] for n in settings],color=['#3973ac','#e9a23b','#408b73'])
    axes[1].set_xticks(range(3),ticklabels);axes[1].set_ylabel('CPU batch p50, 512 samples (ms)')
    axes[1].set_title('Spline direct sampling; FM fixed RK4-32')
    fig.suptitle('Same data split; explicitly different training budgets')
    fig.tight_layout();fig.savefig(root/'budget_comparison.png',dpi=180);plt.close(fig)
    print(summary.to_string());print('Extended-budget comparison:',ext)


if __name__=='__main__':main()
