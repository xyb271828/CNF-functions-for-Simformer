"""Post-hoc descriptive FM comparison on the previously fixed shared holdout."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from benchmark import balanced_weights,prepare,write_json
from flow_matching import VelocityMLP,sample_all
import torch


def joint_sliced_distance(a,b,weights,seed):
    directions=np.random.default_rng(seed).normal(size=(a.shape[1],64))
    directions/=np.linalg.norm(directions,axis=0)
    pa,pb=a@directions,b@directions
    return float(np.mean([wasserstein_distance(pa[:,j],pb[:,j],weights,weights) for j in range(64)]))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=Path('.'))
    ap.add_argument('--fm-dir',default='results_flow_matching')
    ap.add_argument('--report-name',default='FLOW_MATCHING_REPORT_ZH.md');args=ap.parse_args()
    root=args.root;old=root/'results';out=root/args.fm_dir
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data,_=prepare(args.data_dir,out);train_y,_,_=data['train'];y,c,g=data['test']
    weights=balanced_weights(g);config=json.loads((out/'protocol.json').read_text())
    quality=[];coverage=[];samples_for_plot={}
    # All compared models share the SAME train-defined bins and overflow bins.
    edges=[np.r_[-np.inf,np.linspace(train_y[:,j].min(),train_y[:,j].max(),49),np.inf] for j in range(5)]
    meta=json.loads((out/'preprocessing.json').read_text())
    for name in ['conditional_gaussian','conditional_spline','conditional_flow_matching']:
        for seed in config['seeds']:
            path=(out/f'flow_matching_seed{seed}_generated.npz') if name=='conditional_flow_matching' else old/f'{name}_seed{seed}_generated.npz'
            saved=np.load(path);s=saved['generated_standardized']
            np.testing.assert_array_equal(saved['target_standardized'],y)
            np.testing.assert_array_equal(saved['simulation_ids'],g)
            js=[];w1=[]
            for j in range(5):
                a=np.histogram(y[:,j],bins=edges[j],weights=weights)[0]+1e-8
                b=np.histogram(s[:,j],bins=edges[j],weights=weights)[0]+1e-8
                a/=a.sum();b/=b.sum();m=(a+b)/2
                js.append(float(.5*(a*np.log(a/m)).sum()+.5*(b*np.log(b/m)).sum()))
                w1.append(float(wasserstein_distance(y[:,j],s[:,j],weights,weights)))
            physical=s*np.array(meta['target_scale'])+np.array(meta['target_mean'])
            quality.append({'model':name,'seed':seed,'mean_marginal_wasserstein':float(np.mean(w1)),
                'mean_common_bin_js':float(np.mean(js)),
                'joint_target_sliced_wasserstein_64':joint_sliced_distance(y,s,weights,701),
                'joint_context_target_sliced_wasserstein_64':joint_sliced_distance(np.c_[c,y],np.c_[c,s],weights,702),
                'outside_spatial_cut_rate':float(np.average((abs(physical[:,0])>10)|(abs(physical[:,1])>10),weights=weights)),
                'negative_Vz_rate':float(np.average(physical[:,4]<0,weights=weights))})
            if seed==config['seeds'][0]:samples_for_plot[name]=s
            if name=='conditional_flow_matching':
                ckpt=torch.load(out/f'flow_matching_seed{seed}.pt',weights_only=True)
                model=VelocityMLP();model.load_state_dict(ckpt['state_dict']);model.eval()
                rng=np.random.default_rng(5501);chosen=rng.choice(np.unique(g),256,replace=False)
                representative=np.array([rng.choice(np.flatnonzero(g==i)) for i in chosen])
                context=np.repeat(c[representative],128,axis=0)
                draws=sample_all(model,context,seed+2000,32,'rk4').reshape(256,128,5)
                row={'seed':seed,'groups':256,'draws_per_group':128}
                for level in [.5,.9]:
                    lo,hi=np.quantile(draws,[(1-level)/2,1-(1-level)/2],axis=1)
                    row[f'coverage_{int(level*100)}']=((y[representative]>=lo)&(y[representative]<=hi)).mean(0).tolist()
                coverage.append(row)
    q=pd.DataFrame(quality);q.to_csv(out/'common_quality_metrics.csv',index=False)
    write_json(out/'coverage.json',{'target_order':['X','Y','Vx','Vy','log1p(Vz)'],'models':coverage})
    old_metrics=pd.read_csv(old/'metrics.csv');fm=pd.read_csv(out/'metrics.csv')
    all_metrics=pd.concat([old_metrics,fm],ignore_index=True);all_metrics.to_csv(out/'all_model_metrics.csv',index=False)
    summary=all_metrics.groupby('model').test_group_nll.agg(['mean','std','count'])
    summary.to_csv(out/'all_model_summary.csv')
    compare=json.loads((out/'comparisons.json').read_text())['conditional_spline']
    delta=compare['baseline_minus_candidate_group_nll'];ci=compare['paired_simulation_bootstrap_95_ci']
    times=pd.read_csv(out/'cpu_sampling_timing.csv')
    speed=times.groupby(['model','solver','steps']).p50_ms.mean().reset_index()
    speed.to_csv(out/'mean_cpu_sampling_timing.csv',index=False)
    qsum=q.groupby('model').mean(numeric_only=True)
    labels={'conditional_spline':'条件 spline flow','conditional_flow_matching':'条件 Flow Matching',
            'conditional_gaussian':'条件 Gaussian MLP','unconditional_gaussian':'无条件 Gaussian',
            'unconditional_gmm':'无条件 GMM','unconditional_spline':'无条件 spline flow'}
    table=['| 模型 | 测试 NLL：均值 ± 标准差 |','|---|---:|']
    for name,row in summary.iterrows():
        text=f"{row['mean']:.4f}"+(f" ± {row['std']:.4f}" if row['count']>1 else '')
        table.append(f'| {labels[name]} | {text} |')
    qtable=['| 模型 | 边缘 W1 | 共同 bin JS | 五维联合 SWD | 条件+目标联合 SWD |','|---|---:|---:|---:|---:|']
    for name,row in qsum.iterrows():
        qtable.append(f"| {labels[name]} | {row.mean_marginal_wasserstein:.4f} | {row.mean_common_bin_js:.5f} | {row.joint_target_sliced_wasserstein_64:.4f} | {row.joint_context_target_sliced_wasserstein_64:.4f} |")
    timingtable=['| 方法 | Solver | 步数 | 512 个样本 batch 的 p50 用时，三种子平均 |','|---|---|---:|---:|']
    for row in speed.itertuples():
        timingtable.append(f'| {row.model} | {row.solver} | {row.steps} | {row.p50_ms:.2f} ms |')
    if ci[0]>0:verdict='在这组设置和共享测试队列上，Flow Matching 的平均 NLL 更低。'
    elif ci[1]<0:verdict='在这组设置和共享测试队列上，spline flow 的平均 NLL 更低。Flow Matching 不应被描述为性能升级。'
    else:verdict='该配对区间跨过零，不能认定 NLL 上存在稳定优势。'
    s_mean=float(summary.loc['conditional_spline','mean']);f_mean=float(summary.loc['conditional_flow_matching','mean'])
    trainfm=float(fm.training_seconds.mean());oldtrain=float(old_metrics[old_metrics.model=='conditional_spline'].training_seconds.mean())
    cov=np.mean([v['coverage_90'] for v in coverage],axis=0)
    report=f'''# Flow Matching 扩展实验

## 结论

建议把 Flow Matching 加入研究对照。它提供了另一种训练条件生成模型的方式，但是否用于简历主结果，
应由实测的质量、稳定性和成本决定。本次没有为了使新方法胜出而更改既有测试划分。

{verdict}

{chr(10).join(table)}

神经模型运行三个固定初始化种子 11、22、33；标准差反映种子波动。主要 NLL 在
1,872 个 simulation、5,372 个原子记录上逐 simulation 等权，表示空间与上轮 spline benchmark 完全一致。
**Spline 减 FM 的平均 NLL 差为 {delta:.4f} nats，配对 simulation bootstrap 的 95% 区间为
[{ci[0]:.4f}, {ci[1]:.4f}]**。正值有利于 FM，负值有利于 spline。区间基于种子平均后的逐组损失，
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
速度 MLP 有 4 个 224 单元隐藏层、SiLU 激活和时间 Fourier 特征，共 {int(fm.parameters.iloc[0]):,} 个参数；
spline baseline 有 174,728 个参数。FM 最多 {config['epochs']} epochs，spline baseline 最多 30 epochs，batch 512，AdamW、学习率 0.001、weight decay 0.0001，
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

{chr(10).join(qtable)}

所有值越低越好。W1 是五个训练标准化坐标的平均 Wasserstein 距离。
JS 使用三种模型共同的训练集定义 bin edges，并包含上下 overflow bins；
这组 JS 是新计算的，不能与旧报告各模型单独范围的 JS 直接混用。
五维 SWD 用固定 64 个随机单位方向，反映目标联合分布；14 维 SWD 对标准化条件与目标一起投影，
提供条件关联的额外诊断，但不能证明每个条件下的完整分布正确，也不等价于条件似然。

FM 的 90% 边缘预测区间覆盖率（固定 256 个测试 simulation，每个 128 个生成样本；种子平均），
依 X、Y、Vx、Vy、log1p(Vz) 顺序为：{', '.join(f'{100*v:.1f}%' for v in cov)}。
这是有限样本诊断，不是联合预测区间覆盖率。物理空间超界和负 Vz 比例记录在 common_quality_metrics.csv；
没有通过裁剪或拒绝采样改善这些数值。

## 速度与数值积分成本

{chr(10).join(timingtable)}

同一进程、CPU、2 个线程、batch 512，3 次预热后计时 10 次；推断时间包括模型采样路径，
FM 使用预先生成的起始噪声，spline sample 包含起始噪声生成。因此这是近似可比的离线批计算计时，
不能解释为线上请求延迟，也不能宣称相对物理模拟器的加速。
一次 RK4 步需要 4 次速度网络求值；训练不用积分不代表推断不用积分。
FM 平均训练用时 {trainfm:.2f} 秒，之前 spline 平均 {oldtrain:.2f} 秒，
记录在各自日志中；两次不同会话的 wall time 只作参考，不能将该比值视为严格训练加速实验。

## 限制与下一步

本次仍只使用参数映射无歧义的约 13% 去重记录；唯一键不替代原始 batch/run provenance 确认。
测试集此前已经用于 spline 实验，本次是共享测试集上的探索性扩展，不是新的盲测。
如据结果继续选择模型、路径、网络或 solver，应使用验证集，并保留新的独立测试队列作最终确认。
本次固定预算和独立配对 linear CFM 不能代表所有 Flow Matching 方法的最优能力。
若增加 OT pairing，必须控制参数条件与配对方式，避免把不同实验条件的原子错误混配。

## 复现与来源

```bash
python src/flow_matching.py --data-dir /path/to/data/compiled --baseline-results results --output-dir {args.fm_dir} --epochs {config['epochs']} --seeds 11 22 33
python src/report_flow_matching.py --data-dir /path/to/data/compiled --root . --fm-dir {args.fm_dir} --report-name {args.report_name}
python src/predict.py --model flow_matching --results results --fm-results {args.fm_dir} --context-json example_context.json --samples 1000 --output fm_samples.csv
```

- Lipman et al., Flow Matching for Generative Modeling: https://arxiv.org/abs/2210.02747
- 官方 ODE solver / likelihood 文档：https://facebookresearch.github.io/flow_matching/generated/flow_matching.solver.ODESolver.html

本实现依照上述数学定义用 PyTorch 独立实现，并非调用官方 solver。
'''
    (root/args.report_name).write_text(report)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,name in enumerate(['conditional_spline','conditional_flow_matching']):
        vals=all_metrics[all_metrics.model==name].test_group_nll
        axes[0].scatter([i]*len(vals),vals)
    axes[0].set_xticks([0,1],['Spline flow','Flow Matching']);axes[0].set_ylabel('Test group NLL (lower is better)')
    axes[0].set_title('Three initialization seeds; same held-out simulations')
    quality_solver=pd.read_csv(out/'solver_quality.csv').groupby(['solver','steps']).mean(numeric_only=True).reset_index()
    for method in ['euler','rk4']:
        qq=quality_solver[quality_solver.solver==method]
        tt=speed[(speed.model=='flow_matching')&(speed.solver==method)]
        merge=qq.merge(tt,on=['solver','steps'])
        axes[1].plot(merge.p50_ms,merge.mean_wasserstein_group_weighted,'o-',label=method)
        for row in merge.itertuples():axes[1].annotate(str(row.steps),(row.p50_ms,row.mean_wasserstein_group_weighted),fontsize=7)
    axes[1].set_xlabel('CPU p50 per 512-sample batch (ms)');axes[1].set_ylabel('Mean marginal Wasserstein (lower is better)')
    axes[1].set_title('FM solver tradeoff; labels are integration steps');axes[1].legend()
    fig.tight_layout();fig.savefig(out/'flow_matching_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(17,3.3))
    for j,ax in enumerate(axes):
        bins=np.linspace(*np.quantile(y[:,j],[.005,.995]),51)
        ax.hist(y[:,j],bins=bins,weights=weights,density=True,histtype='step',color='black',label='Simulation')
        for name in ['conditional_spline','conditional_flow_matching']:
            ax.hist(samples_for_plot[name][:,j],bins=bins,weights=weights,density=True,histtype='step',label=name)
        ax.set_title(['X','Y','Vx','Vy','log1p(Vz)'][j]);ax.set_xlabel('Train-standardized value')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(out/'flow_matching_marginals.png',dpi=180);plt.close(fig)
    print('Wrote Flow Matching report and common quality diagnostics',flush=True)


if __name__=='__main__':main()
