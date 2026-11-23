"""Post-training audit and sampling diagnostics. No test-informed retraining."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from benchmark import (CONTEXT, ConditionalGaussian, balanced_weights, make_flow,
                       prepare, sample_neural, write_json)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--results',type=Path,default=Path('results'));args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    data,audit=prepare(args.data_dir,args.results)
    y,c,g=data['test'];meta=json.loads((args.results/'preprocessing.json').read_text())
    manifest=pd.read_csv(args.results/'split_manifest.csv')
    keys=pd.read_csv(args.data_dir/'input_data(full).csv')[['simulation']+CONTEXT].drop_duplicates()
    n=keys.groupby('simulation').size();keys=keys[keys.simulation.isin(n[n==1].index)]
    unique=manifest[['simulation','split']].drop_duplicates().merge(keys,on='simulation',validate='one_to_one')
    assert unique.groupby('simulation').split.nunique().max()==1
    assert (unique.groupby(CONTEXT).split.nunique()<=1).all()
    tr_y,tr_c,tr_g=data['train'];w=balanced_weights(tr_g)
    # Train-only weighted standardization should yield unit-scale target and context distributions.
    mean_y=np.average(tr_y,axis=0,weights=w);mean_c=np.average(tr_c,axis=0,weights=w)
    np.testing.assert_allclose(mean_y,0,atol=1e-6);np.testing.assert_allclose(mean_c,0,atol=1e-6)
    np.testing.assert_allclose(np.average(tr_y**2,axis=0,weights=w),1,atol=1e-5)
    np.testing.assert_allclose(np.average(tr_c**2,axis=0,weights=w),1,atol=1e-5)
    write_json(args.results/'integrity_checks.json',{
        'simulation_disjoint':True,'identical_context_configurations_disjoint':True,
        'duplicate_context_configurations_across_retained_ids':int(unique.duplicated(CONTEXT).sum()),
        'training_weighted_target_mean_max_abs':float(abs(mean_y).max()),
        'training_weighted_context_mean_max_abs':float(abs(mean_c).max())})
    # Inspect predefined parameter quartiles using TRAIN-only group-level cut points.
    # This is descriptive heterogeneity analysis, not another model-selection step.
    group_losses=pd.read_csv(args.results/'per_simulation_nll.csv')
    losses=group_losses.groupby(['model','group'],as_index=False).nll.mean()
    context_by_split=unique.set_index('simulation')
    strata=[]
    for parameter in CONTEXT:
        train_values=context_by_split.loc[context_by_split.split=='train',parameter]
        cuts=np.unique(np.quantile(train_values,[.25,.5,.75]))
        test_values=context_by_split.loc[context_by_split.split=='test',parameter]
        bins=np.searchsorted(cuts,test_values.to_numpy(),side='right')+1
        assignment=pd.DataFrame({'group':test_values.index,'quartile':bins})
        merged=losses.merge(assignment,on='group',validate='many_to_one')
        summary=merged.groupby(['model','quartile']).nll.agg(['mean','count']).reset_index()
        summary['parameter']=parameter
        strata.append(summary.rename(columns={'mean':'test_group_nll','count':'simulations'}))
    pd.concat(strata,ignore_index=True).to_csv(args.results/'parameter_stratified_nll.csv',index=False)
    config=json.loads((args.results/'run_config.json').read_text());results=[]
    # Fixed group sample, one randomly chosen observed atom per group, with reproducible RNG.
    rng=np.random.default_rng(5501);ids=np.unique(g);chosen=rng.choice(ids,min(256,len(ids)),replace=False)
    representative=np.array([rng.choice(np.flatnonzero(g==i)) for i in chosen])
    for seed in config['seeds']:
        for name in ['conditional_gaussian','conditional_spline']:
            path=args.results/f'{name}_seed{seed}.pt'
            ckpt=torch.load(path,map_location='cpu',weights_only=True)
            model=ConditionalGaussian() if name=='conditional_gaussian' else make_flow(True)
            model.load_state_dict(ckpt['state_dict']);model.eval()
            generated=sample_neural(model,name,c,seed+1000)
            physical=generated*np.array(meta['target_scale'])+np.array(meta['target_mean'])
            empirical=y*np.array(meta['target_scale'])+np.array(meta['target_mean'])
            weights=balanced_weights(g)
            spatial=((abs(physical[:,0])>10)|(abs(physical[:,1])>10))
            # No clipping/rejection. Log1p inverse can produce Vz in (-1,0).
            row={'model':name,'seed':seed,'generated_samples':len(g),
                 'all_samples_finite':bool(np.isfinite(generated).all()),
                 'group_weighted_outside_spatial_cut_rate':float(np.average(spatial,weights=weights)),
                 'group_weighted_negative_Vz_rate':float(np.average(physical[:,4]<0,weights=weights)),
                 'observed_group_weighted_negative_Vz_rate':float(np.average(empirical[:,4]<0,weights=weights))}
            # Conditional marginal predictive interval coverage: 256 groups, 128 samples per group.
            context=np.repeat(c[representative],128,axis=0)
            draws=sample_neural(model,name,context,seed+2000).reshape(len(representative),128,5)
            observed=y[representative]
            for level in [.5,.9]:
                lo,hi=np.quantile(draws,[(1-level)/2,1-(1-level)/2],axis=1)
                coverage=((observed>=lo)&(observed<=hi)).mean(axis=0)
                row[f'marginal_coverage_{int(level*100)}']=coverage.tolist()
            # Batch CPU timing; not a simulator speedup or service latency benchmark.
            ctxt=torch.from_numpy(np.repeat(c[:1],512,axis=0));timings=[]
            with torch.no_grad():
                for i in range(23):
                    begin=time.perf_counter()
                    if name=='conditional_gaussian':model.sample_y(ctxt)
                    else:model.sample(512,ctxt)
                    elapsed=time.perf_counter()-begin
                    if i>=3:timings.append(elapsed*1000)
            row['cpu_512_sample_batch_ms_p50']=float(np.quantile(timings,.5))
            row['cpu_512_sample_batch_ms_p95']=float(np.quantile(timings,.95))
            results.append(row);print(json.dumps(row),flush=True)
            np.savez_compressed(args.results/f'{name}_seed{seed}_generated.npz',
                generated_standardized=generated,target_standardized=y,simulation_ids=g)
    write_json(args.results/'sampling_diagnostics.json',{'coverage_group_count':len(representative),
        'samples_per_coverage_group':128,'target_order': ['X','Y','Vx','Vy','log1p(Vz)'],
        'notes':'Marginal interval coverage is a small fixed diagnostic; not joint coverage. Timing is CPU offline batching.',
        'models':results})


if __name__=='__main__':main()
