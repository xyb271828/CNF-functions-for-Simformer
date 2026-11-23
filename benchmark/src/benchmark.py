"""Reproducible, simulation-group-held-out benchmark for the supplied 2D MOT data.

NLLs are densities in TRAIN-standardized (X_mm,Y_mm,Vx,Vy,log1p(Vz)) space.
They must not be compared numerically with the historical, leaky experiment.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import normflows as nf
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

TARGETS = ['X', 'Y', 'Vx', 'Vy', 'Vz']
CONTEXT = ['cooling_beam_detuning', 'cooling_beam_radius', 'cooling_beam_power_mw',
           'push_beam_detuning', 'push_beam_radius', 'push_beam_power',
           'push_beam_offset', 'quadrupole_gradient', 'vertical_bias_field']


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_join(atoms, params):
    """Discard ambiguous keys; never pick an arbitrary parameter row."""
    params = params[['simulation'] + CONTEXT].drop_duplicates()
    counts = params.groupby('simulation').size()
    ambiguous = set(counts[counts > 1].index)
    unique_params = params[params.simulation.isin(counts[counts == 1].index)]
    matched = atoms.merge(unique_params, on='simulation', how='inner', validate='many_to_one')
    return matched, ambiguous


def balanced_weights(groups):
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inv]
    return w / w.mean()


def group_means(values, groups):
    return pd.DataFrame({'group': groups, 'value': values}).groupby('group').value.mean()


def prepare(data_dir, out):
    atom_path = data_dir / 'compiled_vectors-0.05.csv'
    param_path = data_dir / 'input_data(full).csv'
    atoms_raw, params = pd.read_csv(atom_path), pd.read_csv(param_path)
    atoms = atoms_raw.drop_duplicates().copy()
    audit = {'raw_atom_rows': len(atoms_raw), 'exact_duplicate_atom_rows': len(atoms_raw)-len(atoms),
             'deduplicated_atom_rows': len(atoms), 'parameter_rows': len(params),
             'parameter_id_resets': int((params.simulation.diff() < 0).sum()),
             'input_sha256': {atom_path.name: sha256(atom_path), param_path.name: sha256(param_path)}}
    atoms = atoms[atoms.X.between(-.01, .01) & atoms.Y.between(-.01, .01)].copy()
    audit['after_spatial_cut_rows'] = len(atoms)
    joined, ambiguous = safe_join(atoms, params)
    audit['ambiguous_parameter_ids'] = len(ambiguous)
    audit['atom_rows_with_ambiguous_ids'] = int(atoms.simulation.isin(ambiguous).sum())
    audit['atom_rows_with_missing_ids'] = int((~atoms.simulation.isin(params.simulation)).sum())
    audit['unique_key_matched_rows_before_finite_filter'] = len(joined)
    finite = np.isfinite(joined[TARGETS+CONTEXT]).all(axis=1) & (joined.Vz > -1)
    joined = joined.loc[finite].copy().reset_index(drop=True)
    audit['final_rows'] = len(joined)
    audit['final_simulations'] = int(joined.simulation.nunique())
    audit['retained_fraction_of_unique_atoms'] = len(joined)/len(atoms_raw.drop_duplicates())
    audit['scope'] = ('Unique-key subset only; selection bias is possible. Unique simulation keys '
                      'are consistent in the supplied table but do not independently prove source provenance.')
    joined['record_id'] = np.arange(len(joined))
    train_ix, rest_ix = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=42).split(
        joined, groups=joined.simulation))
    vi, ti = next(GroupShuffleSplit(n_splits=1, test_size=.50, random_state=43).split(
        joined.iloc[rest_ix], groups=joined.iloc[rest_ix].simulation))
    indices = {'train': train_ix, 'validation': rest_ix[vi], 'test': rest_ix[ti]}
    sets = {k: set(joined.iloc[ix].simulation) for k,ix in indices.items()}
    assert not sets['train'] & sets['test'] and not sets['train'] & sets['validation']
    assert not sets['validation'] & sets['test']
    y = joined[TARGETS].to_numpy(dtype=np.float64)
    y[:, :2] *= 1000.
    y[:, 4] = np.log1p(y[:, 4])
    c = joined[CONTEXT].to_numpy(dtype=np.float64)
    # Fit transforms to training only, weighting simulations equally.
    trweights = balanced_weights(joined.iloc[train_ix].simulation.to_numpy())
    ys = StandardScaler().fit(y[train_ix], sample_weight=trweights)
    cs = StandardScaler().fit(c[train_ix], sample_weight=trweights)
    yz, cz = ys.transform(y).astype('float32'), cs.transform(c).astype('float32')
    data = {k: (yz[ix], cz[ix], joined.iloc[ix].simulation.to_numpy()) for k,ix in indices.items()}
    split_rows=[]
    for k, ix in indices.items():
        audit[k]={'rows':len(ix), 'simulations': len(sets[k])}
        part=joined.iloc[ix][['record_id','simulation']].copy();part['split']=k;split_rows.append(part)
    pd.concat(split_rows).sort_values('record_id').to_csv(out/'split_manifest.csv',index=False)
    write_json(out/'data_audit.json',audit)
    write_json(out/'preprocessing.json', {'targets':TARGETS,'context':CONTEXT,
        'transforms': ['X * 1000', 'Y * 1000', 'Vx', 'Vy', 'log1p(Vz)'],
        'target_mean':ys.mean_.tolist(),'target_scale':ys.scale_.tolist(),
        'context_mean':cs.mean_.tolist(),'context_scale':cs.scale_.tolist()})
    return data, audit


class ConditionalGaussian(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(9,64), nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,10))
    def distribution(self,c):
        mu,ls=self.net(c).chunk(2,-1)
        return mu,ls.clamp(-4,3)
    def log_prob(self,y,c):
        mu,ls=self.distribution(c)
        return (-.5*((y-mu)*torch.exp(-ls))**2-ls-.5*math.log(2*math.pi)).sum(-1)
    def sample_y(self,c):
        mu,ls=self.distribution(c)
        return mu+torch.exp(ls)*torch.randn_like(mu)


def make_flow(conditional):
    flows=[]
    for _ in range(8):
        flows.extend([nf.flows.CoupledRationalQuadraticSpline(5,2,64,
            num_context_channels=9 if conditional else None),nf.flows.LULinearPermute(5)])
    return nf.ConditionalNormalizingFlow(nf.distributions.DiagGaussian(5,trainable=False),flows)


def log_prob(model,name,y,c):
    return model.log_prob(y, c if name!='unconditional_spline' else None)


def predict_nll(model,name,y,c,batch=1024):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(y),batch):
            result.append(-log_prob(model,name,torch.from_numpy(y[start:start+batch]),
                                    torch.from_numpy(c[start:start+batch])).numpy())
    return np.concatenate(result).astype('float64')


def fit_neural(name,seed,data,out,max_epochs,patience):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=ConditionalGaussian() if name=='conditional_gaussian' else make_flow(name=='conditional_spline')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_epochs,eta_min=1e-4)
    y,c,g=data['train'];yt,ct=torch.from_numpy(y),torch.from_numpy(c)
    wt=torch.tensor(balanced_weights(g),dtype=torch.float32)
    best=float('inf');best_state=None;stale=0;history=[];begin=time.perf_counter()
    for ep in range(max_epochs):
        model.train();order=torch.randperm(len(y));total=0.
        for ix in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            loss=(-log_prob(model,name,yt[ix],ct[ix])*wt[ix]).mean()
            if not torch.isfinite(loss): raise ValueError(f'Nonfinite training loss {name} seed {seed} epoch {ep}')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            total+=loss.item()*len(ix)
        scheduler.step()
        vy,vc,vg=data['validation'];vnll=predict_nll(model,name,vy,vc)
        val=float(group_means(vnll,vg).mean())
        history.append({'epoch':ep+1,'train_group_weighted_nll':total/len(y),'validation_group_nll':val})
        if val<best-1e-4:
            best=val;best_state=copy.deepcopy(model.state_dict());best_ep=ep+1;stale=0
        else: stale+=1
        if ep==0 or (ep+1)%5==0:
            print(f'{name} seed={seed} epoch={ep+1} validation_group_nll={val:.4f}',flush=True)
        if stale>=patience:break
    model.load_state_dict(best_state)
    prefix=f'{name}_seed{seed}'
    pd.DataFrame(history).to_csv(out/f'{prefix}_history.csv',index=False)
    torch.save({'state_dict':best_state,'model':name,'seed':seed,'best_epoch':best_ep},out/f'{prefix}.pt')
    return model,{'best_epoch':best_ep,'epochs_run':len(history),'best_validation_group_nll':best,
                  'training_seconds':time.perf_counter()-begin,'parameters':sum(p.numel() for p in model.parameters())}


def grouped_hist_diagnostics(y,samples,groups):
    weights=balanced_weights(groups);diagnostics={}
    for j,label in enumerate(TARGETS):
        # Shared edges include both datasets, including overflow: finite-sample marginal JS only.
        edges=np.linspace(min(y[:,j].min(),samples[:,j].min()),max(y[:,j].max(),samples[:,j].max()),51)
        a=np.histogram(y[:,j],bins=edges,weights=weights)[0].astype(float)
        b=np.histogram(samples[:,j],bins=edges,weights=weights)[0].astype(float)
        a=(a+1e-8)/(a.sum()+50e-8);b=(b+1e-8)/(b.sum()+50e-8);m=(a+b)/2
        js=.5*np.sum(a*np.log(a/m))+.5*np.sum(b*np.log(b/m))
        diagnostics[label]={'wasserstein_group_weighted':float(wasserstein_distance(y[:,j],samples[:,j],weights,weights)),
            'histogram_js_group_weighted':float(js),
            'ks_statistic_atom_weighted':float(ks_2samp(y[:,j],samples[:,j]).statistic)}
    diagnostics['mean_wasserstein_group_weighted']=float(np.mean([diagnostics[k]['wasserstein_group_weighted'] for k in TARGETS]))
    diagnostics['mean_histogram_js_group_weighted']=float(np.mean([diagnostics[k]['histogram_js_group_weighted'] for k in TARGETS]))
    corr_true=np.corrcoef(y,rowvar=False);corr_gen=np.corrcoef(samples,rowvar=False)
    diagnostics['correlation_matrix_frobenius_atom_weighted']=float(np.linalg.norm(corr_true-corr_gen))
    return diagnostics


def sample_neural(model,name,c,seed):
    torch.manual_seed(seed);result=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(c),512):
            context=torch.from_numpy(c[start:start+512])
            if name=='conditional_gaussian':s=model.sample_y(context)
            else:s,_=model.sample(len(context),context if name=='conditional_spline' else None)
            result.append(s.numpy())
    return np.concatenate(result)


def group_bootstrap_delta(a,b,seed=2026):
    """Paired resampling of simulation means; positive delta favors b."""
    common=a.index.intersection(b.index);d=(a.loc[common]-b.loc[common]).to_numpy()
    rng=np.random.default_rng(seed)
    boot=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(2000)])
    return {'baseline_minus_candidate_group_nll':float(d.mean()),
            'paired_simulation_bootstrap_95_ci':np.quantile(boot,[.025,.975]).tolist(),
            'simulations':len(d)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path('results'))
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--patience',type=int,default=6)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    begin=time.perf_counter();data,audit=prepare(args.data_dir,out)
    write_json(out/'run_config.json',{'seeds':args.seeds,'epochs':args.epochs,'patience':args.patience,
        'split_seed':42,'secondary_split_seed':43,'selection_metric':'validation mean of per-simulation NLL',
        'training_weighting':'equal weight per simulation','flow_blocks':8,'hidden_layers':2,'hidden_units':64,
        'batch_size':512,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'gradient_clip':5.,
        'python':platform.python_version(),'packages':{k:importlib.metadata.version(k) for k in
            ['torch','normflows','numpy','pandas','scipy','scikit-learn','matplotlib']},'device':'cpu','threads':2})
    print(json.dumps(audit),flush=True)
    y,c,g=data['test'];tr_y,_,tr_g=data['train'];weights=balanced_weights(tr_g)
    rows=[];all_groups={};samples_for_plot={};group_frames=[]
    def record(name,seed,nll,samples,extra):
        gm=group_means(nll,g);all_groups[(name,seed)]=gm
        frame=gm.rename('nll').reset_index();frame['model']=name;frame['seed']=seed;group_frames.append(frame)
        stats=grouped_hist_diagnostics(y,samples,g)
        write_json(out/f'{name}_seed{seed}_fidelity.json',stats)
        rows.append({'model':name,'seed':seed,'test_group_nll':float(gm.mean()),'test_atom_nll':float(nll.mean()),
                     'mean_wasserstein_group_weighted':stats['mean_wasserstein_group_weighted'],
                     'mean_histogram_js_group_weighted':stats['mean_histogram_js_group_weighted'],**extra})
        if seed==args.seeds[0] or seed==0:samples_for_plot[name]=samples
        print('RESULT '+json.dumps(rows[-1]),flush=True)
    # Analytic full-covariance Gaussian, fit with equal simulation weights.
    mean=np.average(tr_y,axis=0,weights=weights)
    cov=np.cov(tr_y,rowvar=False,aweights=weights,ddof=0)+np.eye(5)*1e-5
    inv=np.linalg.inv(cov);logdet=np.linalg.slogdet(cov)[1]
    nll=.5*(np.einsum('ni,ij,nj->n',y-mean,inv,y-mean)+logdet+5*np.log(2*np.pi))
    record('unconditional_gaussian',0,nll,np.random.default_rng(100).multivariate_normal(mean,cov,len(y)),{})
    # Group-balanced resampling gives sklearn's unweighted GMM a balanced training population.
    rng=np.random.default_rng(100);ix=rng.choice(len(tr_y),size=50000,replace=True,p=weights/weights.sum())
    candidates=[]
    for k in [4,8]:
        mix=GaussianMixture(k,covariance_type='full',reg_covar=1e-4,max_iter=150,random_state=100,n_init=1).fit(tr_y[ix])
        vy,_,vg=data['validation'];score=float(group_means(-mix.score_samples(vy),vg).mean())
        candidates.append((score,k,mix))
    score,k,mix=min(candidates,key=lambda z:z[0]);s,_=mix.sample(len(y))
    # sklearn samples grouped by component; shuffle before pairing with group weights.
    s=s[rng.permutation(len(s))]
    record('unconditional_gmm',0,-mix.score_samples(y),s,{'components':k,'best_validation_group_nll':score})
    np.savez(out/'unconditional_gmm.npz',weights=mix.weights_,means=mix.means_,covariances=mix.covariances_)
    for seed in args.seeds:
        for name in ['conditional_gaussian','unconditional_spline','conditional_spline']:
            model,extra=fit_neural(name,seed,data,out,args.epochs,args.patience)
            nll=predict_nll(model,name,y,c)
            if name.startswith('conditional'):
                # Permute complete context vectors at SIMULATION level, not individual columns.
                ids,first=np.unique(g,return_index=True);rng=np.random.default_rng(seed+800)
                shuffled=c[first][rng.permutation(len(first))]
                permuted=shuffled[np.searchsorted(ids,g)]
                perm_nll=predict_nll(model,name,y,permuted)
                extra['permuted_context_test_group_nll']=float(group_means(perm_nll,g).mean())
                extra['context_permutation_nll_penalty']=extra['permuted_context_test_group_nll']-float(group_means(nll,g).mean())
            samples=sample_neural(model,name,c,seed+1000)
            record(name,seed,nll,samples,extra)
    results=pd.DataFrame(rows);results.to_csv(out/'metrics.csv',index=False)
    pd.concat(group_frames,ignore_index=True).to_csv(out/'per_simulation_nll.csv',index=False)
    summary=results.groupby('model').test_group_nll.agg(['mean','std','count']).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    comparisons={}
    for baseline in ['unconditional_gaussian','unconditional_gmm','conditional_gaussian','unconditional_spline']:
        base=[v for (n,_),v in all_groups.items() if n==baseline]
        cand=[v for (n,_),v in all_groups.items() if n=='conditional_spline']
        # Seed-averaged per-group loss; CI covers simulation sampling, not optimization uncertainty.
        comparisons[baseline]=group_bootstrap_delta(pd.concat(base,axis=1).mean(axis=1),pd.concat(cand,axis=1).mean(axis=1))
    write_json(out/'comparisons.json',comparisons)
    plot_results(out,results,samples_for_plot,y,g)
    write_json(out/'completion.json',{'completed':True,'wall_seconds':time.perf_counter()-begin})


def plot_results(out,results,samples,y,g):
    fig,axes=plt.subplots(1,5,figsize=(17,3.3));weights=balanced_weights(g)
    for j,ax in enumerate(axes):
        bounds=np.quantile(y[:,j],[.005,.995]);edges=np.linspace(*bounds,51)
        ax.hist(y[:,j],bins=edges,density=True,weights=weights,histtype='step',color='black',label='Simulation')
        for name in ['unconditional_gmm','unconditional_spline','conditional_spline']:
            ax.hist(samples[name][:,j],bins=edges,density=True,weights=weights,histtype='step',label=name.replace('_',' '),alpha=.8)
        ax.set_title(['X (mm)','Y (mm)','Vx','Vy','log1p(Vz)'][j]+' standardized');ax.set_xlabel('Training-standardized value')
    axes[0].legend(fontsize=6);fig.suptitle('Unseen simulations: group-weighted marginals (central 99%; full tails included in metrics)')
    fig.tight_layout();fig.savefig(out/'marginal_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5));names=list(results.model.unique())
    for i,name in enumerate(names):
        values=results[results.model==name].test_group_nll
        ax.scatter([i]*len(values),values,label=name)
    ax.set_xticks(range(len(names)),[n.replace('_','\n') for n in names]);ax.set_ylabel('Test NLL, equal weight per simulation (lower is better)')
    ax.set_title('Group-disjoint test set; neural models use three fixed seeds' if len(results)>5 else 'Group-disjoint test set')
    fig.tight_layout();fig.savefig(out/'benchmark_comparison.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for name in ['conditional_gaussian','unconditional_spline','conditional_spline']:
        path=next(out.glob(f'{name}_seed*_history.csv'));h=pd.read_csv(path)
        ax.plot(h.epoch,h.validation_group_nll,label=name.replace('_',' '))
    ax.set_xlabel('Epoch');ax.set_ylabel('Validation group NLL');ax.legend();fig.tight_layout()
    fig.savefig(out/'validation_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
