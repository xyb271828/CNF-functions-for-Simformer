"""Independent-coupling conditional flow matching, with explicit ODE evaluation.

Path: x_t=(1-t)z+t*x, z~N(0,I), u_t=x-z. Context is fixed, not transported.
No minibatch OT, diffusion model, or likelihood training is claimed.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from benchmark import (balanced_weights, group_means, group_bootstrap_delta,
                       grouped_hist_diagnostics, prepare, sha256, write_json, make_flow)


class VelocityMLP(nn.Module):
    def __init__(self, hidden=224):
        super().__init__()
        self.register_buffer('frequencies',torch.arange(1,5,dtype=torch.float32)*math.pi)
        self.net=nn.Sequential(nn.Linear(23,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,5))

    def forward(self,x,t,context):
        t=torch.as_tensor(t,dtype=x.dtype,device=x.device)
        if t.numel()==1:t=t.expand(len(x))
        t=t.reshape(-1,1)
        angles=t*self.frequencies
        return self.net(torch.cat([x,context,t,angles.sin(),angles.cos()],dim=-1))


def exact_velocity_divergence(model,x,t,context):
    """Trace of the five-dimensional spatial Jacobian. No stochastic estimator."""
    with torch.enable_grad():
        x=x.detach().requires_grad_(True)
        v=model(x,t,context)
        divergence=torch.zeros(len(x),dtype=x.dtype,device=x.device)
        for j in range(x.shape[1]):
            derivative=torch.autograd.grad(v[:,j].sum(),x,retain_graph=j<x.shape[1]-1)[0]
            divergence=divergence+derivative[:,j]
    return v.detach(),divergence.detach()


@torch.no_grad()
def integrate_samples(model,z,context,steps=32,method='rk4'):
    if steps<1:raise ValueError('steps must be positive')
    x=z.clone();h=1./steps
    for k in range(steps):
        t=k*h;k1=model(x,t,context)
        if method=='euler':x=x+h*k1
        elif method=='rk4':
            k2=model(x+.5*h*k1,t+.5*h,context)
            k3=model(x+.5*h*k2,t+.5*h,context)
            k4=model(x+h*k3,t+h,context)
            x=x+h*(k1+2*k2+2*k3+k4)/6.
        else:raise ValueError(method)
    return x


def nll_ode(model,y,context,steps=128,batch=256):
    """Reverse ODE with RK4; log p1=log p0(z)+integral_1^0 div(v) dt."""
    if steps<1:raise ValueError('steps must be positive')
    model.eval();result=[];h=-1./steps
    for start in range(0,len(y),batch):
        x=torch.as_tensor(y[start:start+batch]).detach().clone()
        c=torch.as_tensor(context[start:start+batch],dtype=x.dtype,device=x.device)
        integral=torch.zeros(len(x),dtype=x.dtype,device=x.device)
        for k in range(steps):
            t=1+k*h
            k1,d1=exact_velocity_divergence(model,x,t,c)
            k2,d2=exact_velocity_divergence(model,x+.5*h*k1,t+.5*h,c)
            k3,d3=exact_velocity_divergence(model,x+.5*h*k2,t+.5*h,c)
            k4,d4=exact_velocity_divergence(model,x+h*k3,t+h,c)
            x=x+h*(k1+2*k2+2*k3+k4)/6.
            integral=integral+h*(d1+2*d2+2*d3+d4)/6.
        nll=.5*(x*x).sum(-1)+.5*x.shape[1]*math.log(2*math.pi)-integral
        if not torch.isfinite(nll).all():raise ValueError('Nonfinite ODE likelihood')
        result.append(nll.cpu().numpy())
    return np.concatenate(result).astype('float64')


def sample_all(model,c,seed,steps=32,method='rk4',batch=512):
    rng=torch.Generator().manual_seed(seed)
    z=torch.randn(len(c),5,generator=rng);parts=[]
    model.eval()
    for start in range(0,len(c),batch):
        parts.append(integrate_samples(model,z[start:start+batch],torch.from_numpy(c[start:start+batch]),steps,method).numpy())
    return np.concatenate(parts)


def validation_tuples(y,seed=7001,draws=4):
    rng=torch.Generator().manual_seed(seed);y=torch.from_numpy(y)
    return [(torch.randn(y.shape,generator=rng),torch.rand(len(y),1,generator=rng)) for _ in range(draws)]


def train_one(seed,data,out,epochs=30,patience=6):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=VelocityMLP();optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=epochs,eta_min=.0001)
    y,c,g=data['train'];yt=torch.from_numpy(y);ct=torch.from_numpy(c)
    w=torch.tensor(balanced_weights(g),dtype=torch.float32)
    vy,vc,vg=data['validation'];vyt=torch.from_numpy(vy);vct=torch.from_numpy(vc)
    vw=torch.tensor(balanced_weights(vg),dtype=torch.float32)
    draws=validation_tuples(vy);history=[];best=float('inf');stale=0
    begin=time.perf_counter()
    for ep in range(epochs):
        model.train();total=0.
        for ix in torch.randperm(len(y)).split(512):
            z=torch.randn_like(yt[ix]);t=torch.rand(len(ix),1)
            xt=(1-t)*z+t*yt[ix];target=yt[ix]-z
            loss=(((model(xt,t,ct[ix])-target)**2).mean(-1)*w[ix]).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite FM loss')
            optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step()
            total+=loss.item()*len(ix)
        scheduler.step();model.eval();validation=0.
        with torch.no_grad():
            for z,t in draws:
                for ix in torch.arange(len(vy)).split(1024):
                    pred=model((1-t[ix])*z[ix]+t[ix]*vyt[ix],t[ix],vct[ix])
                    validation+=float((((pred-(vyt[ix]-z[ix]))**2).mean(-1)*vw[ix]).sum())/(len(vy)*len(draws))
        history.append({'epoch':ep+1,'training_cfm_mse':total/len(y),'validation_fixed_mc_cfm_mse':validation})
        if validation<best-1e-5:
            best=validation;best_epoch=ep+1;best_state=copy.deepcopy(model.state_dict());stale=0
        else:stale+=1
        if ep==0 or (ep+1)%5==0:print(f'FM seed={seed} epoch={ep+1} val_CFM_MSE={validation:.6f}',flush=True)
        if stale>=patience:break
    model.load_state_dict(best_state)
    torch.save({'state_dict':best_state,'seed':seed,'best_epoch':best_epoch,'hidden':224},out/f'flow_matching_seed{seed}.pt')
    pd.DataFrame(history).to_csv(out/f'flow_matching_seed{seed}_history.csv',index=False)
    return model,{'best_epoch':best_epoch,'epochs_run':len(history),'best_validation_cfm_mse':best,
        'training_seconds':time.perf_counter()-begin,'parameters':sum(p.numel() for p in model.parameters())}


def cpu_timing(model,c,seed):
    ctxt=torch.from_numpy(np.repeat(c[:1],512,axis=0));torch.manual_seed(seed)
    z=torch.randn(512,5);rows=[]
    for method,steps in [('euler',8),('euler',16),('euler',32),('euler',64),('rk4',8),('rk4',16),('rk4',32)]:
        times=[]
        for i in range(13):
            begin=time.perf_counter();integrate_samples(model,z,ctxt,steps,method)
            if i>=3:times.append((time.perf_counter()-begin)*1000)
        rows.append({'seed':seed,'model':'flow_matching','solver':method,'steps':steps,
                     'velocity_evaluations':steps*(4 if method=='rk4' else 1),
                     'batch_size':512,'p50_ms':float(np.quantile(times,.5)),
                     'p95_ms':float(np.quantile(times,.95))})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--baseline-results',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path('results_flow_matching'))
    ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33]);ap.add_argument('--epochs',type=int,default=30)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    begin=time.perf_counter();data,audit=prepare(args.data_dir,out)
    for name in ['split_manifest.csv','preprocessing.json']:
        if sha256(out/name)!=sha256(args.baseline_results/name):raise AssertionError(f'Baseline mismatch: {name}')
    write_json(out/'protocol.json',{'method':'independent-coupling conditional flow matching',
        'path':'x_t=(1-t)z+t*x; z~N(0,I)','target_velocity':'x-z','condition_dimensions':9,
        'model_hidden':224,'model_hidden_layers':4,'time_encoding':'t plus sin/cos of pi*[1,2,3,4]*t',
        'selection_metric':'validation fixed-Monte-Carlo flow-matching MSE (4 shared noise/time draws)',
        'selection_caveat':'Different training objectives; not identical NLL checkpoint selection to spline baseline',
        'likelihood':'reverse RK4 ODE, exact five-dimensional divergence, 128 steps, float32',
        'likelihood_convergence_check':'64 versus 128 steps on a fixed validation subset, before test NLL',
        'primary_sampling':'RK4,32 steps; solver sweep is descriptive, not test-selected',
        'epochs':args.epochs,'patience':6,'seeds':args.seeds,'threads':2,'device':'cpu',
        'batch_size':512,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'gradient_clip':5.,
        'split_manifest_matches_baseline':True,'preprocessing_matches_baseline':True,
        'comparison_status':'Exploratory extension on the existing shared test split; not a new untouched confirmation set',
        'sources':['https://arxiv.org/abs/2210.02747',
                   'https://facebookresearch.github.io/flow_matching/generated/flow_matching.solver.ODESolver.html']})
    y,c,g=data['test'];metrics=[];groups=[];solver_metrics=[];timings=[];convergence=[]
    for seed in args.seeds:
        model,extra=train_one(seed,data,out,args.epochs)
        print(f'FM seed={seed}: evaluating ODE numerical convergence',flush=True)
        vy,vc,vg=data['validation'];_,first=np.unique(vg,return_index=True)
        subset=first[np.random.default_rng(9001).permutation(len(first))[:256]]
        n64=nll_ode(model,vy[subset],vc[subset],64)
        n128=nll_ode(model,vy[subset],vc[subset],128)
        conv={'seed':seed,'validation_simulations':len(subset),
              'mean_abs_nll_64_vs_128':float(np.abs(n64-n128).mean()),
              'max_abs_nll_64_vs_128':float(np.abs(n64-n128).max()),
              'mean_signed_nll_64_minus_128':float((n64-n128).mean())}
        convergence.append(conv);print('CONVERGENCE '+json.dumps(conv),flush=True)
        # Escalation checks numerical error only, using validation, never test performance.
        steps=128
        if conv['mean_abs_nll_64_vs_128']>.005 or conv['max_abs_nll_64_vs_128']>.1:
            raise RuntimeError('ODE likelihood has not converged; refine solver before reporting test metrics')
        print(f'FM seed={seed}: evaluating held-out likelihood',flush=True)
        then=time.perf_counter();nll=nll_ode(model,y,c,steps)
        extra['test_likelihood_seconds']=time.perf_counter()-then
        gm=group_means(nll,g);frame=gm.rename('nll').reset_index();frame['seed']=seed;groups.append(frame)
        samples=sample_all(model,c,seed+1000,32,'rk4')
        fidelity=grouped_hist_diagnostics(y,samples,g)
        write_json(out/f'flow_matching_seed{seed}_fidelity.json',fidelity)
        np.savez_compressed(out/f'flow_matching_seed{seed}_generated.npz',generated_standardized=samples,
                            target_standardized=y,simulation_ids=g)
        row={'model':'conditional_flow_matching','seed':seed,'test_group_nll':float(gm.mean()),
             'test_atom_nll':float(nll.mean()),'mean_wasserstein_group_weighted':fidelity['mean_wasserstein_group_weighted'],
             'mean_histogram_js_group_weighted':fidelity['mean_histogram_js_group_weighted'],**extra}
        metrics.append(row);print('RESULT '+json.dumps(row),flush=True)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        # Common random numbers for all solvers; never use these results to retune the model.
        for method,steps in [('euler',8),('euler',16),('euler',32),('euler',64),('rk4',8),('rk4',16),('rk4',32)]:
            s=samples if (method,steps)==('rk4',32) else sample_all(model,c,seed+1000,steps,method)
            f=grouped_hist_diagnostics(y,s,g)
            solver_metrics.append({'seed':seed,'solver':method,'steps':steps,
                'velocity_evaluations':steps*(4 if method=='rk4' else 1),
                'mean_wasserstein_group_weighted':f['mean_wasserstein_group_weighted'],
                'mean_histogram_js_group_weighted':f['mean_histogram_js_group_weighted'],
                'rms_difference_from_rk4_32':float(np.sqrt(np.mean((s-samples)**2)))})
        timings.extend(cpu_timing(model,c,seed))
        # Remeasure spline in the same process/hardware/thread setting, same batch size.
        spline=make_flow(True);ckpt=torch.load(args.baseline_results/f'conditional_spline_seed{seed}.pt',weights_only=True)
        spline.load_state_dict(ckpt['state_dict']);spline.eval();ctxt=torch.from_numpy(np.repeat(c[:1],512,axis=0))
        ts=[]
        with torch.no_grad():
            for i in range(13):
                t0=time.perf_counter();spline.sample(512,ctxt)
                if i>=3:ts.append((time.perf_counter()-t0)*1000)
        timings.append({'seed':seed,'model':'conditional_spline','solver':'direct','steps':0,
                        'velocity_evaluations':0,'batch_size':512,
                        'p50_ms':float(np.quantile(ts,.5)),'p95_ms':float(np.quantile(ts,.95))})
        write_json(out/'convergence.json',convergence)
        pd.DataFrame(solver_metrics).to_csv(out/'solver_quality.csv',index=False)
        pd.DataFrame(timings).to_csv(out/'cpu_sampling_timing.csv',index=False)
    grouped=pd.concat(groups,ignore_index=True);grouped.to_csv(out/'per_simulation_nll.csv',index=False)
    fm=grouped.groupby('group').nll.mean()
    old=pd.read_csv(args.baseline_results/'per_simulation_nll.csv')
    comparisons={}
    for name in ['conditional_spline','unconditional_spline','conditional_gaussian','unconditional_gmm']:
        base=old[old.model==name].groupby('group').nll.mean()
        comparisons[name]=group_bootstrap_delta(base,fm)
    write_json(out/'comparisons.json',comparisons)
    pd.DataFrame(metrics).groupby('model').test_group_nll.agg(['mean','std','count']).to_csv(out/'summary.csv')
    write_json(out/'completion.json',{'completed':True,'wall_seconds':time.perf_counter()-begin})


if __name__=='__main__':main()
