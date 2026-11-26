"""Cold-atom likelihood-specialized Simformer adaptation, official JAX backbone.

The 14 scalar nodes are [nine controls, five atomic states]. The nine controls
are observed and never diffused. No arbitrary-condition posterior claim is made.
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['JAX_PLATFORMS']='cpu'
if hasattr(os,'sched_getaffinity'):
    _available_cpus=sorted(os.sched_getaffinity(0))
    _cpu_offset=int(os.environ.get('SIMFORMER_CPU_OFFSET','0'))
    os.sched_setaffinity(0,_available_cpus[_cpu_offset:_cpu_offset+2])

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from vendor_simformer.minimal_model import get_model
from benchmark import (prepare,sha256,write_json,balanced_weights,group_means,
                       grouped_hist_diagnostics,group_bootstrap_delta)

SIGMA_MIN=1e-3
SIGMA_MAX=15.
T_MIN=1e-2
LOG_RATIO=math.log(SIGMA_MAX/SIGMA_MIN)
MODEL_CONFIG=dict(value_embedding=20,node_id_embedding=20,condition_embedding=10,
    node_embedding_init_std=3.,condition_embedding_init_std=.5,time_embedding_dim=64,
    num_heads=2,num_layers=2,attn_size=10,widening_factor=3,num_hidden_layers=1,
    output_scale_floor=.01,source='original minimal-example model')


def sigma(t):return SIGMA_MIN*jnp.exp(LOG_RATIO*jnp.asarray(t))


def make_model():
    init,apply=get_model()
    ids=jnp.arange(14)
    mask=jnp.r_[jnp.ones(9,dtype=bool),jnp.zeros(5,dtype=bool)][None,:]
    def score(params,t,y,c):
        t=jnp.broadcast_to(jnp.asarray(t), (len(y),))
        joint=jnp.concatenate([c,y],-1)[...,None]
        return apply(params,t,joint,ids,mask)[:,9:,0]
    return init,score,ids,mask


INIT,SCORE,IDS,MASK=make_model()


def loss_vector(params,y,c,t,noise):
    sd=sigma(t)[:,None]
    score=SCORE(params,t,y+sd*noise,c)
    # Official VE MLE weighting g(t)^2 * sum((score-score_target)^2).
    return 2*LOG_RATIO*jnp.sum((sd*score+noise)**2,axis=-1)


def velocity(params,t,y,c):
    return -LOG_RATIO*sigma(t)**2*SCORE(params,t,y,c)


def velocity_and_divergence(params,t,y,c):
    # Each row is independent; the directional trace is only over the five
    # generated coordinates, never the nine observed control variables.
    v,linear=jax.linearize(lambda state:velocity(params,t,state,c),y)
    basis=jnp.eye(5,dtype=y.dtype)
    tangent=jax.vmap(lambda e:linear(jnp.broadcast_to(e,y.shape)))(basis)
    div=jnp.einsum('ibi->b',tangent)
    return v,div


def rk4_integrate(field,x,c,start,end,steps):
    h=(end-start)/steps
    def body(i,x):
        t=start+i*h
        a=field(t,x,c);b=field(t+h/2,x+h*a/2,c)
        d=field(t+h/2,x+h*b/2,c);e=field(t+h,x+h*d,c)
        return x+h*(a+2*b+2*d+e)/6
    return jax.lax.fori_loop(0,steps,body,x)


def integrate_logdensity(field_and_div,x,c,start,end,steps):
    h=(end-start)/steps
    def body(i,carry):
        x,acc=carry;t=start+i*h
        a,da=field_and_div(t,x,c)
        b,db=field_and_div(t+h/2,x+h*a/2,c)
        d,dd=field_and_div(t+h/2,x+h*b/2,c)
        e,de=field_and_div(t+h,x+h*d,c)
        return x+h*(a+2*b+2*d+e)/6,acc+h*(da+2*db+2*dd+de)/6
    return jax.lax.fori_loop(0,steps,body,(x,jnp.zeros(len(x),dtype=x.dtype)))


@functools.partial(jax.jit,static_argnames=['steps'])
def likelihood_batch(params,y,c,terminal_mean,terminal_std,steps=128):
    z,integral=integrate_logdensity(lambda t,x,c:velocity_and_divergence(params,t,x,c),y,c,T_MIN,1.,steps)
    prior_nll=.5*jnp.sum(((z-terminal_mean)/terminal_std)**2,axis=-1)
    prior_nll+=jnp.log(terminal_std).sum()+2.5*math.log(2*math.pi)
    return prior_nll-integral


@functools.partial(jax.jit,static_argnames=['steps','method'])
def sample_batch(params,c,key,terminal_mean,terminal_std,steps=128,method='ode'):
    z=terminal_mean+terminal_std*jax.random.normal(key,(len(c),5))
    if method=='ode':
        return rk4_integrate(lambda t,x,c:velocity(params,t,x,c),z,c,1.,T_MIN,steps)
    h=(T_MIN-1.)/steps
    def body(i,carry):
        x,key=carry;t=1+i*h;key,noise_key=jax.random.split(key)
        g=sigma(t)*math.sqrt(2*LOG_RATIO)
        drift=-g*g*SCORE(params,t,x,c)
        x=x+h*drift+g*math.sqrt(-h)*jax.random.normal(noise_key,x.shape)
        return x,key
    return jax.lax.fori_loop(0,steps,body,(z,key))[0]


def batches_padded(y,c,size=256):
    for start in range(0,len(y),size):
        n=min(size,len(y)-start)
        yy=y[start:start+n];cc=c[start:start+n]
        if n<size:
            yy=np.concatenate([yy,np.repeat(yy[:1],size-n,axis=0)])
            cc=np.concatenate([cc,np.repeat(cc[:1],size-n,axis=0)])
        yield yy,cc,n


def compute_nll(params,y,c,mean,std,steps=128):
    parts=[]
    for yy,cc,n in batches_padded(y,c):
        value=np.asarray(likelihood_batch(params,jnp.asarray(yy),jnp.asarray(cc),mean,std,steps))[:n]
        if not np.isfinite(value).all():raise ValueError('Nonfinite probability-flow likelihood')
        parts.append(value)
    return np.concatenate(parts).astype(float)


def generate(params,c,seed,mean,std,steps=128,method='ode'):
    output=[];key=jax.random.PRNGKey(seed)
    for _,cc,n in batches_padded(np.zeros((len(c),5),dtype='float32'),c,512):
        key,subkey=jax.random.split(key)
        value=np.asarray(sample_batch(params,jnp.asarray(cc),subkey,mean,std,steps,method))[:n]
        if not np.isfinite(value).all():raise ValueError('Nonfinite generated sample')
        output.append(value)
    return np.concatenate(output)


def fit(seed,data,out,max_epochs=30,patience=6):
    key=jax.random.PRNGKey(seed);key,initkey=jax.random.split(key)
    params=INIT(initkey,jnp.ones(2),jnp.ones((2,14,1)),IDS,MASK)
    param_count=sum(x.size for x in jax.tree_util.tree_leaves(params))
    y,c,g=data['train'];n=len(y);epoch_steps=math.ceil(n/512)
    weights=balanced_weights(g).astype('float32')
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(
        optax.cosine_decay_schedule(.001,max_epochs*epoch_steps,alpha=.1),weight_decay=.0001))
    optstate=optimizer.init(params)
    @jax.jit
    def update(params,state,key,y,c,w,count):
        kt,kn=jax.random.split(key)
        t=jax.random.uniform(kt,(len(y),),minval=T_MIN,maxval=1.)
        noise=jax.random.normal(kn,y.shape)
        def objective(p):return jnp.sum(loss_vector(p,y,c,t,noise)*w)/count
        loss,grad=jax.value_and_grad(objective)(params)
        updates,state=optimizer.update(grad,state,params)
        return optax.apply_updates(params,updates),state,loss
    @jax.jit
    def validation(params,y,c,t,noise,w,count):return jnp.sum(loss_vector(params,y,c,t,noise)*w)/count
    # Compile without changing the actual training state or training PRNG.
    count=min(n,512);yy=jnp.asarray(y[:512]);cc=jnp.asarray(c[:512]);ww=jnp.asarray(weights[:512])
    start_compile=time.perf_counter()
    warm=update(params,optstate,jax.random.PRNGKey(701),yy,cc,ww,jnp.float32(count))
    jax.block_until_ready(warm)
    compile_seconds=time.perf_counter()-start_compile
    vy,vc,vg=data['validation'];vw=balanced_weights(vg).astype('float32')
    rng=np.random.default_rng(7001)
    val_draws=[(rng.uniform(T_MIN,1.,len(vy)).astype('float32'),rng.standard_normal(vy.shape).astype('float32')) for _ in range(4)]
    indices=np.arange(n);order_rng=np.random.default_rng(seed);history=[];best=float('inf');stale=0
    begin=time.perf_counter()
    for ep in range(max_epochs):
        order=order_rng.permutation(indices);total=0.
        for pos in range(0,n,512):
            ix=order[pos:pos+512];valid=len(ix)
            if valid<512:ix=np.r_[ix,np.repeat(ix[0],512-valid)]
            w=weights[ix].copy();w[valid:]=0
            key,subkey=jax.random.split(key)
            params,optstate,loss=update(params,optstate,subkey,jnp.asarray(y[ix]),jnp.asarray(c[ix]),jnp.asarray(w),jnp.float32(valid))
            value=float(loss)
            if not np.isfinite(value):raise ValueError('Nonfinite training loss')
            total+=value*valid
        val=0.
        for t,noise in val_draws:
            for pos in range(0,len(vy),512):
                valid=min(512,len(vy)-pos);ix=np.arange(pos,pos+valid)
                if valid<512:ix=np.r_[ix,np.repeat(ix[0],512-valid)]
                w=vw[ix].copy();w[valid:]=0
                val+=float(validation(params,jnp.asarray(vy[ix]),jnp.asarray(vc[ix]),jnp.asarray(t[ix]),jnp.asarray(noise[ix]),jnp.asarray(w),jnp.float32(len(vy)*4)))
        history.append({'epoch':ep+1,'train_group_dsm':total/n,'validation_fixed_mc_group_dsm':val,'optimizer_steps':(ep+1)*epoch_steps})
        if val<best-1e-4:
            best=val;best_epoch=ep+1;best_params=jax.tree_util.tree_map(lambda x:np.asarray(x).copy(),params);stale=0
        else:stale+=1
        if ep==0 or (ep+1)%5==0:print(f'Simformer seed={seed} epoch={ep+1} validation_DSM={val:.6f} wall={time.perf_counter()-begin:.1f}s',flush=True)
        if stale>=patience:break
    pd.DataFrame(history).to_csv(out/f'simformer_seed{seed}_history.csv',index=False)
    with (out/f'simformer_seed{seed}.pkl').open('wb') as f:
        pickle.dump({'params':best_params,'seed':seed,'best_epoch':best_epoch,'model_config':MODEL_CONFIG},f)
    return jax.tree_util.tree_map(jnp.asarray,best_params),{'parameters':int(param_count),'best_epoch':best_epoch,
        'epochs_run':len(history),'optimizer_steps':len(history)*epoch_steps,'best_validation_dsm':best,
        'training_seconds':time.perf_counter()-begin,'initial_update_compilation_seconds':compile_seconds}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--baseline-dir',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--seeds',type=int,nargs='+',default=[11,22,33])
    ap.add_argument('--train-only',action='store_true');args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    begin=time.perf_counter();data,audit=prepare(args.data_dir,out)
    for f in ['split_manifest.csv','preprocessing.json']:
        if sha256(out/f)!=sha256(args.baseline_dir/f):raise ValueError(f'Data mismatch: {f}')
    train_y,_,train_g=data['train'];w=balanced_weights(train_g)
    mean=jnp.asarray(np.average(train_y,axis=0,weights=w),dtype=jnp.float32)
    std=jnp.sqrt(jnp.asarray(np.average((train_y-np.asarray(mean))**2,axis=0,weights=w))+SIGMA_MAX**2)
    provenance=json.loads((Path(__file__).parent/'vendor_simformer/PROVENANCE.json').read_text())
    write_json(out/'protocol.json',{'model':'Simformer, original minimal-example JAX backbone; likelihood-specialized cold-atom adaptation',
        'upstream_commit':provenance['commit'],'model_config':MODEL_CONFIG,'sde':'VE','sigma_min':SIGMA_MIN,
        'sigma_max':SIGMA_MAX,'minimum_time':T_MIN,'condition_mask':'nine controls observed, five states generated',
        'arbitrary_condition_training':False,'edge_mask':'dense; no marginalization augmentation',
        'selection':'validation MLE-weighted DSM; four fixed noise/time draws',
        'train_weighting':'equal total weight per simulation','batch_size':512,'max_epochs':args.epochs,
        'epoch_steps':math.ceil(len(train_y)/512),'seeds':args.seeds,'patience':6,
        'optimizer':'AdamW, lr .001 cosine to .0001, weight_decay .0001, global gradient clip 5',
        'terminal_mean':np.asarray(mean).tolist(),'terminal_std':np.asarray(std).tolist(),
        'terminal_prior':'training group-weighted moments plus VE terminal variance, diagonal Gaussian approximation',
        'data_manifest_and_scaler_match_baselines':True,'device':'CPU','CPU_affinity':sorted(os.sched_getaffinity(0)),
        'packages':{m:importlib.metadata.version(m) for m in ['jax','jaxlib','dm-haiku','optax','numpy','torch']},
        'likelihood':'probability-flow ODE, exact 5D divergence, numerical RK4; fixed t_min endpoint',
        'comparison_scope':'same retained subset and shared exploratory holdout; different architecture/framework/objective',
        'historical_toy_steps':75000,'historical_toy_not_reused_as_cold_atom_result':True})
    y,c,g=data['test'];metrics=[];all_groups=[];convergence=[];timings=[]
    for seed in args.seeds:
        params,extra=fit(seed,data,out,args.epochs)
        if args.train_only:continue
        print(f'Simformer seed={seed}: likelihood convergence check',flush=True)
        vy,vc,vg=data['validation'];_,first=np.unique(vg,return_index=True)
        selected=first[np.random.default_rng(9001).permutation(len(first))[:256]]
        n64=compute_nll(params,vy[selected],vc[selected],mean,std,64)
        n128=compute_nll(params,vy[selected],vc[selected],mean,std,128)
        diff=np.abs(n64-n128)
        conv={'seed':seed,'steps_low':64,'steps_high':128,'mean_abs_nll_difference':float(diff.mean()),'max_abs_nll_difference':float(diff.max())}
        steps=128
        if diff.mean()>.005 or diff.max()>.1:
            n256=compute_nll(params,vy[selected],vc[selected],mean,std,256);diff=np.abs(n128-n256)
            conv.update({'steps_low':128,'steps_high':256,'mean_abs_nll_difference':float(diff.mean()),'max_abs_nll_difference':float(diff.max())});steps=256
        if diff.mean()>.005 or diff.max()>.1:raise RuntimeError('Likelihood integration is not converged; do not report test NLL')
        convergence.append(conv);print('CONVERGENCE '+json.dumps(conv),flush=True)
        print(f'Simformer seed={seed}: evaluating held-out NLL',flush=True)
        start=time.perf_counter();nll=compute_nll(params,y,c,mean,std,steps)
        extra['test_likelihood_seconds']=time.perf_counter()-start
        gm=group_means(nll,g);frame=gm.rename('nll').reset_index();frame['seed']=seed;all_groups.append(frame)
        samples=generate(params,c,seed+1000,mean,std,128,'ode')
        fidelity=grouped_hist_diagnostics(y,samples,g)
        write_json(out/f'simformer_seed{seed}_ode_fidelity.json',fidelity)
        np.savez_compressed(out/f'simformer_seed{seed}_generated.npz',generated_standardized=samples,target_standardized=y,simulation_ids=g)
        sde_samples=generate(params,c,seed+1000,mean,std,500,'sde')
        sde_fidelity=grouped_hist_diagnostics(y,sde_samples,g)
        write_json(out/f'simformer_seed{seed}_sde500_fidelity.json',sde_fidelity)
        row={'model':'simformer_likelihood_adaptation','seed':seed,'test_group_nll':float(gm.mean()),
            'test_atom_nll':float(nll.mean()),'likelihood_rk4_steps':steps,
            'ode_mean_wasserstein':fidelity['mean_wasserstein_group_weighted'],
            'sde500_mean_wasserstein':sde_fidelity['mean_wasserstein_group_weighted'],**extra}
        metrics.append(row);print('RESULT '+json.dumps(row),flush=True)
        pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
        context=jnp.asarray(np.repeat(c[:1],512,axis=0))
        for method,steps in [('ode',32),('ode',128),('sde',100),('sde',500)]:
            ts=[]
            for i in range(8):
                key=jax.random.PRNGKey(seed+i)
                start=time.perf_counter();value=sample_batch(params,context,key,mean,std,steps,method);value.block_until_ready()
                if i>=3:ts.append((time.perf_counter()-start)*1000)
            timings.append({'seed':seed,'method':method,'steps':steps,'batch_size':512,
                'p50_ms':float(np.quantile(ts,.5)),'p95_ms':float(np.quantile(ts,.95))})
        write_json(out/'convergence.json',convergence);pd.DataFrame(timings).to_csv(out/'sampling_timing.csv',index=False)
    if args.train_only:
        write_json(out/'training_only_completion.json',{'completed':True});return
    raw=pd.concat(all_groups,ignore_index=True);raw.to_csv(out/'per_simulation_nll.csv',index=False)
    pd.DataFrame(metrics).groupby('model').test_group_nll.agg(['mean','std','count']).to_csv(out/'summary.csv')
    base=pd.read_csv(args.baseline_dir/'per_simulation_nll.csv');candidate=raw.groupby('group').nll.mean();comparisons={}
    for model in ['conditional_spline','unconditional_spline','conditional_gaussian']:
        original=base[base.model==model].groupby('group').nll.mean()
        comparisons[model]=group_bootstrap_delta(candidate,original)
    write_json(out/'simformer_minus_flow_nll.json',comparisons)
    write_json(out/'completion.json',{'completed':True,'wall_seconds':time.perf_counter()-begin})


if __name__=='__main__':main()
