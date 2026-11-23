"""Generate physical atomic states for nine experimental parameters.

Usage: python src/predict.py --results results --context-json context.json --samples 1000 --output samples.csv
The context JSON is an object containing the nine named physical parameters.
This interface is an offline research utility, not a deployed service.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from benchmark import CONTEXT, make_flow


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,default=Path('results'))
    ap.add_argument('--model',choices=['spline','flow_matching'],default='spline')
    ap.add_argument('--fm-results',type=Path,default=Path('results_flow_matching'))
    ap.add_argument('--steps',type=int,default=32)
    ap.add_argument('--solver',choices=['rk4','euler'],default='rk4')
    ap.add_argument('--context-json',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=11)
    ap.add_argument('--samples',type=int,default=1000)
    ap.add_argument('--output',type=Path,default=Path('samples.csv'))
    args=ap.parse_args()
    if args.samples < 1:raise ValueError('--samples must be positive')
    torch.set_num_threads(2)
    meta=json.loads((args.results/'preprocessing.json').read_text())
    context=json.loads(args.context_json.read_text())
    missing=set(CONTEXT)-set(context)
    if missing:raise ValueError(f'Missing experimental parameters: {sorted(missing)}')
    values=np.array([context[k] for k in CONTEXT],dtype='float64')
    if not np.isfinite(values).all():raise ValueError('All context values must be finite')
    c=(values-np.array(meta['context_mean']))/np.array(meta['context_scale'])
    if args.model=='spline':
        checkpoint=torch.load(args.results/f'conditional_spline_seed{args.seed}.pt',map_location='cpu',weights_only=True)
        model=make_flow(True)
    else:
        from flow_matching import VelocityMLP,integrate_samples
        checkpoint=torch.load(args.fm_results/f'flow_matching_seed{args.seed}.pt',map_location='cpu',weights_only=True)
        model=VelocityMLP(hidden=checkpoint['hidden'])
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    torch.manual_seed(args.seed);parts=[]
    with torch.no_grad():
        for start in range(0,args.samples,512):
            batch=min(512,args.samples-start)
            ctxt=torch.tensor(np.repeat(c[None,:],batch,axis=0),dtype=torch.float32)
            if args.model=='spline':s,_=model.sample(batch,ctxt)
            else:s=integrate_samples(model,torch.randn(batch,5),ctxt,args.steps,args.solver)
            parts.append(s.numpy())
    physical=np.concatenate(parts)*np.array(meta['target_scale'])+np.array(meta['target_mean'])
    physical[:,:2]/=1000.
    physical[:,4]=np.expm1(physical[:,4])
    pd.DataFrame(physical,columns=['X_m','Y_m','Vx','Vy','Vz']).to_csv(args.output,index=False)
    print(f'Saved {args.samples} atomic states to {args.output}; no physical rejection filter was applied.')


if __name__=='__main__':main()
