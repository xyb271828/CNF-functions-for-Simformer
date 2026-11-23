"""Extend the saved spline benchmark without silently rerunning its baselines."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--extended',action='store_true',help='Also run the explicitly separate 120-epoch budget')
    args=ap.parse_args();root=Path(__file__).resolve().parent;data=args.data_dir.resolve()
    if not (root/'results/per_simulation_nll.csv').exists():
        raise FileNotFoundError('Run run_experiment.py to obtain the spline baselines first.')
    steps=[['-m','unittest','discover','-s',str(root/'tests'),'-v']]
    for budget,folder,report in [(30,'results_flow_matching','FLOW_MATCHING_REPORT_ZH.md')]+(
        [(120,'results_flow_matching_120','FLOW_MATCHING_120_REPORT_ZH.md')] if args.extended else []):
        steps.extend([
            ['-u',str(root/'src/flow_matching.py'),'--data-dir',str(data),'--baseline-results',str(root/'results'),
             '--output-dir',str(root/folder),'--epochs',str(budget),'--seeds','11','22','33'],
            ['-u',str(root/'src/report_flow_matching.py'),'--data-dir',str(data),'--root',str(root),
             '--fm-dir',folder,'--report-name',report]])
    if args.extended:steps.append([str(root/'src/compare_fm_budgets.py'),'--root',str(root)])
    with open(root/'flow_matching_reproduction.log','w') as log:
        for step in steps:
            log.write(' '.join([sys.executable]+step)+'\n');log.flush()
            subprocess.run([sys.executable]+step,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Flow Matching experiments and reports completed.')

if __name__=='__main__':main()
