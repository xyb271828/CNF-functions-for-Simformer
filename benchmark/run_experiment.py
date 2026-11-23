"""One command: correctness tests, benchmark, diagnostics and evidence-based report."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parent
    data=args.data_dir.resolve();out=root/'results'
    for name in ['compiled_vectors-0.05.csv','input_data(full).csv']:
        if not (data/name).is_file():raise FileNotFoundError(data/name)
    steps=[['-m','unittest','discover','-s',str(root/'tests'),'-v'],
           ['-u',str(root/'src/benchmark.py'),'--data-dir',str(data),'--output-dir',str(out),
            '--epochs','30','--patience','6','--seeds','11','22','33'],
           ['-u',str(root/'src/diagnostics.py'),'--data-dir',str(data),'--results',str(out)],
           [str(root/'src/write_report.py'),'--results',str(out)]]
    out.mkdir(exist_ok=True)
    with open(out/'reproduction.log','w') as log:
        for step in steps:
            log.write(' '.join([sys.executable]+step)+'\n');log.flush()
            subprocess.run([sys.executable]+step,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Completed. See results/ and EXPERIMENT_REPORT_ZH.md.')

if __name__=='__main__':main()
