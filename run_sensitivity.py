"""Optional development-only density and dense-width controls (not yet run)."""
from dataclasses import replace
from pathlib import Path
import argparse
from research_set import Config, run_experiment

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--metadata',type=Path)
    p.add_argument('--out',type=Path,default=Path('sensitivity_results'))
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37])
    args=p.parse_args()
    cfg=replace(Config.pilot(),epochs=args.epochs,seeds=tuple(args.seeds))
    # Fixed-SET trade-off curves prevent comparison against only one arbitrary
    # initial sparsity. Native-width dense controls test whether starting with
    # a smaller dense network is as effective as pruning a larger one.
    for epsilon in (5.,10.,20.,40.):
        run_experiment(args.data_dir,args.out/f'epsilon_{epsilon:g}',
            replace(cfg,epsilon=epsilon,methods=('static_sparse','fixed_set','gradient_set')),args.metadata)
    for width in (50,75,100):
        run_experiment(args.data_dir,args.out/f'dense_width_{width}',
            replace(cfg,hidden=width,methods=('dense',)),args.metadata)
