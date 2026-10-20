"""Vector figures for descriptive pilot comparisons (not significance claims)."""
from pathlib import Path
import csv,sys
import numpy as np
import matplotlib.pyplot as plt

def figures(folder):
    folder=Path(folder)
    with (folder/'macro_metrics.csv').open() as handle:rows=list(csv.DictReader(handle))
    methods=list(dict.fromkeys(r['method'] for r in rows));architectures=list(dict.fromkeys(r['architecture'] for r in rows))
    y=np.arange(len(methods));fig,axes=plt.subplots(1,len(architectures),figsize=(11,7),sharey=True,squeeze=False)
    for i,architecture in enumerate(architectures):
        ax=axes[0,i];group=[next(r for r in rows if r['architecture']==architecture and r['method']==method) for method in methods]
        means=np.array([float(r['macro_f1_mean']) for r in group]);sd=np.array([float(r['macro_f1_seed_sd']) for r in group])
        for j,method in enumerate(methods):
            ax.errorbar(means[j],j,xerr=sd[j],fmt='o',capsize=3,color='tab:orange' if method=='adaptive_group_set' else 'tab:blue')
        ax.set_yticks(y,methods);ax.set(title=architecture,xlabel='Macro F1 (mean ± seed SD, n=3)')
        ax.set_xlim(float((means-sd).min())-.01,float((means+sd).max())+.01)
        ax.grid(axis='x',alpha=.25);ax.invert_yaxis()
    fig.suptitle('All original method families and new ablations — developmental pilot')
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(folder/('macro_f1_comparison.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,len(architectures),figsize=(11,7),sharey=True,squeeze=False)
    for i,architecture in enumerate(architectures):
        ax=axes[0,i];group=[next(r for r in rows if r['architecture']==architecture and r['method']==method) for method in methods]
        costs=np.array([float(r['mean_compact_weight_bytes'])/1024 for r in group])
        ax.barh(y,costs,color=['tab:orange' if method=='adaptive_group_set' else 'tab:blue' for method in methods],alpha=.8)
        ax.set_yticks(y,methods);ax.set(title=architecture,xlabel='Exported variable storage (KiB; mean)')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.25)
    fig.suptitle('Native compact models; framework and activation memory excluded')
    fig.tight_layout()
    for suffix in ['png','svg']:fig.savefig(folder/('compact_storage_comparison.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':figures(sys.argv[1] if len(sys.argv)>1 else 'pilot_results')
