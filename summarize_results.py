"""Create an honest, seed-aware summary of completed results."""
import csv,json,sys
from pathlib import Path
import numpy as np
from research_set import save_csv

def summarize(folder):
    folder=Path(folder)
    with (folder/'metrics.csv').open() as handle:raw=list(csv.DictReader(handle))
    rows=[]
    for row in raw:
        item={}
        for key,value in row.items():
            try:item[key]=float(value)
            except ValueError:item[key]=value
        rows.append(item)
    config=json.loads((folder/'config.json').read_text())['config']
    expected=len(config['methods'])*len(config['architectures'])*len(config['targets'])*len(config['seeds'])
    assert len(rows)==expected,(len(rows),expected)
    macro=[]
    for architecture in config['architectures']:
        for method in config['methods']:
            selected=[r for r in rows if r['architecture']==architecture and r['method']==method]
            seed_f1=[np.mean([r['f1'] for r in selected if r['seed']==seed]) for seed in config['seeds']]
            item=dict(architecture=architecture,method=method,macro_f1_mean=float(np.mean(seed_f1)),
                macro_f1_seed_sd=float(np.std(seed_f1,ddof=1)) if len(seed_f1)>1 else 0.,
                mean_accuracy=float(np.mean([r['accuracy'] for r in selected])),
                mean_neurons=float(np.mean([r['selected_neurons'] for r in selected])),
                mean_compact_macs=float(np.mean([r['compact_dense_macs'] for r in selected])),
                mean_compact_weight_bytes=float(np.mean([r['compact_weight_bytes'] for r in selected])),
                mean_compact_latency_ms=float(np.mean([r['compact_latency_median_ms'] for r in selected])))
            macro.append(item)
    save_csv(folder/'macro_metrics.csv',macro)
    lines=[f"Completed {len(rows)} developmental {config['mode']} runs; {config['epochs']} epochs and {len(config['seeds'])} seeds.",
        'These are synthetic-window presence-classification results, not verified cross-house generalization.',
        'Macro F1 first averages target columns within each seed, then averages seeds.','']
    for architecture in config['architectures']:
        lines.append(architecture)
        for r in [r for r in macro if r['architecture']==architecture]:
            lines.append(f"  {r['method']}: macro F1 {r['macro_f1_mean']:.4f} +/- {r['macro_f1_seed_sd']:.4f}; mean neurons {r['mean_neurons']:.1f}")
        candidate=next(r for r in macro if r['architecture']==architecture and r['method']=='adaptive_group_set')
        fixed=next(r for r in macro if r['architecture']==architecture and r['method']=='fixed_set')
        delta=candidate['macro_f1_mean']-fixed['macro_f1_mean']
        saved=1-candidate['mean_compact_weight_bytes']/fixed['mean_compact_weight_bytes']
        mac=1-candidate['mean_compact_macs']/fixed['mean_compact_macs']
        lines.append(f"  Candidate versus fixed SET: delta macro F1 {delta:+.4f}; exported weight reduction {saved:.1%}; native dense MAC reduction {mac:.1%}.")
        lines.append('  Latency is host-dependent; inspect the recorded median/p95 timings and repeat on deployment hardware.')
        lines.append('')
    lines+=['The candidate does not uniformly dominate the baselines; retain all negative comparisons.',
        'Three-seed pilot p-values are too coarse for significance claims. See paired_uncertainty.csv and POWER_NOTE.txt.',
        'Full validation-only tuning, source-independent evaluation and optional sensitivity sweeps are not completed.',
        'Publication readiness and literature novelty are not established.']
    (folder/'RESULTS_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return macro

if __name__=='__main__':summarize(sys.argv[1] if len(sys.argv)>1 else 'pilot_results')
