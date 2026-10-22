"""Auditable labels, metric definitions and storage accounting (no TF dependency)."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

METRIC_PROTOCOL = {
    'binary_f1': 'Positive class is appliance present (1); 2TP/(2TP+FP+FN); zero denominator returns 0.',
    'macro_f1': 'Unweighted mean of positive-class F1 across selected targets WITHIN each seed; then mean and sample SD across seeds. Not two-class macro F1.',
    'threshold': 'Maximize validation positive-class F1 on 0.05:0.01:0.95. Ties within 1e-12: nearest 0.5, then lower threshold. Predict positive when probability >= threshold.',
    'checkpoint': 'Smallest retained neuron count, then connection budget, BCE, then epoch among Pareto checkpoints within configured absolute BCE tolerance of best validation BCE.',
    'uncertainty': 'Paired seed and test-group bootstrap, conditional on these data. Not evidence of house independence or accuracy equivalence.',
    'macs': 'One multiplication plus accumulation counts as one MAC; kernel MACs only. Bias, activation, pooling and normalization operations excluded.',
    'weight_bytes': 'Sum of array nbytes, including kernels, biases, BN gamma/beta and moving statistics; training representation also includes float masks. Excludes optimizer slots, activations, framework and archive overhead.',
    'timing': 'Host wall time of synchronous inference including NumPy output transfer; five warmups, configured repetitions. Batch latency and batch-size-one latency are measured separately. No energy measurement.',
}

def load_label_schema(data_dir, schema_path=None):
    path=Path(schema_path) if schema_path else Path(data_dir)/'label_schema.json'
    with (Path(data_dir)/'samY.csv').open(newline='') as h:columns=next(csv.reader(h))[1:]
    if not path.exists():
        return dict(status='unknown',source=None,source_verified=False,
                    labels=[dict(target=i,csv_column=c,name=f'T{i}',source_verified=False) for i,c in enumerate(columns)])
    schema=json.loads(path.read_text());labels=schema.get('labels',[])
    if len(labels)!=4 or {r.get('target') for r in labels}!=set(range(4)):
        raise ValueError('Label schema must specify targets 0,1,2,3 exactly once.')
    labels=sorted(labels,key=lambda r:r['target'])
    for i,r in enumerate(labels):
        if r.get('csv_column')!=columns[i] or not str(r.get('name','')).strip():
            raise ValueError('Label schema columns/names do not match samY.csv.')
        if not isinstance(r.get('source_verified',False),bool):raise ValueError('source_verified must be boolean.')
        if r.get('source_verified',False) and not str(r.get('verification_evidence','')).strip():
            raise ValueError('Verified labels require verification_evidence from data generation/source records.')
    return dict(status=schema.get('status','declared'),source=schema.get('source'),
                source_verified=all(r.get('source_verified',False) for r in labels),labels=labels,
                schema_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def storage_accounting(model):
    rows=[]
    for layer in model.layers:
        for v in layer.weights:
            a=v.numpy();name=getattr(v,'path',v.name)
            category=('mask' if name.rsplit('/',1)[-1]=='mask' else
                      'batch_normalization' if layer.__class__.__name__=='BatchNormalization' else
                      'bias' if name.rsplit('/',1)[-1]=='bias' else 'kernel_or_other')
            rows.append(dict(layer=layer.name,variable=name,shape=list(a.shape),dtype=str(a.dtype),
                             trainable=bool(v.trainable),category=category,bytes=int(a.nbytes)))
    return dict(total_bytes=sum(r['bytes'] for r in rows),mask_bytes=sum(r['bytes'] for r in rows if r['category']=='mask'),
                dtypes=sorted({r['dtype'] for r in rows}),variables=rows)

def macro_summary(rows,targets):
    """Never treat appliance scores as independent seed replicates."""
    output=[]
    keys=sorted({(r['architecture'],r['method']) for r in rows})
    for architecture,method in keys:
        group=[r for r in rows if (r['architecture'],r['method'])==(architecture,method)]
        scores=[]
        for seed in sorted({r['seed'] for r in group}):
            values=[r for r in group if r['seed']==seed]
            if len(values)!=len(targets) or {r['target'] for r in values}!=set(targets):
                raise ValueError('Macro F1 requires exactly one result per selected target and seed.')
            scores.append(float(np.mean([r['f1'] for r in values])))
        output.append(dict(architecture=architecture,method=method,n_targets=len(targets),n_seeds=len(scores),
                           macro_positive_f1_mean=float(np.mean(scores)),
                           macro_positive_f1_seed_sd=float(np.std(scores,ddof=1)) if len(scores)>1 else 0.,
                           seed_scores=json.dumps(scores)))
    return output


def paired_macro_statistics(rows, targets, repeats=2000):
    """Paired seed bootstrap after averaging targets, with one Holm family."""
    import itertools
    rng=np.random.default_rng(20261022);output=[]
    for architecture in sorted({r['architecture'] for r in rows}):
        methods=sorted({r['method'] for r in rows if r['architecture']==architecture})
        if 'adaptive_group_set' not in methods:continue
        for baseline in methods:
            if baseline=='adaptive_group_set':continue
            subset=[r for r in rows if r['architecture']==architecture and r['method'] in (baseline,'adaptive_group_set')]
            seeds=sorted({r['seed'] for r in subset});deltas=[]
            for seed in seeds:
                scores={}
                for method in (baseline,'adaptive_group_set'):
                    selected=[r for r in subset if r['method']==method and r['seed']==seed]
                    if len(selected)!=len(targets) or {r['target'] for r in selected}!=set(targets):
                        raise ValueError('Paired macro inference requires complete aligned seed/target pairs.')
                    scores[method]=np.mean([r['f1'] for r in selected])
                deltas.append(scores['adaptive_group_set']-scores[baseline])
            deltas=np.asarray(deltas);observed=abs(float(deltas.mean()))
            sampled=np.mean(deltas[rng.integers(len(deltas),size=(repeats,len(deltas)))],axis=1)
            if len(deltas)<=16:
                null=np.array([abs(float(np.mean(np.asarray(signs)*deltas)))
                               for signs in itertools.product([-1,1],repeat=len(deltas))])
                p=float(np.mean(null>=observed-1e-12))
            else:
                null=np.abs((rng.choice([-1,1],size=(10000,len(deltas)))*deltas).mean(1))
                p=float((1+np.count_nonzero(null>=observed-1e-12))/(len(null)+1))
            output.append(dict(architecture=architecture,candidate='adaptive_group_set',baseline=baseline,
                n_seeds=len(seeds),n_targets=len(targets),mean_macro_f1_difference=float(deltas.mean()),
                paired_seed_differences=json.dumps(deltas.tolist()),
                seed_bootstrap_ci_low=float(np.quantile(sampled,.025)),
                seed_bootstrap_ci_high=float(np.quantile(sampled,.975)),seed_signflip_p=p,holm_adjusted_p=1.,
                interpretation='Conditional paired-seed inference; targets averaged before bootstrap. No test-population uncertainty or accuracy-equivalence claim.'))
    previous=0.
    for rank,index in enumerate(np.argsort([r['seed_signflip_p'] for r in output])):
        previous=max(previous,min(1.,output[index]['seed_signflip_p']*(len(output)-rank)))
        output[index]['holm_adjusted_p']=previous
    return output
