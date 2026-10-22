"""Locked analysis of completed CNN training; held-out scores cannot select recipes."""
# Reuse the adaptive engine; keep a single implementation of sparse layers.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_sparsity"))

import os
os.environ.setdefault("MPLCONFIGDIR",str(__import__("pathlib").Path(__file__).resolve().parents[2]/"mpl-cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import argparse,csv,itertools,json,time,hashlib
from pathlib import Path
import numpy as np
from tensorflow import keras
from research_set import Config,load_data,predict,infer_function,metrics,sha_file
from train_accuracy import METHODS,save_csv

COMPRESSED=('gradual_80','distill_80','distill_64')
REFINED=('refine_90','refine_80')
CONSERVATIVE=('refine_96',)
ANALYSIS_PLAN={
 'primary_baseline':'fixed_set',
 'methods':list(METHODS),
 'policy':'For each target, select among gradual_80,distill_80,distill_64 by highest mean validation F1 across the three seeds; ties minimum mean compact weight bytes, then method name. Apply one selected recipe to all seeds for that target.',
 'primary_metrics':'Positive-class F1 averaged over four targets within each seed, then mean/sample SD across seeds; mean binary accuracy; compact array bytes; dense kernel MACs; absolute active first-layer edges.',
 'test_selection_forbidden':True,
 'uncertainty':'Paired differences across seeds; exploratory, conditional on data; three seeds do not establish statistical superiority.',
 'family':'All seven methods and the validation-selected policy are reported, including negative results.'}

def evaluate(folder,data_dir):
    protocol=json.loads((folder/'protocol.json').read_text());plan=ANALYSIS_PLAN
    existing=folder/'analysis_plan.json'
    if existing.exists() and json.loads(existing.read_text())!=plan:raise ValueError('Analysis plan changed.')
    existing.write_text(json.dumps(plan,indent=2))
    completed=json.loads((folder/'TRAINING_COMPLETE.json').read_text())
    expected=len(protocol['methods'])*len(protocol['targets'])*len(protocol['seeds'])
    assert completed['runs']==expected
    assert protocol['implementation_sha256']==sha_file(Path(__file__).with_name('train_accuracy.py'))
    # The validation-motivated extension is declared separately; preserve the original analysis plan.
    extension=(folder/'REFINEMENT_COMPLETE.json').exists()
    conservative=(folder/'CONSERVATIVE_COMPLETE.json').exists()
    refined_candidates=REFINED+(CONSERVATIVE if conservative else ())
    methods=list(protocol['methods'])+(list(refined_candidates) if extension else [])
    if extension:
        assert json.loads((folder/'REFINEMENT_COMPLETE.json').read_text())['runs']==len(REFINED)*len(protocol['targets'])*len(protocol['seeds'])
        extension_analysis=dict(candidates=list(COMPRESSED+refined_candidates),selection=plan['policy'].replace('gradual_80,distill_80,distill_64','all declared compressed candidates'),
                                extra_cost='Refinement is 50 source + 25 fine-tuning epochs; original KD is 50 teacher + 50 student.')
        (folder/'refinement_analysis_plan.json').write_text(json.dumps(extension_analysis,indent=2))
    # Select the policies BEFORE loading test features or outcomes.
    validation=list(csv.DictReader((folder/'validation_results.csv').open()))
    if extension:validation+=list(csv.DictReader((folder/'refinement_validation_results.csv').open()))
    if conservative:validation+=list(csv.DictReader((folder/'conservative_validation_results.csv').open()))
    policy={}
    for target in protocol['targets']:
        candidates=[]
        for method in COMPRESSED:
            rs=[r for r in validation if r['method']==method and int(r['target'])==target]
            assert len(rs)==len(protocol['seeds'])
            candidates.append((float(np.mean([float(r['val_f1']) for r in rs])),
                               -float(np.mean([float(r['compact_weight_bytes']) for r in rs])),method))
        best=sorted(candidates,key=lambda c:(-c[0],-c[1],c[2]))[0]
        policy[str(target)]=best[2]
    (folder/'validation_selected_policy.json').write_text(json.dumps(policy,indent=2))
    refined_policy={}
    if extension:
        for target in protocol['targets']:
            candidates=[]
            for method in COMPRESSED+refined_candidates:
                rs=[r for r in validation if r['method']==method and int(r['target'])==target]
                assert len(rs)==len(protocol['seeds'])
                candidates.append((float(np.mean([float(r['val_f1']) for r in rs])),
                                   -float(np.mean([float(r['compact_weight_bytes']) for r in rs])),method))
            refined_policy[str(target)]=sorted(candidates,key=lambda c:(-c[0],-c[1],c[2]))[0][2]
        (folder/'refined_selected_policy.json').write_text(json.dumps(refined_policy,indent=2))
    cfg=Config(mode='accuracy',targets=tuple(protocol['targets']))
    data=load_data(data_dir,cfg);test=data['test'];rows=[]
    for r in validation:
        key=f"CNN_{r['method']}_t{r['target']}_s{r['seed']}";cache=folder/(key+'_test.json')
        if cache.exists():row=json.loads(cache.read_text())
        else:
            keras.backend.clear_session()
            model=keras.models.load_model(folder/(key+'.keras'))
            prob=predict(infer_function(model),test['x'],128);y=test['y'][:,int(r['target']):int(r['target'])+1]
            scores=metrics(y,prob,float(r['threshold']))
            row={k:(float(v) if k not in ['architecture','method'] else v) for k,v in r.items()}
            row.update(scores);row['target']=int(row['target']);row['seed']=int(row['seed'])
            row['f1_at_05']=metrics(y,prob,.5)['f1']
            np.savez_compressed(folder/(key+'_predictions.npz'),labels=y,probabilities=prob,
                                indices=test['indices'],threshold=float(r['threshold']))
            cache.write_text(json.dumps(row,indent=2))
        rows.append(row)
    save_csv(folder/'test_results.csv',rows)
    policy_rows=[]
    for r in rows:
        if r['method']==policy[str(r['target'])]:
            item=dict(r);item['source_method']=r['method'];item['method']='validation_policy';policy_rows.append(item)
    assert len(policy_rows)==len(protocol['targets'])*len(protocol['seeds'])
    refined_rows=[]
    if extension:
        for r in rows:
            if r['method']==refined_policy[str(r['target'])]:
                item=dict(r);item['source_method']=r['method'];item['method']='refined_policy';refined_rows.append(item)
        assert len(refined_rows)==len(policy_rows)
    allrows=rows+policy_rows+refined_rows;summary=[];seed_summary=[]
    report_methods=methods+['validation_policy']+(['refined_policy'] if extension else [])
    for method in report_methods:
        rs=[r for r in allrows if r['method']==method];byseed=[]
        for seed in protocol['seeds']:
            ss=[r for r in rs if r['seed']==seed];assert len(ss)==len(protocol['targets'])
            item=dict(method=method,seed=seed,macro_f1=float(np.mean([r['f1'] for r in ss])),
                      mean_accuracy=float(np.mean([r['accuracy'] for r in ss])))
            seed_summary.append(item);byseed.append(item)
        item=dict(method=method,macro_f1_mean=float(np.mean([r['macro_f1'] for r in byseed])),
                  macro_f1_seed_sd=float(np.std([r['macro_f1'] for r in byseed],ddof=1)),
                  accuracy_mean=float(np.mean([r['mean_accuracy'] for r in byseed])),
                  n_seeds=len(protocol['seeds']),n_targets=len(protocol['targets']))
        for k in ['val_f1','selected_neurons','compact_weight_bytes','compact_dense_macs','first_layer_active_edges',
                  'first_layer_sparsity_vs_full100','first_layer_sparsity_within_retained','training_seconds','teacher_extra_epochs','student_epochs']:
            item[k+'_mean']=float(np.mean([r[k] for r in rs]))
        item['total_training_epochs_including_teacher_mean']=item['student_epochs_mean']+item['teacher_extra_epochs_mean']
        summary.append(item)
    baseline=next(r for r in summary if r['method']=='fixed_set')
    for item in summary:
        item['delta_f1_pp_vs_fixed_set']=100*(item['macro_f1_mean']-baseline['macro_f1_mean'])
        item['weight_reduction_pct_vs_fixed_set']=100*(1-item['compact_weight_bytes_mean']/baseline['compact_weight_bytes_mean'])
        item['mac_reduction_pct_vs_fixed_set']=100*(1-item['compact_dense_macs_mean']/baseline['compact_dense_macs_mean'])
        item['edge_reduction_pct_vs_fixed_set']=100*(1-item['first_layer_active_edges_mean']/baseline['first_layer_active_edges_mean'])
        item['joint_f1_and_storage_improvement']=item['delta_f1_pp_vs_fixed_set']>0 and item['weight_reduction_pct_vs_fixed_set']>0
    save_csv(folder/'summary.csv',summary);save_csv(folder/'seed_summary.csv',seed_summary)
    target_summary=[]
    for target in protocol['targets']:
        for method in report_methods:
            rs=[r for r in allrows if r['target']==target and r['method']==method]
            target_summary.append(dict(target=target,method=method,f1_mean=float(np.mean([r['f1'] for r in rs])),
                                       f1_sd=float(np.std([r['f1'] for r in rs],ddof=1)),
                                       accuracy_mean=float(np.mean([r['accuracy'] for r in rs]))))
    save_csv(folder/'target_summary.csv',target_summary)
    paired=[];rng=np.random.default_rng(20261022)
    for method in report_methods:
        if method=='fixed_set':continue
        differences=[]
        for seed in protocol['seeds']:
            a=next(r for r in seed_summary if r['method']==method and r['seed']==seed)['macro_f1']
            b=next(r for r in seed_summary if r['method']=='fixed_set' and r['seed']==seed)['macro_f1']
            differences.append(a-b)
        differences=np.array(differences)
        boot=differences[rng.integers(len(differences),size=(2000,len(differences)))].mean(1)
        null=np.array([abs(float(np.mean(np.asarray(signs)*differences))) for signs in itertools.product([-1,1],repeat=len(differences))])
        paired.append(dict(method=method,paired_differences=json.dumps(differences.tolist()),mean_difference=float(differences.mean()),
                           seed_bootstrap_ci_low=float(np.quantile(boot,.025)),seed_bootstrap_ci_high=float(np.quantile(boot,.975)),
                           signflip_p=float(np.mean(null>=abs(differences.mean())-1e-12))))
    save_csv(folder/'paired_seed_differences.csv',paired)
    (folder/'EVALUATION_COMPLETE.json').write_text(json.dumps(dict(runs=len(rows),policy=policy,refined_policy=refined_policy,source_independence_verified=False,
        label_mapping_verified=False,analysis_sha256=sha_file(__file__),test_rows=len(test['x'])),indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1] / 'data and requirements')
    args=parser.parse_args();evaluate(args.out,args.data_dir)
