"""Predeclared CNN accuracy/compression study. Training never reads test arrays."""
# Reuse the adaptive engine; keep a single implementation of sparse layers.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_sparsity"))

import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path(__file__).resolve().parents[2]/'mpl-cache'))
import argparse,csv,hashlib,json,math,platform,sys,time
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import tensorflow as tf
from tensorflow import keras
from research_set import (Config,build_model,initial_mask,rewire,reset_adam_slots,
                          compact_model,infer_function,predict,select_threshold,metrics,mac_counts,load_data,sha_file)
from protocol import storage_accounting

METHODS=('dense_teacher','fixed_set','dense_80','dense_64','gradual_80','distill_80','distill_64')

def save_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def schedule(epoch,epochs,final_units,final_density=.25):
    """Cubic pruning from end-of-epoch 15 to 35 (scaled for smoke tests)."""
    warm=max(1,round(.3*epochs));end=max(warm+1,round(.7*epochs))
    progress=np.clip((epoch-warm)/(end-warm),0.,1.)
    keep=(1-progress)**3
    return int(round(final_units+(100-final_units)*keep)),float(final_density+(1-final_density)*keep),warm,end

def prune_exact(layer,next_layer,units,density,optimizer,allowed):
    """Monotone neuron and edge removal, with stable magnitude tie breaks."""
    w=layer.kernel.numpy()*layer.mask.numpy();b=layer.bias.numpy()
    active=np.flatnonzero(allowed)
    if units>len(active):raise ValueError('Monotone pruning cannot revive columns.')
    if units<len(active):
        score=np.sqrt(np.sum(w*w,axis=0)+b*b)*np.linalg.norm(next_layer.kernel.numpy(),axis=1)
        chosen=active[np.argsort(-score[active],kind='stable')[:units]]
        allowed=np.zeros(w.shape[1],bool);allowed[chosen]=True
    eligible=layer.mask.numpy().astype(bool)&allowed[None,:]
    desired=int(round(w.shape[0]*units*density))
    candidates=np.flatnonzero(eligible.ravel())
    if desired>len(candidates):raise ValueError('Requested mask density exceeds retained connections.')
    kept=candidates[np.argsort(-np.abs(w.ravel()[candidates]),kind='stable')[:desired]]
    newmask=np.zeros(w.size,bool);newmask[kept]=True;newmask=newmask.reshape(w.shape)
    changed=layer.mask.numpy().astype(bool)&~newmask
    layer.mask.assign(newmask.astype('float32'));layer.kernel.assign(w*newmask);layer.bias.assign(b*allowed)
    if optimizer.built:
        reset_adam_slots(optimizer,layer.kernel,changed)
        reset_adam_slots(optimizer,layer.bias,~allowed)
    assert int(newmask.sum())==desired
    assert int(allowed.sum())==units
    return allowed

def validate_checkpoint(infer,vx,vy,batch):
    prob=predict(infer,vx,batch)
    threshold=select_threshold(vy,prob);m=metrics(vy,prob,threshold)
    return prob,threshold,m

def train_one(method,target,seed,data,out,epochs=50,batch_size=128,teacher=None):
    start=time.perf_counter();keras.backend.clear_session();keras.utils.set_random_seed(seed)
    hidden=int(method.split('_')[-1]) if method.startswith('dense_') and method!='dense_teacher' else 100
    cfg=Config(mode='accuracy',hidden=hidden,epochs=epochs)
    model=build_model('CNN',cfg);layer=model.get_layer('sparse_hidden');next_layer=model.get_layer('hidden2')
    tx=data['train']['x'];ty=data['train']['y'][:,target:target+1]
    vx=data['validation']['x'];vy=data['validation']['y'][:,target:target+1]
    kd=method.startswith('distill_');gradual=kd or method.startswith('gradual_')
    final_units=int(method.rsplit('_',1)[-1]) if gradual else hidden
    rng=np.random.default_rng(seed+1000)
    if method=='fixed_set':
        mask=initial_mask(tuple(layer.kernel.shape),cfg,rng);layer.mask.assign(mask);layer.kernel.assign(layer.kernel*mask)
    initial_budget=int(layer.mask.numpy().sum())
    steps=int(np.ceil(len(tx)/batch_size))
    lr_schedule=keras.optimizers.schedules.CosineDecay(.001,epochs*steps,alpha=.02)
    opt=keras.optimizers.Adam(lr_schedule)
    allowed=np.ones(hidden,bool);allowed_var=tf.Variable(allowed.astype('float32'),trainable=False)
    if kd:
        if teacher is None:raise ValueError('Distillation requires train-only teacher probabilities.')
        tp=np.clip(teacher,1e-6,1-1e-6)
        soft=(1/(1+np.exp(-np.log(tp/(1-tp))/2))).astype('float32')
    else:soft=np.zeros_like(ty)
    features=keras.Model(model.input,model.get_layer('output').input);head=model.get_layer('output')
    @tf.function(input_signature=[tf.TensorSpec((None,600,1),tf.float32),tf.TensorSpec((None,1),tf.float32),tf.TensorSpec((None,1),tf.float32)])
    def step(x,y,q):
        with tf.GradientTape() as tape:
            h=features(x,training=True);logits=tf.matmul(h,head.kernel)+head.bias
            supervised=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
            if kd:
                kd_ce=tf.nn.sigmoid_cross_entropy_with_logits(labels=q,logits=logits/2.)
                entropy=-(q*tf.math.log(tf.clip_by_value(q,1e-7,1.))+(1-q)*tf.math.log(tf.clip_by_value(1-q,1e-7,1.)))
                objective=.5*supervised+.5*4*tf.reduce_mean(kd_ce-entropy)
            else:objective=supervised
        grads=tape.gradient(objective,model.trainable_variables)
        opt.apply_gradients(zip(grads,model.trainable_variables))
        layer.kernel.assign(layer.kernel*layer.mask);layer.bias.assign(layer.bias*allowed_var)
        return supervised,objective
    inference=infer_function(model)
    shuffler=np.random.default_rng(seed+2000);history=[];best_key=None;chosen=None
    for e in range(1,epochs+1):
        t0=time.perf_counter()
        if gradual:
            units,density,warm,finish=schedule(e,epochs,final_units)
            allowed=prune_exact(layer,next_layer,units,density,opt,allowed)
            allowed_var.assign(allowed.astype('float32'))
        else:finish=1
        order=shuffler.permutation(len(tx));hard=objective=0.
        for a in range(0,len(tx),batch_size):
            ix=order[a:a+batch_size];h,o=step(tx[ix],ty[ix],soft[ix]);hard+=float(h)*len(ix);objective+=float(o)*len(ix)
        vp,threshold,val=validate_checkpoint(inference,vx,vy,batch_size)
        current_units=int(np.count_nonzero(np.any(layer.kernel.numpy()*layer.mask.numpy()!=0,axis=0)|(layer.bias.numpy()!=0)))
        edge_count=int(np.count_nonzero(layer.kernel.numpy()*layer.mask.numpy()))
        history.append(dict(epoch=e,train_bce=hard/len(tx),train_objective=objective/len(tx),val_bce=val['bce'],
            val_f1=val['f1'],val_accuracy=val['accuracy'],threshold=threshold,hidden_units=current_units,
            first_layer_active_edges=edge_count,first_layer_sparsity_vs_full100=1-edge_count/(1184*100),
            learning_rate=float(lr_schedule(opt.iterations)),seconds=time.perf_counter()-t0,
            eligible=(e>=finish if gradual else True)))
        # All methods optimize validation F1. Compressed methods must reach their final budget first.
        if history[-1]['eligible']:
            key=(val['f1'],-val['bce'],-e)
            if best_key is None or key>best_key:
                best_key=key;chosen=dict(epoch=e,threshold=threshold,val_f1=val['f1'],val_bce=val['bce'],weights=model.get_weights())
        if method=='fixed_set' and e<epochs:
            rewire(layer,initial_budget,rng,cfg,opt)
        if e%10==0 or e==epochs:
            print(f'{method} T{target} seed{seed} epoch {e}/{epochs} valF1={val["f1"]:.4f} units={current_units}',flush=True)
    assert chosen is not None
    model.set_weights(chosen.pop('weights'))
    native,units=compact_model(model);npred=predict(infer_function(native),vx,batch_size)
    reference=predict(inference,vx,batch_size)
    np.testing.assert_allclose(reference,npred,rtol=1e-5,atol=2e-6)
    assert np.array_equal(reference>=chosen['threshold'],npred>=chosen['threshold'])
    stem=f'CNN_{method}_t{target}_s{seed}'
    native.save(out/(stem+'.keras'))
    reloaded=keras.models.load_model(out/(stem+'.keras'))
    reload_pred=predict(infer_function(reloaded),vx,batch_size)
    np.testing.assert_allclose(npred,reload_pred,rtol=1e-5,atol=2e-6)
    assert np.array_equal(npred>=chosen['threshold'],reload_pred>=chosen['threshold'])
    save_csv(out/(stem+'_history.csv'),history)
    w=layer.kernel.numpy()*layer.mask.numpy();weight_arrays=[v.numpy() for v in native.weights]
    kernels=[v.numpy() for l in native.layers for v in l.weights if len(v.shape)>1]
    _,macs=mac_counts(native)
    row=dict(architecture='CNN',method=method,target=target,seed=seed,epochs=epochs,**chosen,
        selected_neurons=units,first_layer_active_edges=int(np.count_nonzero(w)),
        first_layer_sparsity_vs_full100=1-float(np.count_nonzero(w))/(1184*100),
        first_layer_sparsity_within_retained=1-float(np.count_nonzero(w))/(1184*units),
        all_kernel_sparsity=1-sum(np.count_nonzero(k) for k in kernels)/sum(k.size for k in kernels),
        compact_weight_bytes=storage_accounting(native)['total_bytes'],compact_dense_macs=macs,
        training_seconds=time.perf_counter()-start,student_epochs=epochs,teacher_extra_epochs=epochs if kd else 0,
        val_compaction_max_abs_error=float(np.max(np.abs(reference-npred))),
        val_reload_max_abs_error=float(np.max(np.abs(npred-reload_pred))),
        archive_bytes=(out/(stem+'.keras')).stat().st_size)
    (out/(stem+'_selection.json')).write_text(json.dumps(row,indent=2))
    if method=='dense_teacher':
        train_prob=predict(infer_function(native),tx,batch_size)
        np.save(out/(stem+'_train_prob.npy'),train_prob)
    print(f'FINISHED {stem} selected epoch {row["epoch"]} valF1={row["val_f1"]:.4f} seconds={row["training_seconds"]:.1f}',flush=True)
    return row

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=Path('.'))
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--epochs',type=int,default=50)
    parser.add_argument('--seeds',type=int,nargs='+',default=[11,23,37]);parser.add_argument('--targets',type=int,nargs='+',default=[0,1,2,3])
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    cfg=Config(mode='accuracy',targets=tuple(args.targets));full=load_data(args.data_dir,cfg)
    data={k:full[k] for k in ['train','validation']}  # Deliberately excludes test arrays.
    protocol=dict(architecture='CNN',methods=METHODS,epochs=args.epochs,seeds=args.seeds,targets=args.targets,
        data_audit=full['audit'],implementation_sha256=sha_file(__file__),base_module_sha256=sha_file(Path(__file__).resolve().parents[1] / 'adaptive_sparsity/research_set.py'),
        tensorflow=tf.__version__,keras=keras.__version__,numpy=np.__version__,platform=platform.platform(),
        threads=dict(intra_op=4,inter_op=1),deterministic_ops=True,
        selection='Maximum validation positive-class F1; ties minimum validation BCE, then earliest epoch. Compressed candidates eligible only after final budget.',
        distillation=dict(alpha=.5,temperature=2.,teacher_selection='validation F1',inputs='training examples only'),
        sparsity=dict(widths=[80,64],retained_density=.25,warmup_fraction=.3,pruning_end_fraction=.7,law='cubic magnitude pruning'),
        lr=dict(initial=.001,law='cosine',minimum=.00002),
        status='predeclared exploratory comparison; no cross-house independence, no claimed outcome')
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):raise ValueError('Protocol mismatch: choose new output folder.')
    path.write_text(json.dumps(protocol,indent=2));rows=[]
    for target in args.targets:
        for seed in args.seeds:
            # Fixed predeclared ordering; timing is diagnostic, not a speed benchmark.
            for method in METHODS:
                stem=f'CNN_{method}_t{target}_s{seed}';cache=out/(stem+'_selection.json')
                if cache.exists() and (out/(stem+'.keras')).exists():row=json.loads(cache.read_text())
                else:
                    teacher=np.load(out/f'CNN_dense_teacher_t{target}_s{seed}_train_prob.npy') if method.startswith('distill_') else None
                    row=train_one(method,target,seed,data,out,args.epochs,teacher=teacher)
                rows.append(row);save_csv(out/'validation_results.csv',rows)
    (out/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(runs=len(rows),test_evaluation_performed=False)))

if __name__=='__main__':main()
