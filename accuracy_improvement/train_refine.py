"""Validation-motivated extension, declared before any new held-out evaluation.
Warm-start a validation-selected full-width teacher and compress for 25 epochs.
"""
# Reuse the adaptive engine; keep a single implementation of sparse layers.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_sparsity"))

import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'mpl-cache'))
import argparse,json,time,csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from train_accuracy import save_csv
from research_set import (Config,build_model,load_data,infer_function,predict,select_threshold,metrics,
                          reset_adam_slots,compact_model,mac_counts,sha_file)
from protocol import storage_accounting

EXTENSION_PLAN={
 'methods':['refine_90','refine_80'], 'fine_tune_epochs':25,'pretraining_epochs':50,
 'teacher_choice':'Per target, highest mean validation F1 across seeds among dense_teacher and fixed_set; tie fixed_set. No test scores used.',
 'initialization':'Restore selected source weights and existing zero mask; no random edge regrowth.',
 'pruning':'Cubic monotone pruning during epochs 1–15, freeze topology in epochs 16–25. Final widths 90/80, maximum retained density .20. Cap desired edges at available retained mask capacity.',
 'optimizer':'Adam cosine 0.0003 to 0.000006 for 25 fine-tuning epochs',
 'distillation':'alpha .25, temperature 2, frozen selected source probabilities on training examples only',
 'selection':'Maximum validation F1 among final-budget checkpoints, then BCE and earliest epoch',
 'cost':'50 source epochs plus 25 fine-tuning epochs; not an equal-total-training-budget comparison against the 50-epoch baseline.'}

def restore(source):
    model=build_model('CNN',Config(mode='accuracy'))
    for name in ['conv1','conv2','bn']:model.get_layer(name).set_weights(source.get_layer(name).get_weights())
    w,b=source.get_layer('compact_hidden').get_weights();layer=model.get_layer('sparse_hidden')
    layer.kernel.assign(w);layer.bias.assign(b);layer.mask.assign((w!=0).astype('float32'))
    model.get_layer('hidden2').set_weights(source.get_layer('compact_hidden2').get_weights())
    model.get_layer('output').set_weights(source.get_layer('compact_output').get_weights())
    return model

def compress(layer,next_layer,allowed,units,density,opt):
    w=layer.kernel.numpy()*layer.mask.numpy();b=layer.bias.numpy();active=np.flatnonzero(allowed)
    score=np.sqrt(np.sum(w*w,axis=0)+b*b)*np.linalg.norm(next_layer.kernel.numpy(),axis=1)
    choose=active[np.argsort(-score[active],kind='stable')[:units]]
    allowed=np.zeros(100,bool);allowed[choose]=True
    old=layer.mask.numpy().astype(bool);candidates=np.flatnonzero((old&allowed[None,:]).ravel())
    desired=min(len(candidates),int(round(w.shape[0]*units*density)))
    keep=candidates[np.argsort(-np.abs(w.ravel()[candidates]),kind='stable')[:desired]]
    mask=np.zeros(w.size,bool);mask[keep]=True;mask=mask.reshape(w.shape)
    layer.kernel.assign(w*mask);layer.mask.assign(mask.astype('float32'));layer.bias.assign(b*allowed)
    if opt.built:
        reset_adam_slots(opt,layer.kernel,old&~mask);reset_adam_slots(opt,layer.bias,~allowed)
    assert mask.sum()<=int(round(w.shape[0]*units*density))
    return allowed

def refine(method,target,seed,source_path,data,out):
    start=time.perf_counter();keras.backend.clear_session();keras.utils.set_random_seed(seed)
    teacher=keras.models.load_model(source_path);model=restore(teacher)
    tx=data['train']['x'];ty=data['train']['y'][:,target:target+1]
    vx=data['validation']['x'];vy=data['validation']['y'][:,target:target+1]
    teacher_p=predict(infer_function(teacher),tx,128)
    teacher_q=(1/(1+np.exp(-np.log(np.clip(teacher_p,1e-6,1-1e-6)/(1-np.clip(teacher_p,1e-6,1-1e-6)))/2))).astype('float32')
    source_v=predict(infer_function(teacher),vx,128);restored_v=predict(infer_function(model),vx,128)
    np.testing.assert_allclose(source_v,restored_v,rtol=1e-5,atol=2e-6)
    layer=model.get_layer('sparse_hidden');next_layer=model.get_layer('hidden2')
    density0=float(layer.mask.numpy().mean());initial_budget=int(layer.mask.numpy().sum());final_units=int(method.rsplit('_',1)[-1])
    allowed=np.ones(100,bool);allowed_var=tf.Variable(allowed.astype('float32'),trainable=False)
    lr=keras.optimizers.schedules.CosineDecay(.0003,25*int(np.ceil(len(tx)/128)),alpha=.02)
    opt=keras.optimizers.Adam(lr);features=keras.Model(model.input,model.get_layer('output').input);head=model.get_layer('output')
    @tf.function(input_signature=[tf.TensorSpec((None,600,1),tf.float32),tf.TensorSpec((None,1),tf.float32),tf.TensorSpec((None,1),tf.float32)])
    def step(x,y,q):
        with tf.GradientTape() as tape:
            h=features(x,training=True);logits=tf.matmul(h,head.kernel)+head.bias
            hard=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=logits))
            ce=tf.nn.sigmoid_cross_entropy_with_logits(labels=q,logits=logits/2.)
            entropy=-(q*tf.math.log(tf.clip_by_value(q,1e-7,1.))+(1-q)*tf.math.log(tf.clip_by_value(1-q,1e-7,1.)))
            loss=.75*hard+.25*4*tf.reduce_mean(ce-entropy)
        grads=tape.gradient(loss,model.trainable_variables);opt.apply_gradients(zip(grads,model.trainable_variables))
        layer.kernel.assign(layer.kernel*layer.mask);layer.bias.assign(layer.bias*allowed_var)
        return hard,loss
    infer=infer_function(model);rng=np.random.default_rng(seed+2000);history=[];best=None;chosen=None
    for e in range(1,26):
        t0=time.perf_counter();progress=min(1.,e/15);fraction=(1-progress)**3
        units=int(round(final_units+(100-final_units)*fraction));density=.20+(density0-.20)*fraction
        allowed=compress(layer,next_layer,allowed,units,density,opt);allowed_var.assign(allowed.astype('float32'))
        order=rng.permutation(len(tx));hard=loss=0.
        for a in range(0,len(tx),128):
            ix=order[a:a+128];h,l=step(tx[ix],ty[ix],teacher_q[ix]);hard+=float(h)*len(ix);loss+=float(l)*len(ix)
        vp=predict(infer,vx,128);threshold=select_threshold(vy,vp);m=metrics(vy,vp,threshold)
        active=int(np.count_nonzero(layer.kernel.numpy()*layer.mask.numpy()))
        history.append(dict(epoch=e,train_bce=hard/len(tx),train_objective=loss/len(tx),val_bce=m['bce'],val_f1=m['f1'],
            val_accuracy=m['accuracy'],threshold=threshold,hidden_units=int(allowed.sum()),first_layer_active_edges=active,
            first_layer_sparsity_vs_full100=1-active/118400,learning_rate=float(lr(opt.iterations)),seconds=time.perf_counter()-t0,eligible=e>=15))
        if e>=15:
            key=(m['f1'],-m['bce'],-e)
            if best is None or key>best:best=key;chosen=dict(epoch=e,threshold=threshold,val_f1=m['f1'],val_bce=m['bce'],weights=model.get_weights())
        if e%5==0:print(f'{method} T{target} seed{seed} fine-tune {e}/25 valF1={m["f1"]:.4f} units={units}',flush=True)
    model.set_weights(chosen.pop('weights'));native,units=compact_model(model)
    vp=predict(infer,vx,128);npred=predict(infer_function(native),vx,128)
    np.testing.assert_allclose(vp,npred,rtol=1e-5,atol=2e-6);assert np.array_equal(vp>=chosen['threshold'],npred>=chosen['threshold'])
    stem=f'CNN_{method}_t{target}_s{seed}';native.save(out/(stem+'.keras'))
    reloaded=keras.models.load_model(out/(stem+'.keras'));rp=predict(infer_function(reloaded),vx,128)
    np.testing.assert_allclose(npred,rp,rtol=1e-5,atol=2e-6);assert np.array_equal(npred>=chosen['threshold'],rp>=chosen['threshold'])
    w=layer.kernel.numpy()*layer.mask.numpy();kernels=[v.numpy() for l in native.layers for v in l.weights if len(v.shape)>1]
    _,macs=mac_counts(native)
    row=dict(architecture='CNN',method=method,target=target,seed=seed,epochs=25,**chosen,selected_neurons=units,
        first_layer_active_edges=int(np.count_nonzero(w)),first_layer_sparsity_vs_full100=1-float(np.count_nonzero(w))/118400,
        first_layer_sparsity_within_retained=1-float(np.count_nonzero(w))/(1184*units),
        all_kernel_sparsity=1-sum(np.count_nonzero(k) for k in kernels)/sum(k.size for k in kernels),
        compact_weight_bytes=storage_accounting(native)['total_bytes'],compact_dense_macs=macs,training_seconds=time.perf_counter()-start,
        student_epochs=25,teacher_extra_epochs=50,val_compaction_max_abs_error=float(np.max(np.abs(vp-npred))),
        val_reload_max_abs_error=float(np.max(np.abs(rp-npred))),archive_bytes=(out/(stem+'.keras')).stat().st_size)
    save_csv(out/(stem+'_history.csv'),history);(out/(stem+'_selection.json')).write_text(json.dumps(row,indent=2))
    print(f'FINISHED {stem} valF1={row["val_f1"]:.4f} seconds={row["training_seconds"]:.1f}',flush=True)
    return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1] / 'data and requirements')
    args=p.parse_args();out=args.out
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    if not (out/'TRAINING_COMPLETE.json').exists():raise ValueError('Complete primary training first.')
    if any(out.glob('*_test.json')):raise ValueError('Refinement must precede held-out evaluation.')
    config=json.loads((out/'protocol.json').read_text());base=list(csv.DictReader((out/'validation_results.csv').open()))
    plan=dict(EXTENSION_PLAN,implementation_sha256=sha_file(__file__),base_train_sha256=sha_file(Path(__file__).with_name('train_accuracy.py')))
    (out/'refinement_plan.json').write_text(json.dumps(plan,indent=2));sources={}
    for t in config['targets']:
        choices=[]
        for method in ['fixed_set','dense_teacher']:
            rs=[r for r in base if r['method']==method and int(r['target'])==t]
            choices.append((np.mean([float(r['val_f1']) for r in rs]),method))
        sources[str(t)]=sorted(choices,key=lambda x:(-x[0],0 if x[1]=='fixed_set' else 1))[0][1]
    (out/'refinement_sources.json').write_text(json.dumps(sources,indent=2))
    full=load_data(args.data_dir,Config(mode='accuracy',targets=tuple(config['targets'])))
    data={k:full[k] for k in ['train','validation']};extra=[]
    for t in config['targets']:
        for seed in config['seeds']:
            for method in EXTENSION_PLAN['methods']:
                stem=f'CNN_{method}_t{t}_s{seed}';cache=out/(stem+'_selection.json')
                if cache.exists():row=json.loads(cache.read_text())
                else:row=refine(method,t,seed,out/f'CNN_{sources[str(t)]}_t{t}_s{seed}.keras',data,out)
                extra.append(row);save_csv(out/'refinement_validation_results.csv',extra)
    (out/'REFINEMENT_COMPLETE.json').write_text(json.dumps(dict(runs=len(extra),test_evaluation_performed=False)))
if __name__=='__main__':main()
