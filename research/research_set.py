"""Controlled NILM appliance-classification experiments; no publication claim."""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
@keras.utils.register_keras_serializable(package='AdaptiveSET')
class MaskedDense(layers.Layer):
    def __init__(self, units=100, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    def build(self, shape):
        self.kernel = self.add_weight(name='kernel', shape=(int(shape[-1]), self.units), initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias', shape=(self.units,), initializer='zeros')
        self.mask = self.add_weight(name='mask', shape=self.kernel.shape, initializer='ones', trainable=False)
    def call(self, inputs):
        return tf.nn.relu(tf.matmul(inputs, self.kernel * self.mask) + self.bias)
    def get_config(self):
        return dict(super().get_config(), units=self.units)


def initial_mask(shape, cfg, rng):
    probability = min(1., cfg.epsilon * sum(shape)/np.prod(shape))
    # Exact budget facilitates matching fixed/adaptive initial states.
    count = int(round(probability*np.prod(shape)))
    mask = np.zeros(int(np.prod(shape)), dtype=np.float32)
    mask[rng.choice(len(mask),count,replace=False)] = 1
    return mask.reshape(shape)


def save_csv(path,rows):
    with Path(path).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


METHODS=('dense','lasso','group_lasso','static_sparse','fixed_set','set_lasso',
         'gradient_set','adaptive_set','adaptive_gradient_set','adaptive_structured_set','fixed_group_set','adaptive_group_set')
DYNAMIC={'fixed_set','set_lasso','gradient_set','adaptive_set','adaptive_gradient_set','adaptive_structured_set','fixed_group_set','adaptive_group_set'}
ADAPTIVE={'adaptive_set','adaptive_gradient_set','adaptive_structured_set','adaptive_group_set'}
GRADIENT={'gradient_set','adaptive_gradient_set','adaptive_structured_set','fixed_group_set','adaptive_group_set'}

@dataclass
class Config:
    mode: str='smoke'
    epochs: int=3
    seeds: tuple=(42,)
    architectures: tuple=('DNN','CNN')
    methods: tuple=METHODS
    targets: tuple=(0,1,2,3)
    batch_size: int=128
    learning_rate: float=.001
    epsilon: float=20.
    rewire_fraction: float=.3
    min_density: float=.02
    max_density: float=.6
    shrink_factor: float=.9
    grow_factor: float=1.1
    tolerance: float=.01
    warmup_epochs: int=1
    controller_patience: int=1
    cooldown: int=1
    l1_strength: float=1e-4
    l21_strength: float=1e-3
    hidden: int=100
    second_hidden: int=20
    conv_filters: int=16
    bootstrap_repeats: int=200
    latency_repeats: int=20
    checkpoint_tolerance: float=.005
    search_trials: int=1
    search_epochs: int=10
    @classmethod
    def pilot(cls):
        return cls(mode='pilot',epochs=10,seeds=(11,23,37),warmup_epochs=3,
                   controller_patience=2,bootstrap_repeats=1000,latency_repeats=30)
    @classmethod
    def full(cls):
        return cls(mode='full',epochs=50,seeds=(11,23,37,41,53,67,79,83,97,101),
                   warmup_epochs=5,controller_patience=3,bootstrap_repeats=2000,latency_repeats=50,search_trials=3)

class StableBudgetController:
    """Per-target validation feedback with patience, cooldown and bounded changes."""
    def __init__(self,cfg):
        self.cfg=cfg; self.best=float('inf'); self.good=0; self.bad=0; self.last=-100
    def decide(self,epoch,val_bce,current,total):
        acceptable=val_bce <= self.best*(1+self.cfg.tolerance)
        self.best=min(self.best,val_bce)
        self.good=self.good+1 if acceptable else 0
        self.bad=self.bad+1 if not acceptable else 0
        if epoch+1 < self.cfg.warmup_epochs:return current,'warmup'
        if epoch-self.last < self.cfg.cooldown:return current,'cooldown'
        factor=1.;action='hold'
        if self.bad>=self.cfg.controller_patience:factor=self.cfg.grow_factor;action='grow'
        elif self.good>=self.cfg.controller_patience:factor=self.cfg.shrink_factor;action='shrink'
        if action!='hold':self.good=self.bad=0;self.last=epoch
        lo=int(np.ceil(total*self.cfg.min_density));hi=int(np.floor(total*self.cfg.max_density))
        return int(np.clip(round(current*factor),lo,hi)),action

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_data(data_dir,cfg,metadata_path=None):
    data_dir=Path(data_dir)
    x=np.loadtxt(data_dir/'samX.csv',delimiter=',',skiprows=1,usecols=range(1,601),dtype=np.float32)
    y=np.loadtxt(data_dir/'samY.csv',delimiter=',',skiprows=1,usecols=range(1,5),dtype=np.float32)
    assert x.ndim==2 and x.shape[1]==600 and y.shape==(len(x),4)
    assert np.isfinite(x).all() and np.isin(y,[0,1]).all()
    fingerprints=np.array([hashlib.sha256(row.tobytes()).hexdigest() for row in x])
    # Audit contradictory exact duplicates before evaluation.
    label_by_hash={}
    for key,label in zip(fingerprints,y):
        if key in label_by_hash and not np.array_equal(label_by_hash[key],label):
            raise ValueError('Identical aggregate input has conflicting labels; investigate data generation.')
        label_by_hash[key]=label
    groups=fingerprints.copy()
    if metadata_path:
        with Path(metadata_path).open() as handle:
            metadata=list(csv.DictReader(handle))
        required={'row_index','split','group_id'}
        if not metadata or not required.issubset(metadata[0]):raise ValueError('Metadata requires row_index,split,group_id.')
        mapping={int(row['row_index']):row for row in metadata}
        if len(mapping)!=len(x) or set(mapping)!=set(range(len(x))):raise ValueError('Metadata must map each CSV row exactly once, zero-based.')
        split={name:np.array([i for i in range(len(x)) if mapping[i]['split']==name],dtype=int)
               for name in ['train','validation','test']}
        if sum(map(len,split.values()))!=len(x):raise ValueError('split must be train,validation,test.')
        groups=np.array([mapping[i]['group_id'] for i in range(len(x))])
        if any(not value.strip() for value in groups):raise ValueError('group_id must be populated.')
        for a,b in itertools.combinations(split,2):
            if set(groups[split[a]]) & set(groups[split[b]]):raise ValueError('Provenance groups overlap between splits.')
        provenance='Provided groups are disjoint; their scientific meaning still needs source verification.'
    else:
        split={'test':np.arange(2000),'validation':np.arange(2000,3600),'train':np.arange(3600,len(x))}
        provenance='CSV-only: source-window overlap and unseen-house generalization are NOT verified.'
    # Deduplicate globally, prioritizing held-out evaluation partitions. This changes
    # the distribution to unique aggregate windows; disclose the resulting estimand.
    seen=set();clean={};removed={}
    for name in ['test','validation','train']:
        kept=[]
        for i in split[name]:
            key=fingerprints[i]
            if key not in seen:kept.append(i);seen.add(key)
        removed[name]=int(len(split[name])-len(kept));clean[name]=np.array(kept,dtype=int)
    if cfg.mode=='smoke':limits={'train':128,'validation':64,'test':64}
    elif cfg.mode=='pilot':limits={'train':None,'validation':None,'test':None}
    else:limits={k:None for k in clean}
    for key,limit in limits.items():
        if limit:clean[key]=clean[key][:limit]
    mean=float(x[clean['train']].mean());scale=max(float(x[clean['train']].std()),1e-6)
    data={}
    for name,indices in clean.items():
        assert len(indices)>0
        assert all(0<y[indices,t].mean()<1 for t in cfg.targets), f'Missing class in {name}'
        data[name]=dict(x=((x[indices]-mean)/scale)[...,None],y=y[indices],indices=indices,
                        groups=groups[indices],input_hashes=fingerprints[indices])
    data['audit']=dict(input_rows=len(x),unique_input_windows=len(set(fingerprints)),
        removed_duplicates=removed,counts={k:len(v) for k,v in clean.items()},
        positive_rates={k:y[v].mean(0).tolist() for k,v in clean.items()},mean=mean,scale=scale,
        samX_sha256=sha_file(data_dir/'samX.csv'),samY_sha256=sha_file(data_dir/'samY.csv'),
        metadata_sha256=sha_file(metadata_path) if metadata_path else None,
        provenance=provenance,source_independence_verified=False,
        estimand='Unique synthetic aggregate windows; results do not estimate the original duplicate-weighted mixture distribution.')
    return data

def build_model(architecture,cfg):
    inp=layers.Input((600,1));x=inp
    if architecture=='CNN':
        # Compact controlled backbone; original architecture retained in legacy/.
        x=layers.Conv1D(cfg.conv_filters,5,strides=2,activation='relu',name='conv1')(x)
        x=layers.Conv1D(cfg.conv_filters,3,strides=2,activation='relu',name='conv2')(x)
        x=layers.BatchNormalization(name='bn')(x)
        x=layers.MaxPooling1D(2,name='pool')(x)
    x=layers.Flatten(name='flatten')(x)
    x=layers.Dropout(.2,name='dropout')(x)
    x=MaskedDense(cfg.hidden,name='sparse_hidden')(x)
    x=layers.Dense(cfg.second_hidden,activation='relu',name='hidden2')(x)
    out=layers.Dense(1,activation='sigmoid',name='output')(x)
    return keras.Model(inp,out)

def f1_only(y,p,threshold=.5):
    y=np.asarray(y).reshape(-1).astype(bool);prediction=np.asarray(p).reshape(-1)>=threshold
    tp=np.count_nonzero(y&prediction);fp=np.count_nonzero(~y&prediction);fn=np.count_nonzero(y&~prediction)
    return float(2*tp/max(2*tp+fp+fn,1))

def select_threshold(y,p):
    # Validation-only F1 selection; deterministic tie break nearest 0.5.
    thresholds=np.unique(np.r_[np.linspace(.05,.95,91),.5])
    scores=np.array([f1_only(y,p,t) for t in thresholds])
    best=np.flatnonzero(np.isclose(scores,scores.max(),rtol=0,atol=1e-12))
    return float(thresholds[best[np.argmin(np.abs(thresholds[best]-.5))]])

def metrics(y,p,threshold=.5):
    y=np.asarray(y).reshape(-1).astype(bool);p=np.asarray(p).reshape(-1).astype(float)
    pred=p>=threshold;tp=int(np.sum(y&pred));tn=int(np.sum(~y&~pred))
    fp=int(np.sum(~y&pred));fn=int(np.sum(y&~pred))
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1)
    q=np.clip(p,1e-7,1-1e-7)
    # Tie-aware AUROC using averaged ranks; AP uses groups of tied scores.
    order=np.argsort(p,kind='stable');sorted_p=p[order];ranks=np.empty(len(p),float)
    starts=np.r_[0,np.flatnonzero(np.diff(sorted_p))+1];ends=np.r_[starts[1:],len(p)]
    for a,b in zip(starts,ends):ranks[order[a:b]]=(a+1+b)/2
    pos=int(y.sum());neg=len(y)-pos
    auc=(ranks[y].sum()-pos*(pos+1)/2)/(pos*neg) if pos and neg else float('nan')
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    endpoints=np.r_[np.flatnonzero(np.diff(ps)),len(ps)-1]
    cum=np.cumsum(ys)[endpoints];rec=cum/max(pos,1);prec=cum/(endpoints+1)
    ap=float(np.sum(np.diff(np.r_[0,rec])*prec))
    ece=0.
    for low,high in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        inside=(p>=low)&((p<high) if high<1 else (p<=high))
        if inside.any():ece+=inside.mean()*abs(p[inside].mean()-y[inside].mean())
    denominator=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    return dict(accuracy=float(np.mean(pred==y)),balanced_accuracy=(recall+specificity)/2,
        precision=precision,recall=recall,f1=2*tp/max(2*tp+fp+fn,1),
        mcc=(tp*tn-fp*fn)/denominator if denominator else 0.,auroc=float(auc),average_precision=ap,
        bce=float(-np.mean(y*np.log(q)+(~y)*np.log(1-q))),brier=float(np.mean((p-y)**2)),
        mae=float(np.mean(np.abs(p-y))),ece=float(ece),tp=tp,tn=tn,fp=fp,fn=fn)

def prox_step(layer,next_layer,method,step,cfg):
    # Proximal shrinkage makes exact zero weights/groups, unlike a smooth penalty.
    w=layer.kernel;bias=layer.bias
    if method in {'lasso','set_lasso'}:
        w.assign(tf.sign(w)*tf.maximum(tf.abs(w)-step*cfg.l1_strength,0.))
    if method in {'group_lasso','fixed_group_set','adaptive_group_set'}:
        # L21 on incoming-neuron groups: each column AND its bias form a group.
        norm=tf.sqrt(tf.reduce_sum(tf.square(w),axis=0)+tf.square(bias))
        factor=tf.maximum(1.-step*cfg.l21_strength/tf.maximum(norm,1e-12),0.)
        w.assign(w*factor);bias.assign(bias*factor)
    w.assign(w*layer.mask)

def dense_growth_gradient(model,layer,x,y):
    """Dense candidate gradient at current sparse weights, using TRAINING data only."""
    features=keras.Model(model.input,layer.input)(x,training=False)
    effective=tf.convert_to_tensor(layer.kernel*layer.mask)
    with tf.GradientTape() as tape:
        tape.watch(effective)
        z=tf.nn.relu(tf.matmul(features,effective)+layer.bias)
        z=model.get_layer('hidden2')(z)
        p=model.get_layer('output')(z)
        loss=keras.losses.binary_crossentropy(y,p)
        loss=tf.reduce_mean(loss)
    gradient=tape.gradient(loss,effective)
    assert gradient is not None
    return gradient.numpy()

def rewire(layer,target_budget,rng,cfg,optimizer=None,gradient=None,allowed_columns=None):
    old=layer.mask.numpy().astype(bool);w=layer.kernel.numpy();core=old.copy()
    if allowed_columns is None:allowed_columns=np.ones(w.shape[1],bool)
    core[:,~allowed_columns]=False
    flat=w.ravel();keep=core.ravel()
    for sign in (-1,1):
        ix=np.flatnonzero(keep & ((flat<0) if sign==-1 else (flat>=0)))
        count=int(np.floor(cfg.rewire_fraction*len(ix)))
        if count:keep[ix[np.argsort(np.abs(flat[ix]))[:count]]]=False
    allowed=np.broadcast_to(allowed_columns,w.shape).ravel()
    target_budget=min(int(target_budget),int(allowed.sum()))
    if keep.sum()>target_budget:
        ix=np.flatnonzero(keep);keep[ix[np.argsort(np.abs(flat[ix]))[:int(keep.sum()-target_budget)]]]=False
    mask=keep.copy();need=int(target_budget-mask.sum());candidates=np.flatnonzero(~keep&allowed)
    if need:
        if gradient is None:chosen=rng.choice(candidates,need,replace=False)
        else:
            importance=np.abs(gradient).ravel()[candidates]
            # Seeded tiny noise resolves exactly tied gradients reproducibly.
            importance=importance+rng.uniform(0,1e-12,len(candidates))
            chosen=candidates[np.argpartition(importance,-need)[-need:]]
        mask[chosen]=True
    grown=mask&~keep;bound=np.sqrt(6/sum(w.shape))
    flat[~mask]=0.;flat[grown]=rng.uniform(-bound,bound,int(grown.sum()))
    layer.kernel.assign(flat.reshape(w.shape));layer.mask.assign(mask.reshape(w.shape).astype('float32'))
    bias=layer.bias.numpy();bias[~allowed_columns]=0.;layer.bias.assign(bias)
    reset=(~mask|grown).reshape(w.shape)
    if optimizer is not None:
        for variable in optimizer.variables:
            if tuple(variable.shape)==tuple(w.shape):variable.assign(tf.where(reset,tf.zeros_like(variable),variable))
            elif tuple(variable.shape)==tuple(bias.shape):variable.assign(tf.where(~allowed_columns,tf.zeros_like(variable),variable))
    assert int(layer.mask.numpy().sum())==target_budget
    assert np.all(layer.kernel.numpy()[layer.mask.numpy()==0]==0)
    return int(grown.sum())

def neuron_columns(layer,next_layer,budget,initial_budget,cfg):
    # Structured ablation, separate from L21 proximal shrinkage. Retain neurons
    # with strongest incoming/outgoing norm product; grows may revive columns.
    count=int(np.clip(round(.9*cfg.hidden*np.sqrt(budget/max(initial_budget,1))),8,cfg.hidden))
    w=layer.kernel.numpy();v=next_layer.kernel.numpy()
    score=np.sqrt((w*w).sum(0)+layer.bias.numpy()**2)*np.sqrt((v*v).sum(1))
    allowed=np.zeros(cfg.hidden,bool);allowed[np.argsort(score)[-count:]]=True
    return allowed

def compact_model(model):
    layer=model.get_layer('sparse_hidden')
    w=layer.kernel.numpy()*layer.mask.numpy();bias=layer.bias.numpy()
    active=np.flatnonzero(np.any(w!=0,axis=0)|(bias!=0))
    if not len(active):active=np.array([0])
    # Clone feature extractor, replace masked layer with a narrower native Dense.
    feature=keras.Model(model.input,layer.input)
    feature_copy=keras.models.clone_model(feature);feature_copy.set_weights(feature.get_weights())
    inp=feature_copy.input;z=feature_copy.output
    first=layers.Dense(len(active),activation='relu',name='compact_hidden')
    z=first(z)
    second=layers.Dense(model.get_layer('hidden2').units,activation='relu',name='compact_hidden2')
    z=second(z);last=layers.Dense(1,activation='sigmoid',name='compact_output');out=last(z)
    compact=keras.Model(inp,out)
    first.set_weights([w[:,active],bias[active]])
    next_w,next_bias=model.get_layer('hidden2').get_weights();second.set_weights([next_w[active],next_bias])
    last.set_weights(model.get_layer('output').get_weights())
    return compact,len(active)

def mac_counts(model):
    logical=0;dense=0
    for layer in model.layers:
        if isinstance(layer,MaskedDense):
            logical+=int(np.count_nonzero(layer.kernel.numpy()*layer.mask.numpy()));dense+=int(np.prod(layer.kernel.shape))
        elif isinstance(layer,layers.Dense):logical+=int(np.count_nonzero(layer.kernel.numpy()));dense+=int(np.prod(layer.kernel.shape))
        elif isinstance(layer,layers.Conv1D):
            value=int(layer.output.shape[1])*int(np.prod(layer.kernel.shape));logical+=value;dense+=value
    return logical,dense

def infer_function(model):
    @tf.function(input_signature=[tf.TensorSpec((None,600,1),tf.float32)])
    def infer(x):return model(x,training=False)
    return infer

def predict(infer,x,batch_size):
    return np.concatenate([infer(tf.convert_to_tensor(x[i:i+batch_size])).numpy() for i in range(0,len(x),batch_size)])

def latency(infer,batch,repeats):
    batch=tf.convert_to_tensor(batch)
    for _ in range(5):infer(batch).numpy()
    times=[]
    for _ in range(repeats):
        start=time.perf_counter();infer(batch).numpy();times.append((time.perf_counter()-start)*1000)
    return float(np.median(times)),float(np.percentile(times,95))

def run_one(architecture,method,target,seed,data,cfg,out,validation_only=False):
    start_run=time.perf_counter();keras.backend.clear_session();keras.utils.set_random_seed(seed)
    model=build_model(architecture,cfg);layer=model.get_layer('sparse_hidden');next_layer=model.get_layer('hidden2')
    weight_hash=hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.trainable_variables)).hexdigest()
    rng=np.random.default_rng(seed+1000)
    mask=initial_mask(tuple(layer.kernel.shape),cfg,rng)
    if method in {'dense','lasso','group_lasso'}:mask[:]=1
    layer.mask.assign(mask);layer.kernel.assign(layer.kernel*mask)
    mask_hash=hashlib.sha256(mask.tobytes()).hexdigest();initial_budget=int(mask.sum());total=mask.size
    optimizer=keras.optimizers.Adam(cfg.learning_rate)
    bce=keras.losses.BinaryCrossentropy();controller=StableBudgetController(cfg)
    @tf.function(input_signature=[tf.TensorSpec((None,600,1),tf.float32),tf.TensorSpec((None,1),tf.float32)])
    def step(x,y):
        with tf.GradientTape() as tape:loss=bce(y,model(x,training=True))
        gradients=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        prox_step(layer,next_layer,method,cfg.learning_rate,cfg)
        return loss
    infer=infer_function(model)
    tx,ty=data['train']['x'],data['train']['y'][:,target:target+1]
    vx,vy=data['validation']['x'],data['validation']['y'][:,target:target+1]
    shuffle=np.random.default_rng(seed+2000)
    candidates=[];history=[];training_time=0.;topology_time=0.;validation_time=0.
    for epoch in range(cfg.epochs):
        order=shuffle.permutation(len(tx));weighted=0.;start=time.perf_counter()
        for a in range(0,len(tx),cfg.batch_size):
            ix=order[a:a+cfg.batch_size];weighted+=float(step(tx[ix],ty[ix]).numpy())*len(ix)
        elapsed=time.perf_counter()-start;training_time+=elapsed
        start=time.perf_counter();val_prob=predict(infer,vx,cfg.batch_size);val=metrics(vy,val_prob)
        validation_time+=time.perf_counter()-start
        budget=int(layer.mask.numpy().sum());units=int(np.count_nonzero(np.any(layer.kernel.numpy()!=0,axis=0)|(layer.bias.numpy()!=0)))
        # Pareto candidates: retain only improvements in validation BCE OR size.
        state=dict(epoch=epoch+1,val_bce=val['bce'],units=units,budget=budget,weights=model.get_weights())
        dominated=any(c['val_bce']<=state['val_bce'] and c['budget']<=budget and c['units']<=units for c in candidates)
        if not dominated:
            candidates=[c for c in candidates if not(state['val_bce']<=c['val_bce'] and budget<=c['budget'] and units<=c['units'])]
            candidates.append(state)
        desired=budget;action='none';grown=0;gradient_seconds=0.;start=time.perf_counter()
        if epoch+1<cfg.epochs and method in DYNAMIC:
            if method in ADAPTIVE:desired,action=controller.decide(epoch,val['bce'],budget,total)
            else:desired,action=initial_budget,'fixed'
            gradient=None
            if method in GRADIENT:
                gstart=time.perf_counter();gix=order[:min(len(order),cfg.batch_size)]
                gradient=dense_growth_gradient(model,layer,tx[gix],ty[gix]);gradient_seconds=time.perf_counter()-gstart
            allowed=neuron_columns(layer,next_layer,desired,initial_budget,cfg) if method in {'adaptive_structured_set','fixed_group_set','adaptive_group_set'} else None
            grown=rewire(layer,desired,rng,cfg,optimizer,gradient,allowed)
        topology_elapsed=time.perf_counter()-start;topology_time+=topology_elapsed
        history.append(dict(epoch=epoch+1,train_bce=weighted/len(tx),val_bce=val['bce'],val_f1_at_05=val['f1'],
            budget=budget,density=budget/total,active_neurons=units,next_budget=desired,decision=action,regrown=grown,
            training_seconds=elapsed,topology_seconds=topology_elapsed,gradient_seconds=gradient_seconds))
    best=min(c['val_bce'] for c in candidates)
    eligible=[c for c in candidates if c['val_bce']<=best+cfg.checkpoint_tolerance]
    chosen=min(eligible,key=lambda c:(c['units'],c['budget'],c['val_bce'],c['epoch']))
    model.set_weights(chosen['weights'])
    vp=predict(infer,vx,cfg.batch_size);threshold=select_threshold(vy,vp)
    if validation_only:
        return dict(validation_bce=metrics(vy,vp)['bce'],config=asdict(cfg)),history
    testx=data['test']['x'];testy=data['test']['y'][:,target:target+1]
    prob=predict(infer,testx,cfg.batch_size);scores=metrics(testy,prob,threshold)
    native,units=compact_model(model);native_infer=infer_function(native)
    native_prob=predict(native_infer,testx,cfg.batch_size)
    np.testing.assert_allclose(prob,native_prob,rtol=1e-5,atol=2e-6)
    batch=testx[:min(cfg.batch_size,len(testx))]
    median,p95=latency(native_infer,batch,cfg.latency_repeats)
    # Common perturbations across methods: labels remain unchanged, no retuning.
    robustness={}
    noise_rng=np.random.default_rng(seed+target*100+9000)
    noise=noise_rng.normal(size=testx.shape).astype('float32')
    for amplitude in [.05,.1]:
        stress_prob=predict(native_infer,testx+amplitude*noise,cfg.batch_size)
        robustness['noise_'+str(amplitude).replace('.','p')+'_f1']=f1_only(testy,stress_prob,threshold)
    logical,dense=mac_counts(model);_,native_macs=mac_counts(native)
    model_bytes=sum(v.numpy().nbytes for v in model.weights)
    native_bytes=sum(v.numpy().nbytes for v in native.weights)
    row=dict(architecture=architecture,method=method,target=target,seed=seed,mode=cfg.mode,
        initial_weight_hash=weight_hash,initial_mask_hash=mask_hash,selected_epoch=chosen['epoch'],
        learning_rate=cfg.learning_rate,l1_strength=cfg.l1_strength,l21_strength=cfg.l21_strength,
        selected_neurons=units,initial_budget=initial_budget,selected_budget=chosen['budget'],threshold=threshold,
        f1_at_05=metrics(testy,prob,.5)['f1'],logical_active_macs=logical,dense_tensor_macs=dense,
        compact_dense_macs=native_macs,original_weight_bytes=model_bytes,compact_weight_bytes=native_bytes,
        compact_latency_median_ms=median,compact_latency_p95_ms=p95,inference_batch_size=len(batch),
        training_seconds=training_time,topology_seconds=topology_time,validation_seconds=validation_time,
        run_wall_seconds=time.perf_counter()-start_run,**robustness,**scores)
    stem=f'{architecture}_{method}_t{target}_s{seed}'
    save_csv(out/(stem+'_history.csv'),history)
    np.savez_compressed(out/(stem+'_predictions.npz'),labels=testy,probabilities=prob,threshold=threshold,
        indices=data['test']['indices'],groups=data['test']['groups'])
    if cfg.mode!='smoke':native.save(out/(stem+'_compact.keras'))
    if method in {'fixed_set','gradient_set','static_sparse','set_lasso','fixed_group_set'}:
        assert all(h['budget']==initial_budget for h in history)
    assert all(np.isfinite(v) for v in row.values() if isinstance(v,(float,int)))
    print(f'{architecture}/{method}/t{target}/s{seed}: F1={scores["f1"]:.3f}, units={units}, epoch={chosen["epoch"]}',flush=True)
    return row,history

def paired_statistics(rows,out,cfg):
    """Paired seed-and-test-group bootstrap; uncertainty is conditional on this dataset."""
    rng=np.random.default_rng(20261020)
    comparisons=[]
    candidate='adaptive_group_set'
    for architecture in cfg.architectures:
        for target in cfg.targets:
            for baseline in cfg.methods:
                if baseline==candidate or candidate not in cfg.methods:continue
                pairs=[];seed_deltas=[]
                for seed in cfg.seeds:
                    a=np.load(out/f'{architecture}_{candidate}_t{target}_s{seed}_predictions.npz')
                    b=np.load(out/f'{architecture}_{baseline}_t{target}_s{seed}_predictions.npz')
                    assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['indices'],b['indices'])
                    y=a['labels'].ravel();pa=a['probabilities'].ravel();pb=b['probabilities'].ravel()
                    ta=float(a['threshold']);tb=float(b['threshold']);groups=a['groups']
                    pairs.append((y,pa,pb,ta,tb,groups))
                    seed_deltas.append(f1_only(y,pa,ta)-f1_only(y,pb,tb))
                unique=np.unique(pairs[0][-1]);group_indices=[np.flatnonzero(pairs[0][-1]==g) for g in unique]
                bootstrap=[]
                # Resample the SAME seed/group indices for both methods.
                for _ in range(cfg.bootstrap_repeats):
                    seed_indices=rng.integers(len(pairs),size=len(pairs))
                    sampled_groups=rng.integers(len(unique),size=len(unique))
                    ix=np.concatenate([group_indices[i] for i in sampled_groups])
                    deltas=[]
                    for j in seed_indices:
                        y,pa,pb,ta,tb,_=pairs[j]
                        deltas.append(f1_only(y[ix],pa[ix],ta)-f1_only(y[ix],pb[ix],tb))
                    bootstrap.append(float(np.mean(deltas)))
                # Exact sign-flip test on paired seed effects; descriptive for small N.
                observed=abs(float(np.mean(seed_deltas)))
                permutations=[abs(float(np.mean(np.array(signs)*seed_deltas))) for signs in itertools.product([-1,1],repeat=len(seed_deltas))]
                pvalue=float(np.mean(np.array(permutations)>=observed-1e-12))
                comparisons.append(dict(architecture=architecture,target=target,candidate=candidate,baseline=baseline,
                    seeds=len(seed_deltas),test_groups=len(unique),mean_f1_difference=float(np.mean(seed_deltas)),
                    bootstrap_ci_low=float(np.quantile(bootstrap,.025)),bootstrap_ci_high=float(np.quantile(bootstrap,.975)),
                    seed_signflip_p=pvalue,holm_adjusted_p=1.,
                    interpretation='Conditional exploratory inference; no verified cross-house independence.'))
    # One Holm family across ALL reported candidate/baseline/appliance comparisons.
    if comparisons:
        order=np.argsort([r['seed_signflip_p'] for r in comparisons]);previous=0.;m=len(order)
        for rank,index in enumerate(order):
            previous=max(previous,min(1.,comparisons[index]['seed_signflip_p']*(m-rank)))
            comparisons[index]['holm_adjusted_p']=previous
        save_csv(out/'paired_uncertainty.csv',comparisons)
    return comparisons

def create_report(rows,histories,out,cfg):
    import matplotlib.pyplot as plt
    numeric=['f1','f1_at_05','balanced_accuracy','auroc','average_precision','mcc','bce','brier','ece',
        'selected_neurons','logical_active_macs','compact_dense_macs','compact_weight_bytes',
        'compact_latency_median_ms','training_seconds','topology_seconds','noise_0p05_f1','noise_0p1_f1']
    aggregate=[]
    for architecture,target,method in itertools.product(cfg.architectures,cfg.targets,cfg.methods):
        selected=[r for r in rows if (r['architecture'],r['target'],r['method'])==(architecture,target,method)]
        if not selected:continue
        item=dict(architecture=architecture,target=target,method=method,n_seeds=len(selected))
        for key in numeric:
            values=[r[key] for r in selected];item[key+'_mean']=float(np.mean(values))
            item[key+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.
        aggregate.append(item)
    save_csv(out/'aggregate_metrics.csv',aggregate)
    colors=plt.get_cmap('tab20')
    fig,axes=plt.subplots(len(cfg.architectures),len(cfg.targets),figsize=(4*len(cfg.targets),4*len(cfg.architectures)),squeeze=False)
    for i,architecture in enumerate(cfg.architectures):
        for j,target in enumerate(cfg.targets):
            ax=axes[i,j]
            for k,method in enumerate(cfg.methods):
                values=[r for r in rows if (r['architecture'],r['target'],r['method'])==(architecture,target,method)]
                ax.scatter([v['compact_dense_macs']/1e6 for v in values],[v['f1'] for v in values],color=colors(k/20),label=method,s=28)
            ax.set(title=f'{architecture}, target {target}',xlabel='Deployed dense MACs (millions)',ylabel='Held-out F1',ylim=(0,1.02))
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
    fig.suptitle(f'Controlled comparison — {cfg.mode}; no publication-readiness claim')
    fig.tight_layout(rect=(0,.14,1,.96));fig.savefig(out/'deployment_tradeoff.png',dpi=150);plt.close(fig)
    # CSV report records negative results and data/publication blockers explicitly.
    limitations=['These are synthetic-window presence-classification experiments, not power disaggregation.',
        'Source-window/house independence requires verification against source data.',
        'Masked training uses dense kernels; topology updates and candidate gradients add cost.',
        'Gradient regrowth is RigL-inspired, not an exact RigL reproduction.',
        'Proposed adaptive structured combination has not been established as literature-novel.',
        'Pilot and smoke modes are developmental; tuning after observing them must be disclosed.',
        'Seed/bootstrap uncertainty is conditional on these evaluation data.',
        'Native compact inference removes mask operations for ALL methods; unstructured zeros remain dense.',
        'Timing reflects this host only; independent hardware replication is required.']
    report=dict(mode=cfg.mode,completed_runs=len(rows),seeds=list(cfg.seeds),publication_ready=False,
                limitations=limitations,primary_metric='Validation-thresholded test F1; also report F1 at 0.5.',
                checkpoints='Minimum size among Pareto candidates within 0.005 validation BCE of best; no test selection.')
    (out/'research_status.json').write_text(json.dumps(report,indent=2))
    return aggregate

def tune_config(architecture,method,target,seed,data,cfg,out):
    # Equal trial count and train budget for every method. All selection is on
    # validation BCE; test arrays are not used by the validation-only branch.
    if cfg.search_trials<=1:return cfg
    rng=np.random.default_rng(seed+target*100+7000)
    trials=[]
    for trial in range(cfg.search_trials):
        candidate=replace(cfg,epochs=min(cfg.search_epochs,cfg.epochs),search_trials=1,
            learning_rate=float(10**rng.uniform(-3.5,-2.5)),
            l1_strength=float(10**rng.uniform(-6,-2)),l21_strength=float(10**rng.uniform(-4,-1)))
        result,_=run_one(architecture,method,target,seed,data,candidate,out,validation_only=True)
        trials.append(dict(trial=trial,validation_bce=result['validation_bce'],
            learning_rate=candidate.learning_rate,l1_strength=candidate.l1_strength,l21_strength=candidate.l21_strength))
    save_csv(out/f'{architecture}_{method}_t{target}_s{seed}_validation_search.csv',trials)
    best=min(trials,key=lambda r:(r['validation_bce'],r['trial']))
    return replace(cfg,learning_rate=best['learning_rate'],l1_strength=best['l1_strength'],l21_strength=best['l21_strength'])

def run_experiment(data_dir,out,cfg=None,metadata_path=None,resume=True):
    cfg=cfg or Config();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=load_data(data_dir,cfg,metadata_path)
    configuration=dict(config=asdict(cfg),data=data['audit'],tensorflow=tf.__version__,
        python=sys.version,platform=platform.platform(),devices=[d.name for d in tf.config.list_physical_devices()],
        source_sha256=sha_file(__file__))
    config_path=out/'config.json'
    # Never mix old results with a changed protocol, data, or implementation.
    if config_path.exists() and json.loads(config_path.read_text())!=json.loads(json.dumps(configuration)):
        raise ValueError('Output directory contains a different protocol/source/data; choose a new directory.')
    config_path.write_text(json.dumps(configuration,indent=2))
    for name in ['train','validation','test']:
        indices=data[name]['indices'];labels=data[name]['y']
        records=[dict(row_index=int(index),group_id=str(group),input_sha256=str(fingerprint),
                      **{f'label_{t}':int(labels[j,t]) for t in range(4)})
                 for j,(index,group,fingerprint) in enumerate(zip(indices,data[name]['groups'],data[name]['input_hashes']))]
        save_csv(out/(name+'_manifest.csv'),records)
    rows=[];histories={}
    for architecture,target,seed in itertools.product(cfg.architectures,cfg.targets,cfg.seeds):
        # Counterbalance execution order instead of timing every baseline first.
        methods=list(cfg.methods);np.random.default_rng(seed+target*17).shuffle(methods)
        paired=[]
        for method in methods:
            stem=f'{architecture}_{method}_t{target}_s{seed}';cache=out/(stem+'_run.json')
            if resume and cache.exists():
                saved=json.loads(cache.read_text());row,history=saved['metrics'],saved['history']
            else:
                run_cfg=tune_config(architecture,method,target,seed,data,cfg,out)
                row,history=run_one(architecture,method,target,seed,data,run_cfg,out)
                cache.write_text(json.dumps(dict(metrics=row,history=history),indent=2))
            rows.append(row);paired.append(row);histories[(architecture,method,target,seed)]=history
            save_csv(out/'metrics.csv',rows)
        hashes={r['initial_weight_hash'] for r in paired};assert len(hashes)==1
        sparse_hashes={r['initial_mask_hash'] for r in paired if r['method'] not in {'dense','lasso','group_lasso'}}
        assert len(sparse_hashes)<=1
    create_report(rows,histories,out,cfg)
    paired_statistics(rows,out,cfg)
    return rows,histories

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['smoke','pilot','full'],default='smoke')
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--out',type=Path,default=Path('runs/smoke'))
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--architectures',choices=['DNN','CNN'],nargs='+')
    parser.add_argument('--methods',choices=METHODS,nargs='+')
    parser.add_argument('--targets',type=int,nargs='+')
    args=parser.parse_args();cfg=Config.full() if args.mode=='full' else Config.pilot() if args.mode=='pilot' else Config()
    for name in ['epochs','seeds','architectures','methods','targets']:
        value=getattr(args,name)
        if value is not None:setattr(cfg,name,tuple(value) if isinstance(value,list) else value)
    run_experiment(args.data_dir,args.out,cfg,args.metadata)
