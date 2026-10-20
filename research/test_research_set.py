import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from research_set import (Config,StableBudgetController,build_model,prox_step,rewire,
    dense_growth_gradient,compact_model,metrics,select_threshold,f1_only,load_data,paired_statistics,tune_config)

class ResearchTests(unittest.TestCase):
    def setUp(self):keras.backend.clear_session();keras.utils.set_random_seed(3)
    def test_known_classification_metrics_and_ties(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.4,.35,.8])
        r=metrics(y,p)
        self.assertAlmostEqual(r['auroc'],.75);self.assertAlmostEqual(r['average_precision'],5/6)
        self.assertAlmostEqual(r['f1'],2/3);self.assertAlmostEqual(r['brier'],np.mean((p-y)**2))
        self.assertEqual(metrics(y,np.ones(4)*.5)['auroc'],.5)
        self.assertEqual(metrics(y,np.ones(4)*.5)['average_precision'],.5)
    def test_threshold_uses_passed_validation_only(self):
        y=np.array([0,0,1,1]);p=np.array([.05,.15,.3,.4])
        threshold=select_threshold(y,p)
        self.assertEqual(f1_only(y,p,threshold),1.)
        self.assertLess(threshold,.5)
    def test_l1_soft_threshold_exact_zero(self):
        cfg=Config(hidden=3);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        layer.kernel.assign(tf.ones_like(layer.kernel)*.00001)
        prox_step(layer,model.get_layer('hidden2'),'lasso',1.,replace(cfg,l1_strength=.001))
        self.assertTrue(np.all(layer.kernel.numpy()==0))
    def test_l21_zeroes_entire_neuron_and_bias(self):
        cfg=Config(hidden=3);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        w=np.ones(layer.kernel.shape,dtype=np.float32);w[:,0]=1e-7
        layer.kernel.assign(w);layer.bias.assign([1e-7,1.,1.])
        prox_step(layer,model.get_layer('hidden2'),'group_lasso',1.,replace(cfg,l21_strength=.001))
        self.assertTrue(np.all(layer.kernel.numpy()[:,0]==0));self.assertEqual(layer.bias.numpy()[0],0.)
        self.assertTrue(np.any(layer.kernel.numpy()[:,1]!=0))
    def test_dense_gradient_sees_inactive_edges(self):
        cfg=Config(hidden=5);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        mask=np.ones(layer.kernel.shape,dtype=np.float32);mask[:100]=0.;layer.mask.assign(mask)
        x=np.ones((4,600,1),np.float32);y=np.array([[0],[1],[0],[1]],np.float32)
        g=dense_growth_gradient(model,layer,x,y)
        self.assertTrue(np.isfinite(g).all());self.assertTrue(np.any(g[:100]!=0))
    def test_gradient_regrowth_prefers_largest_scores(self):
        cfg=Config(hidden=3,rewire_fraction=0.);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        layer.kernel.assign(tf.zeros_like(layer.kernel));layer.mask.assign(tf.zeros_like(layer.mask))
        g=np.zeros(layer.kernel.shape);g[10,1]=100.;g[20,2]=99.
        rewire(layer,2,np.random.default_rng(3),cfg,gradient=g)
        self.assertEqual(layer.mask.numpy()[10,1],1.);self.assertEqual(layer.mask.numpy()[20,2],1.)
    def test_rewiring_budget_and_neuron_constraints(self):
        cfg=Config(hidden=10);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        allowed=np.array([True]*6+[False]*4)
        rewire(layer,1200,np.random.default_rng(3),cfg,allowed_columns=allowed)
        self.assertEqual(layer.mask.numpy().sum(),1200)
        self.assertTrue(np.all(layer.kernel.numpy()[:,~allowed]==0))
        self.assertTrue(np.all(layer.bias.numpy()[~allowed]==0))
    def test_compaction_exact_and_reduces_storage(self):
        cfg=Config(hidden=10);model=build_model('CNN',cfg);layer=model.get_layer('sparse_hidden')
        allowed=np.array([True]*6+[False]*4)
        rewire(layer,1200,np.random.default_rng(3),cfg,allowed_columns=allowed)
        native,units=compact_model(model);self.assertEqual(units,6)
        x=np.random.default_rng(3).normal(size=(5,600,1)).astype('float32')
        np.testing.assert_allclose(model(x,training=False),native(x,training=False),rtol=1e-5,atol=2e-6)
        self.assertLess(sum(v.numpy().nbytes for v in native.weights),sum(v.numpy().nbytes for v in model.weights))
    def test_controller_patience(self):
        c=StableBudgetController(Config(controller_patience=2))
        self.assertEqual(c.decide(0,.5,100,1000),(100,'hold'))
        self.assertEqual(c.decide(1,.5,100,1000),(90,'shrink'))
        self.assertEqual(c.decide(2,.7,90,1000),(90,'hold'))
        self.assertEqual(c.decide(3,.7,90,1000),(99,'grow'))
    def test_metadata_groups_enforced(self):
        import csv
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            x=np.column_stack([np.arange(9),np.repeat(np.arange(1,10)[:,None],600,axis=1)])
            y=np.column_stack([np.arange(9),np.repeat((np.arange(9)%2)[:,None],4,axis=1)])
            np.savetxt(folder/'samX.csv',x,delimiter=',',header=','.join(['id']+list(map(str,range(600)))),comments='')
            np.savetxt(folder/'samY.csv',y,delimiter=',',header='id,a,b,c,d',comments='')
            records=[dict(row_index=i,split=['train','validation','test'][i//3],group_id=str(i)) for i in range(9)]
            path=folder/'provenance.csv'
            def write():
                with path.open('w',newline='') as h:
                    w=csv.DictWriter(h,fieldnames=['row_index','split','group_id']);w.writeheader();w.writerows(records)
            write();data=load_data(folder,Config(),path)
            self.assertEqual(data['audit']['counts'],dict(test=3,validation=3,train=3))
            records[6]['group_id']=records[0]['group_id'];write()
            with self.assertRaisesRegex(ValueError,'overlap'):load_data(folder,Config(),path)
    def test_paired_statistics_identical_predictions(self):
        cfg=Config(architectures=('DNN',),targets=(0,),methods=('adaptive_group_set','dense'),seeds=(1,2),bootstrap_repeats=20)
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for seed in cfg.seeds:
                for method in cfg.methods:
                    np.savez(folder/f'DNN_{method}_t0_s{seed}_predictions.npz',labels=[0,0,1,1],
                        probabilities=[.1,.2,.8,.9],indices=np.arange(4),groups=np.arange(4),threshold=.5)
            rows=paired_statistics([],folder,cfg)
            self.assertEqual(rows[0]['mean_f1_difference'],0.)
            self.assertEqual(rows[0]['bootstrap_ci_low'],0.)
            self.assertEqual(rows[0]['bootstrap_ci_high'],0.)
            self.assertEqual(rows[0]['holm_adjusted_p'],1.)
    def test_validation_search_without_test_data(self):
        cfg=Config(hidden=10,epochs=1,search_trials=2,search_epochs=1)
        rng=np.random.default_rng(3)
        data={name:dict(x=rng.normal(size=(16,600,1)).astype('float32'),
            y=np.tile([[0,1,0,1],[1,0,1,0]],(8,1)).astype('float32')) for name in ['train','validation']}
        with tempfile.TemporaryDirectory() as folder:
            selected=tune_config('DNN','lasso',0,42,data,cfg,Path(folder))
            self.assertGreater(selected.learning_rate,0.)
            self.assertTrue((Path(folder)/'DNN_lasso_t0_s42_validation_search.csv').exists())
    def test_optimizer_state_preserved_on_surviving_edges(self):
        cfg=Config(hidden=3);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        optimizer=keras.optimizers.Adam();optimizer.build(model.trainable_variables)
        optimizer.iterations.assign(5)
        for value in optimizer.variables:
            if tuple(value.shape)==tuple(layer.kernel.shape):value.assign(tf.ones_like(value))
        old=layer.kernel.numpy().copy()
        rewire(layer,1200,np.random.default_rng(8),cfg,optimizer=optimizer)
        retained=(layer.mask.numpy()==1)&(layer.kernel.numpy()==old)
        self.assertTrue(retained.any());self.assertEqual(int(optimizer.iterations),5)
        for value in optimizer.variables:
            if tuple(value.shape)==tuple(layer.kernel.shape):
                self.assertTrue(np.all(value.numpy()[retained]==1))
                self.assertTrue(np.all(value.numpy()[layer.mask.numpy()==0]==0))
    def test_compact_model_loads_without_custom_objects(self):
        cfg=Config(hidden=10);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        allowed=np.array([True]*6+[False]*4)
        rewire(layer,1200,np.random.default_rng(3),cfg,allowed_columns=allowed)
        native,_=compact_model(model);x=np.ones((2,600,1),np.float32)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'native.keras';native.save(path)
            loaded=keras.models.load_model(path)
            np.testing.assert_allclose(native(x,training=False),loaded(x,training=False),rtol=1e-6,atol=1e-7)

if __name__=='__main__':unittest.main()
