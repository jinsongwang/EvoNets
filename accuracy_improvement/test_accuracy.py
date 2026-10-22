# Reuse the adaptive engine; keep a single implementation of sparse layers.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_sparsity"))
import tempfile,unittest
from pathlib import Path
import numpy as np
from tensorflow import keras
from train_accuracy import schedule,prune_exact,train_one
from research_set import Config,build_model

class AccuracyTests(unittest.TestCase):
    def test_cubic_schedule_endpoints_and_monotonicity(self):
        trajectory=[schedule(e,50,64)[:2] for e in range(1,51)]
        self.assertEqual(trajectory[14],(100,1.))
        self.assertEqual(trajectory[34],(64,.25))
        self.assertEqual(trajectory[-1],(64,.25))
        self.assertTrue(all(a[0]>=b[0] and a[1]>=b[1] for a,b in zip(trajectory,trajectory[1:])))
    def test_exact_pruning_freezes_mask_and_bias(self):
        keras.utils.set_random_seed(1);model=build_model('CNN',Config())
        layer=model.get_layer('sparse_hidden');next_layer=model.get_layer('hidden2')
        optimizer=keras.optimizers.Adam();optimizer.build(model.trainable_variables)
        allowed=np.ones(100,bool)
        allowed=prune_exact(layer,next_layer,80,.25,optimizer,allowed)
        self.assertEqual(allowed.sum(),80)
        self.assertEqual(layer.mask.numpy().sum(),1184*80*.25)
        self.assertTrue(np.all(layer.bias.numpy()[~allowed]==0))
        first=layer.mask.numpy().copy()
        allowed=prune_exact(layer,next_layer,64,.25,optimizer,allowed)
        self.assertTrue(np.all(layer.mask.numpy()<=first))
        self.assertEqual(layer.mask.numpy().sum(),1184*64*.25)
    def test_warm_start_refinement_exports_a_smaller_model_without_test_data(self):
        from train_refine import restore,refine
        from research_set import compact_model
        keras.utils.set_random_seed(2)
        source,_=compact_model(build_model('CNN',Config()))
        rng=np.random.default_rng(3)
        data={k:dict(x=rng.normal(size=(8,600,1)).astype('float32'),
                     y=np.tile([[0,1,0,1],[1,0,1,0]],(4,1)).astype('float32')) for k in ['train','validation']}
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);source.save(folder/'teacher.keras')
            row=refine('refine_90',0,11,folder/'teacher.keras',data,folder)
            self.assertEqual(row['selected_neurons'],90)
            self.assertLessEqual(row['first_layer_active_edges'],1184*90*.20)
            self.assertLess(row['compact_weight_bytes'],sum(v.numpy().nbytes for v in source.weights))
            self.assertEqual(row['student_epochs']+row['teacher_extra_epochs'],75)

    def test_recipes_are_selected_before_test_data_is_loaded(self):
        import json
        from unittest.mock import patch
        from evaluate_accuracy import evaluate,COMPRESSED,REFINED
        from train_accuracy import METHODS,save_csv
        from research_set import sha_file
        class StopBeforeTest(Exception):pass
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            config=dict(methods=list(METHODS),targets=[0],seeds=[11,23,37],
                        implementation_sha256=sha_file(Path(__file__).with_name('train_accuracy.py')))
            (folder/'protocol.json').write_text(json.dumps(config))
            (folder/'TRAINING_COMPLETE.json').write_text(json.dumps(dict(runs=21)))
            (folder/'REFINEMENT_COMPLETE.json').write_text(json.dumps(dict(runs=6)))
            rows=[]
            for method in METHODS:
                for seed in [11,23,37]:
                    rows.append(dict(method=method,target=0,seed=seed,
                                     val_f1=.8 if method=='gradual_80' else .7,compact_weight_bytes=100))
            save_csv(folder/'validation_results.csv',rows)
            extra=[dict(method=m,target=0,seed=seed,val_f1=.9 if m=='refine_90' else .75,compact_weight_bytes=90)
                   for m in REFINED for seed in [11,23,37]]
            save_csv(folder/'refinement_validation_results.csv',extra)
            def stop(*args,**kwargs):
                self.assertEqual(json.loads((folder/'validation_selected_policy.json').read_text()),{'0':'gradual_80'})
                self.assertEqual(json.loads((folder/'refined_selected_policy.json').read_text()),{'0':'refine_96' if (folder/'CONSERVATIVE_COMPLETE.json').exists() else 'refine_90'})
                raise StopBeforeTest()
            for conservative in [False,True]:
                if conservative:
                    (folder/'CONSERVATIVE_COMPLETE.json').write_text(json.dumps(dict(runs=3)))
                    save_csv(folder/'conservative_validation_results.csv',[
                        dict(method='refine_96',target=0,seed=seed,val_f1=.95,compact_weight_bytes=96)
                        for seed in [11,23,37]])
                with patch('evaluate_accuracy.load_data',side_effect=stop):
                    with self.assertRaises(StopBeforeTest):evaluate(folder,folder)

    def test_training_without_any_test_partition_and_distillation(self):
        rng=np.random.default_rng(7)
        data={k:dict(x=rng.normal(size=(8,600,1)).astype('float32'),
                     y=np.tile([[0,1,0,1],[1,0,1,0]],(4,1)).astype('float32')) for k in ['train','validation']}
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            for method,teacher in [('dense_teacher',None),('distill_64',np.full((8,1),.6,np.float32))]:
                r=train_one(method,0,11,data,out,epochs=3,batch_size=8,teacher=teacher)
                self.assertLess(r['val_compaction_max_abs_error'],2e-6)
                self.assertTrue((out/f'CNN_{method}_t0_s11.keras').is_file())
                if method=='distill_64':
                    self.assertEqual(r['selected_neurons'],64)
                    self.assertEqual(r['first_layer_active_edges'],1184*64*.25)
if __name__=='__main__':unittest.main()
