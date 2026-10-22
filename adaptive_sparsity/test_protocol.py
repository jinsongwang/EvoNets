import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from tensorflow import keras
from protocol import load_label_schema, storage_accounting, macro_summary, paired_macro_statistics
from research_set import (Config, StableBudgetController, build_model, rewire, neuron_columns,
                          reset_adam_slots, f1_only, select_threshold, run_one, paired_statistics)

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        keras.backend.clear_session();keras.utils.set_random_seed(7)

    def test_labels_are_intended_not_falsely_verified(self):
        schema=load_label_schema(Path(__file__).parents[1])
        self.assertFalse(schema['source_verified'])
        self.assertEqual([r['name'] for r in schema['labels']],
                         ['washing machine','dishwasher','refrigerator','microwave'])
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);(folder/'samY.csv').write_text('id,V1,V2,V3,V4\n')
            for r in schema['labels']:r['source_verified']=True
            (folder/'label_schema.json').write_text(json.dumps(schema))
            with self.assertRaisesRegex(ValueError,'verification_evidence'):load_label_schema(folder)
            schema['labels'][0]['csv_column']='wrong'
            (folder/'label_schema.json').write_text(json.dumps(schema))
            with self.assertRaisesRegex(ValueError,'columns'):load_label_schema(folder)

    def test_f1_zero_denominator_and_deterministic_threshold_tie(self):
        self.assertEqual(f1_only([0,0],[0.,0.]),0.)
        self.assertEqual(select_threshold([0,0],[0.,0.]),.5)
        self.assertEqual(f1_only([1],[.5],.5),1.)

    def test_macro_averages_targets_before_seed_sd(self):
        rows=[dict(architecture='DNN',method='dense',target=t,seed=s,f1=f)
              for s,values in [(1,[.1,.9]),(2,[.9,.1])] for t,f in enumerate(values)]
        result=macro_summary(rows,(0,1))[0]
        self.assertEqual(result['macro_positive_f1_mean'],.5)
        self.assertEqual(result['macro_positive_f1_seed_sd'],0.)
        with self.assertRaisesRegex(ValueError,'exactly one'):macro_summary(rows[:-1],(0,1))

    def test_macro_uncertainty_pairs_seeds_after_averaging_targets(self):
        rows=[]
        for seed in [1,2,3]:
            for target in [0,1]:
                rows.extend([dict(architecture='DNN',method='dense',seed=seed,target=target,f1=.6),
                             dict(architecture='DNN',method='adaptive_group_set',seed=seed,target=target,f1=.7 if target==0 else .5)])
        result=paired_macro_statistics(rows,(0,1),20)[0]
        self.assertAlmostEqual(result['mean_macro_f1_difference'],0.)
        self.assertAlmostEqual(result['seed_bootstrap_ci_low'],0.)
        self.assertEqual(result['seed_signflip_p'],1.)
        with self.assertRaisesRegex(ValueError,'complete aligned'):paired_macro_statistics(rows[:-1],(0,1),20)

    def test_storage_counts_masks_and_batchnorm_state(self):
        model=build_model('CNN',Config(hidden=10))
        report=storage_accounting(model)
        self.assertEqual(report['total_bytes'],sum(v.numpy().nbytes for v in model.weights))
        self.assertEqual(report['mask_bytes'],model.get_layer('sparse_hidden').mask.numpy().nbytes)
        bn=[r for r in report['variables'] if r['category']=='batch_normalization']
        self.assertEqual(len(bn),4)
        self.assertEqual(sum(not r['trainable'] for r in bn),2)

    def test_optimizer_slots_do_not_reset_an_unrelated_same_shape_layer(self):
        cfg=Config(hidden=3,second_hidden=3);model=build_model('DNN',cfg)
        opt=keras.optimizers.Adam();opt.build(model.trainable_variables)
        sparse=model.get_layer('sparse_hidden');other=model.get_layer('hidden2')
        for slot in opt._momentums:slot.assign(np.ones(slot.shape))
        reset_adam_slots(opt,sparse.bias,np.array([True,False,False]))
        np.testing.assert_array_equal(opt._momentums[opt._get_variable_index(sparse.bias)].numpy(),[0,1,1])
        np.testing.assert_array_equal(opt._momentums[opt._get_variable_index(other.bias)].numpy(),[1,1,1])

    def test_controller_streak_reset_and_cooldown(self):
        c=StableBudgetController(Config(controller_patience=2,cooldown=3,warmup_epochs=1))
        self.assertEqual(c.decide(0,.5,100,1000),(100,'hold'))
        self.assertEqual(c.decide(1,.5,100,1000),(90,'shrink'))
        self.assertEqual(c.decide(2,.7,90,1000),(90,'cooldown'))
        self.assertEqual(c.decide(3,.7,90,1000),(90,'cooldown'))
        self.assertEqual(c.decide(4,.7,90,1000),(99,'grow'))
        self.assertEqual((c.good,c.bad),(0,0))
        self.assertEqual(c.best,.5)

    def test_expanding_neuron_count_revives_zeroed_columns(self):
        cfg=Config(hidden=20);model=build_model('DNN',cfg);layer=model.get_layer('sparse_hidden')
        next_layer=model.get_layer('hidden2');rng=np.random.default_rng(1)
        allowed=neuron_columns(layer,next_layer,1000,4000,cfg)
        rewire(layer,1000,rng,cfg,allowed_columns=allowed)
        zeroed=~allowed
        expanded=neuron_columns(layer,next_layer,4000,4000,cfg)
        self.assertTrue(np.any(expanded&zeroed))
        rewire(layer,4000,rng,cfg,allowed_columns=expanded)
        self.assertTrue(np.any(layer.kernel.numpy()[:,zeroed]!=0))

    def test_three_matched_controls_export_same_dense_budget(self):
        cfg=Config(hidden=12,epochs=2,targets=(0,),latency_repeats=2,matched_hidden=8)
        rng=np.random.default_rng(3)
        data={name:dict(x=rng.normal(size=(12,600,1)).astype('float32'),
                       y=np.tile([[0,1,0,1],[1,0,1,0]],(6,1)).astype('float32'),
                       indices=np.arange(12),groups=np.arange(12)) for name in ['train','validation','test']}
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            for architecture in ['DNN','CNN']:
                rows=[]
                for method in ['matched_dense','matched_static_sparse','magnitude_prune']:
                    row,_=run_one(architecture,method,0,7,data,cfg,folder);rows.append(row)
                    self.assertEqual(row['selected_neurons'],8)
                    self.assertEqual(row['compact_mask_bytes'],0)
                    self.assertEqual(row['compaction_threshold_disagreements'],0)
                    self.assertGreater(row['compact_archive_bytes'],row['compact_weight_bytes'])
                    if method=='magnitude_prune':self.assertEqual(row['pruning_epoch'],2)
                self.assertEqual(len({r['compact_weight_bytes'] for r in rows}),1)
                self.assertEqual(len({r['compact_dense_macs'] for r in rows}),1)

if __name__=='__main__':unittest.main()
