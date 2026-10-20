"""Focused correctness checks for the experimental sparsity mechanism."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from adaptive_set import Config, BudgetController, MaskedDense, initial_mask, evolve, build_model, active_macs

class AdaptiveSETTests(unittest.TestCase):
    def setUp(self):
        keras.backend.clear_session()
        keras.utils.set_random_seed(7)
    def test_controller_shrinks_and_recovers(self):
        controller=BudgetController(Config())
        self.assertEqual(controller.decide(0,.5,100,1000),(80,'shrink'))
        self.assertEqual(controller.decide(1,.8,80,1000),(100,'grow'))
        self.assertEqual(controller.decide(2,.8,600,1000),(600,'grow'))
        self.assertEqual(controller.decide(3,.4,20,1000),(20,'shrink'))
    def test_rewire_preserves_and_changes_exact_budget(self):
        layer=MaskedDense(10);layer(tf.ones((2,30)))
        rng=np.random.default_rng(9)
        mask=initial_mask((30,10),Config(epsilon=2),rng)
        layer.mask.assign(mask);layer.kernel.assign(layer.kernel*mask)
        for budget in [int(mask.sum()),40,100]:
            previous = int(layer.mask.numpy().sum())
            grown = evolve(layer,budget,rng)
            if budget >= previous:
                self.assertGreater(grown,0)
            self.assertEqual(int(layer.mask.numpy().sum()),budget)
            self.assertTrue(np.all(layer.kernel.numpy()[layer.mask.numpy()==0]==0))
    def test_mask_blocks_gradients(self):
        layer=MaskedDense(3);layer(tf.ones((2,6)))
        mask=np.ones((6,3),dtype=np.float32);mask[:2]=0
        layer.mask.assign(mask)
        with tf.GradientTape() as tape:
            value=tf.reduce_sum(layer(tf.ones((2,6))))
        gradient=tape.gradient(value,layer.kernel).numpy()
        self.assertTrue(np.all(gradient[:2]==0))
    def test_optimizer_moments_survive_on_retained_edges(self):
        layer=MaskedDense(3);layer(tf.ones((2,6)))
        optimizer=keras.optimizers.Adam();optimizer.build(layer.trainable_variables)
        for v in optimizer.variables:
            if tuple(v.shape)==tuple(layer.kernel.shape):v.assign(tf.ones_like(v))
        old=layer.kernel.numpy().copy()
        evolve(layer,12,np.random.default_rng(1),fraction=0.,optimizer=optimizer)
        retained=(layer.mask.numpy()==1)&(layer.kernel.numpy()==old)
        self.assertTrue(retained.any())
        self.assertEqual(int(optimizer.iterations),0)
        for v in optimizer.variables:
            if tuple(v.shape)==tuple(layer.kernel.shape):
                self.assertTrue(np.all(v.numpy()[retained]==1))
                self.assertTrue(np.all(v.numpy()[layer.mask.numpy()==0]==0))
    def test_saved_model_keeps_mask_and_predictions(self):
        model=build_model('DNN');layer=model.get_layer('adaptive_dense')
        layer.mask.assign(initial_mask(tuple(layer.kernel.shape),Config(),np.random.default_rng(2)))
        x=np.ones((2,600,1),dtype=np.float32)
        before=model(x,training=False).numpy()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'model.keras';model.save(path)
            loaded=keras.models.load_model(path)
            np.testing.assert_allclose(before,loaded(x,training=False).numpy(),rtol=1e-6)
            self.assertEqual(active_macs(model),active_macs(loaded))

if __name__=='__main__':unittest.main()
