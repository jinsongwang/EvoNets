"""Final validation-only width extension: 96 units, declared before test evaluation."""
# Reuse the adaptive engine; keep a single implementation of sparse layers.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adaptive_sparsity"))

import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[2]/'mpl-cache'))
import argparse,json
import tensorflow as tf
from research_set import load_data,Config,sha_file
from train_refine import refine
from train_accuracy import save_csv

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[1] / 'data and requirements')
 a=p.parse_args();out=a.out
 tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(1)
 tf.config.experimental.enable_op_determinism()
 if any(out.glob('*_test.json')):raise ValueError('No extensions after test evaluation.')
 assert (out/'REFINEMENT_COMPLETE.json').exists()
 config=json.loads((out/'protocol.json').read_text());sources=json.loads((out/'refinement_sources.json').read_text())
 plan=dict(method='refine_96',reason='Validation indicates T2 is sensitive to the stronger width reduction; final candidate extension before test.',
           source_rule='Same validation-selected source as width 80/90',epochs=25,source_epochs=50,
           retained_units=96,final_density=.20,primitive_sha256=sha_file(Path(__file__).with_name('train_refine.py')),
           implementation_sha256=sha_file(__file__),test_used=False)
 (out/'conservative_plan.json').write_text(json.dumps(plan,indent=2))
 full=load_data(a.data_dir,Config(mode='accuracy',targets=tuple(config['targets'])))
 data={k:full[k] for k in ['train','validation']};rows=[]
 for t in config['targets']:
  for seed in config['seeds']:
   cache=out/f'CNN_refine_96_t{t}_s{seed}_selection.json'
   if cache.exists():r=json.loads(cache.read_text())
   else:r=refine('refine_96',t,seed,out/f'CNN_{sources[str(t)]}_t{t}_s{seed}.keras',data,out)
   rows.append(r);save_csv(out/'conservative_validation_results.csv',rows)
 (out/'CONSERVATIVE_COMPLETE.json').write_text(json.dumps(dict(runs=len(rows),test_evaluation_performed=False)))
if __name__=='__main__':main()
