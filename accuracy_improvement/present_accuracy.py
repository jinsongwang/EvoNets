"""Create publication-style curves/tables from completed, measured experiments."""
import os
from pathlib import Path
import argparse,tempfile
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--historical-results',type=Path,default=ROOT/'adaptive_sparsity/results.csv')
args=parser.parse_args()
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'evonets-mpl-cache'))
import csv,json,shutil,zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RUN=args.run_dir;OUT=args.out
def historical_rows():
 raw=list(csv.DictReader(args.historical_results.open()));out=[]
 for method in ['fixed_set','adaptive_group_set']:
  group=[r for r in raw if r['architecture']=='CNN' and r['method']==method]
  perseed=[np.mean([float(r['f1']) for r in group if int(r['seed'])==seed]) for seed in [11,23,37]]
  out.append(dict(method='historical_'+method+'_10',epochs=10,macro_f1_mean=float(np.mean(perseed)),macro_f1_seed_sd=float(np.std(perseed,ddof=1)),
      accuracy_mean=float(np.mean([float(r['accuracy']) for r in group])),compact_weight_bytes_mean=float(np.mean([float(r['compact_weight_bytes']) for r in group])),
      compact_dense_macs_mean=float(np.mean([float(r['compact_dense_macs']) for r in group])),
      selected_neurons_mean=float(np.mean([float(r['selected_neurons']) for r in group])),
      protocol='Historical pilot: 10 epochs, constant learning rate, size/BCE-based checkpoint selection; not a matched training-budget control.'))
 return out

NAMES={'fixed_set':'Fixed SET 100','dense_teacher':'Dense teacher 100','dense_80':'Dense 80','dense_64':'Dense 64',
       'gradual_80':'Gradual pruning 80','distill_80':'Pruning + KD 80','distill_64':'Pruning + KD 64',
       'validation_policy':'Selected pruned student',
       'refine_96':'Warm-start pruning 96','refine_90':'Warm-start pruning 90','refine_80':'Warm-start pruning 80','refined_policy':'Selected compressed policy'}
COLORS={'fixed_set':'#333333','dense_teacher':'#0072B2','dense_80':'#56B4E9','dense_64':'#999999',
        'gradual_80':'#E69F00','distill_80':'#009E73','distill_64':'#CC79A7','validation_policy':'#D55E00','refine_96':'#332288','refine_90':'#882255','refine_80':'#44AA99','refined_policy':'#AA4499'}
FOCUS=['fixed_set','dense_teacher','gradual_80','distill_80','distill_64']
def read(path):return list(csv.DictReader(path.open()))
def val(row,key):return float(row[key])
def savefig(fig,name):
 fig.savefig(OUT/(name+'.png'),dpi=240,bbox_inches='tight',facecolor='white')
 fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def series(method,key,target=None):
 seeds=[]
 for seed in protocol['seeds']:
  curves=[]
  for t in ([target] if target is not None else protocol['targets']):
   rows=read(RUN/f'CNN_{method}_t{t}_s{seed}_history.csv')
   curves.append([float(r[key]) for r in rows])
  seeds.append(np.mean(curves,axis=0))
 a=np.array(seeds);return a.mean(0),a.std(0,ddof=1)
def curves():
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 epochs=np.arange(1,protocol['epochs']+1)
 fig,axes=plt.subplots(2,2,figsize=(12,8))
 for ax,key,title,scale in zip(axes.ravel(),['train_bce','val_bce','val_f1','first_layer_sparsity_vs_full100'],
                              ['Training BCE','Validation BCE','Validation macro F1','First-layer sparsity relative to width 100'],[1,1,100,100]):
  for method in FOCUS:
   mean,sd=series(method,key);ax.plot(epochs,mean*scale,label=NAMES[method],color=COLORS[method],lw=1.6)
   ax.fill_between(epochs,(mean-sd)*scale,(mean+sd)*scale,color=COLORS[method],alpha=.10)
  ax.axvline(15,color='#999999',ls=':',lw=.8);ax.axvline(35,color='#999999',ls=':',lw=.8)
  ax.set(title=title,xlabel='Epoch',ylabel='Percent' if scale==100 else 'Binary cross-entropy');ax.grid(alpha=.15)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.03),ncol=3,frameon=False)
 fig.suptitle('CNN training and compression on all four targets',fontsize=14)
 fig.text(.5,.006,'Mean across targets within each seed; shading = sample SD across 3 seeds. Pruning: epochs 16–35.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.14,1,.95));savefig(fig,'training-curves')
 fig,axes=plt.subplots(2,2,figsize=(12,7))
 for ax,t in zip(axes.ravel(),protocol['targets']):
  for method in FOCUS:
   mean,sd=series(method,'val_f1',t);ax.plot(epochs,100*mean,color=COLORS[method],label=NAMES[method],lw=1.5)
   ax.fill_between(epochs,100*(mean-sd),100*(mean+sd),color=COLORS[method],alpha=.09)
  ax.set(title=f'Target T{t}',xlabel='Epoch',ylabel='Validation positive-class F1 (%)');ax.grid(alpha=.15)
 handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
 fig.suptitle('Validation F1 by target',fontsize=14);fig.tight_layout(rect=(0,.10,1,.95));savefig(fig,'target-learning-curves')
 fig,ax=plt.subplots(figsize=(8,5))
 for row in summary:
  m=row['method'];x=val(row,'compact_weight_bytes_mean')/1024;y=100*val(row,'macro_f1_mean')
  ax.errorbar(x,y,yerr=100*val(row,'macro_f1_seed_sd'),fmt='D' if m in ['validation_policy','refined_policy'] else 'o',
              color=COLORS[m],ms=8 if m in ['validation_policy','refined_policy'] else 6,capsize=3,label=NAMES[m])
 ax.set(xlabel='Compact stored weights (KiB)',ylabel='Held-out macro positive-class F1 (%)',title='Accuracy and deployed model size')
 ax.grid(alpha=.15);ax.legend(fontsize=8,loc='best');fig.tight_layout();savefig(fig,'accuracy-size-tradeoff')

def refinement_curves():
 fig,axes=plt.subplots(1,3,figsize=(12,3.3))
 for ax,key,title,scale in zip(axes,['val_f1','val_bce','hidden_units'],['Validation macro F1','Validation BCE','Retained hidden neurons'],[100,1,1]):
  for method in ['refine_96','refine_90','refine_80']:
   mean,sd=series(method,key);x=np.arange(1,len(mean)+1)
   ax.plot(x,scale*mean,label=NAMES[method],color=COLORS[method]);ax.fill_between(x,scale*(mean-sd),scale*(mean+sd),color=COLORS[method],alpha=.15)
  ax.axvline(15,color='#888888',ls=':',lw=.8);ax.set(title=title,xlabel='Fine-tuning epoch');ax.grid(alpha=.15)
 axes[0].set_ylabel('F1 (%)');axes[0].legend(fontsize=8)
 fig.suptitle('Warm-start compression after 50 source-training epochs',fontsize=12);fig.tight_layout();savefig(fig,'warm-start-curves')

def historical_comparison():
 history=historical_rows()
 old=next(r for r in history if r['method']=='historical_fixed_set_10')
 baseline=next(r for r in summary if r['method']=='fixed_set')
 selected=next(r for r in summary if r['method']=='refined_policy')
 compact=next(r for r in summary if r['method']=='validation_policy')
 headers=['Experiment','Macro F1 (%)\nmean ± seed SD','Accuracy (%)','Stored weights\n(KiB)','Weight reduction\nvs paper baseline','Total training\nepochs']
 rows=[]
 for label,r,epochs in [('Paper fixed SET',old,10),('New fixed SET',baseline,50),('Pruned-student policy',compact,val(compact,'total_training_epochs_including_teacher_mean')),('Extended compressed policy',selected,val(selected,'total_training_epochs_including_teacher_mean'))]:
  rows.append([label,f"{100*float(r['macro_f1_mean']):.2f} ± {100*float(r['macro_f1_seed_sd']):.2f}",
               f"{100*float(r['accuracy_mean']):.2f}",f"{float(r['compact_weight_bytes_mean'])/1024:.2f}",
               f"{100*(1-float(r['compact_weight_bytes_mean'])/float(old['compact_weight_bytes_mean'])):.1f}%",f"{epochs:g}"])
 fig,ax=plt.subplots(figsize=(12,3.6));ax.axis('off')
 t=ax.table(cellText=rows,colLabels=headers,cellLoc='center',loc='center',colWidths=[.25,.20,.12,.15,.18,.13])
 t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.0)
 for (i,j),cell in t.get_celld().items():
  cell.set_edgecolor('#D9D9D9');cell.set_linewidth(.6)
  if i==0:cell.set_facecolor('#E7E6E6');cell.set_text_props(weight='bold');cell.set_height(.24)
  if i==4:cell.set_facecolor('#F0F5F3');cell.set_text_props(weight='bold')
 fig.suptitle('Accuracy and compression relative to the paper and a stronger baseline',fontsize=13)
 fig.text(.5,.01,'CNN, four targets, three seeds. Historical and new protocols differ. Extra teacher/pretraining cost is included; this is not an equal-total-training-budget comparison.',ha='center',fontsize=8)
 savefig(fig,'paper-comparison-table')
 with (OUT/'historical_pilot_reference.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
 with (OUT/'paper-comparison.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow([h.replace('\n',' ') for h in headers]);w.writerows(rows)


def table():
 headers=['Method','Macro F1\nmean ± SD (%)','Accuracy\n(%)','ΔF1 vs SET\n(pp)','Weights\n(KiB)','Weight\nreduction (%)','Dense MACs\n(thousands)','Active first-\nlayer edges','Original-matrix\nsparsity (%)','Total train\nepochs']
 data=[]
 for r in summary:
  data.append([NAMES[r['method']],f"{100*val(r,'macro_f1_mean'):.2f} ± {100*val(r,'macro_f1_seed_sd'):.2f}",
   f"{100*val(r,'accuracy_mean'):.2f}",f"{val(r,'delta_f1_pp_vs_fixed_set'):+.2f}",
   f"{val(r,'compact_weight_bytes_mean')/1024:.2f}",f"{val(r,'weight_reduction_pct_vs_fixed_set'):.1f}",
   f"{val(r,'compact_dense_macs_mean')/1000:.2f}",f"{val(r,'first_layer_active_edges_mean'):,.0f}",
   f"{100*val(r,'first_layer_sparsity_vs_full100_mean'):.2f}",f"{val(r,'total_training_epochs_including_teacher_mean'):g}"])
 fig,ax=plt.subplots(figsize=(21,6.3));ax.axis('off')
 tab=ax.table(cellText=data,colLabels=headers,cellLoc='center',loc='center',colWidths=[.21,.12,.07,.08,.07,.09,.09,.10,.10,.08])
 tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2.1)
 for (i,j),cell in tab.get_celld().items():
  cell.set_edgecolor('#D9D9D9');cell.set_linewidth(.6)
  if i==0:cell.set_facecolor('#E7E6E6');cell.set_text_props(weight='bold');cell.set_height(.14)
  elif i==len(data):cell.set_facecolor('#F0F5F3');cell.set_text_props(weight='bold')
  if j==0 and i>0:cell.set_text_props(ha='left')
 fig.suptitle('Measured CNN accuracy and sparsity comparison',fontsize=15,y=.98)
 fig.text(.5,.02,'3 seeds; 4 targets. F1 SD is across seeds. Total epochs include source/teacher training. Positive ΔF1 indicates improvement over the new 50-epoch SET baseline.',ha='center',fontsize=9)
 savefig(fig,'comparison-table')
 return headers,data

if __name__=='__main__':
 assert (RUN/'EVALUATION_COMPLETE.json').exists()
 OUT.mkdir(parents=True,exist_ok=True);protocol=json.loads((RUN/'protocol.json').read_text())
 summary=read(RUN/'summary.csv');curves();refinement_curves();historical_comparison();headers,data=table()
 for name in ['summary.csv','seed_summary.csv','target_summary.csv','paired_seed_differences.csv','test_results.csv','validation_results.csv','validation_selected_policy.json','protocol.json','analysis_plan.json','refinement_plan.json','refinement_analysis_plan.json','refinement_sources.json','refinement_validation_results.csv','refined_selected_policy.json','conservative_plan.json','conservative_validation_results.csv']:
  shutil.copy2(RUN/name,OUT/name)
 policy=next(r for r in summary if r['method']==('refined_policy' if any(x['method']=='refined_policy' for x in summary) else 'validation_policy'))
 status='Observed joint F1 and compact-storage improvement' if val(policy,'delta_f1_pp_vs_fixed_set')>0 else 'Compression achieved; no mean F1 improvement over fixed SET'
 txt=f'''CNN accuracy and sparsity study\n\n{status}.\n\nValidation-selected compressed policy: macro F1 {100*val(policy,'macro_f1_mean'):.3f}% (seed SD {100*val(policy,'macro_f1_seed_sd'):.3f} percentage points); change versus the NEW 50-epoch fixed SET baseline {val(policy,'delta_f1_pp_vs_fixed_set'):+.3f} percentage points. Compact weight reduction {val(policy,'weight_reduction_pct_vs_fixed_set'):.2f}%; dense kernel MAC reduction {val(policy,'mac_reduction_pct_vs_fixed_set'):.2f}%; active first-layer edge reduction {val(policy,'edge_reduction_pct_vs_fixed_set'):.2f}%.\n\nData: 5,770 train, 1,474 validation, 1,863 held-out test windows; all four targets; seeds 11, 23, 37. The original seven methods use 50 student epochs, Adam and the same cosine learning-rate schedule. KD adds a separately trained 50-epoch teacher. Warm-start pruning starts from the stronger validation-selected dense/fixed-SET source and adds 25 fine-tuning epochs (75 total). Its extra training cost is explicit; this is not an equal-total-training-budget comparison. The reference is rerun under this schedule and validation-F1 selection, rather than reusing the older ten-epoch paper score.\n\nRecipe and checkpoint selection use validation only. The original compressed policy chooses among three methods; the extended policy also includes the three warm-start widths. Both use mean validation F1 across seeds per target, and all candidate results are reported. The warm-start and conservative-width extensions were motivated by validation results before any held-out evaluation. Test scores do not choose the policy. Learning curves show per-epoch validation thresholds; final test thresholds come only from the selected validation checkpoint.\n\nSparsity: original first-layer matrix has 118,400 potential edges. Original-matrix sparsity counts removed neurons and zero weights relative to that matrix. Within-retained-layer sparsity is separately recorded in test_results.csv. Dense exported arrays still store unstructured zeros; only neuron removal reduces their dimensions. Weight bytes include kernels, biases and BN state, not optimizer slots or archive overhead. MACs exclude bias, activation and normalization operations.\n\nInference is conditional on the supplied synthetic data. Source-window/house independence and appliance identities remain unverified. Three seeds provide limited uncertainty evidence; observed differences are not a statistical superiority or accuracy-equivalence claim. No energy or hardware speedup is claimed.\n'''
 old=historical_rows()[0]
 txt+=f"\nRelative to the paper's historical ten-epoch fixed SET: macro F1 change {100*(val(policy,'macro_f1_mean')-float(old['macro_f1_mean'])):+.3f} percentage points; binary accuracy change {100*(val(policy,'accuracy_mean')-float(old['accuracy_mean'])):+.3f} percentage points. This historical comparison also changes training length, learning-rate and checkpoint selection, so it cannot isolate the compression algorithm's effect.\n"
 baseline=next(r for r in summary if r['method']=='fixed_set')
 txt+=f"\nBinary accuracy change against the new fixed SET baseline: {100*(val(policy,'accuracy_mean')-val(baseline,'accuracy_mean')):+.6f} percentage points; both round to 91.53%. The observed gain over that stronger baseline is in F1 and stored model size.\n"
 (OUT/'README.txt').write_text(txt)
 (OUT/'presentation-table.json').write_text(json.dumps(dict(headers=headers,data=data,summary=summary)))
 print(txt)
