"""Fixed-cohort OOF prefix evaluation. No training; edit paths below."""
from pathlib import Path
import sys, csv, json, hashlib
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
RESULT_DIR=Path(r'E:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main8\output6_rebuttal\rebuttal_69_dataset_1234_fig_rcc_pooling_visit_direct_concat_td_conv_gru_modality_multimodal_temporal_baseline')
DATA_ROOT=Path(r'E:\wxy_codes\dataset_hydro_1234\raw')
FETALCLIP_WEIGHTS=ROOT/'FetalCLIP_weights.pt'
FETALCLIP_CONFIG=ROOT/'FetalCLIP_config.json'
OUTPUT_ROOT=ROOT/'output6_rebuttal/history_prefix_evaluation'
DEVICE='cuda'
MIN_VISITS=6
PREFIXES=[2,3,4,5]
SMOKE_TEST=False  # True: up to two patients per fold, 100 resamples; not formal results
SEED=42
N_BOOTSTRAP=10000
N_PERMUTATIONS=10000
REPLAY_ATOL=1e-4
METRICS=['Accuracy','Sensitivity','Specificity','Precision_PPV','NPV','F1','AUC','BrierScore','ECE']

def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(path,rows):
 if not rows:return
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def select_cohort(rows,minimum):
 groups={};seen=set()
 for r in rows:
  if r['sample_id'] in seen:raise ValueError('Duplicate sample ID')
  seen.add(r['sample_id']);groups.setdefault(r['case_id'],[]).append(r)
 selected=[]
 for case,rs in sorted(groups.items()):
  if len(rs)!=2 or {r['side'] for r in rs}!={'left','right'}:raise ValueError(f'{case}: expected two renal units')
  if len({r['fold_id'] for r in rs})!=1:raise ValueError('Patient spans folds')
  if min(int(r['num_visits']) for r in rs)>=minimum:selected.extend(rs)
 return selected

def prefix_sample(sample,n):
 if n is None:n=len(sample['visits'])
 if not 1<=n<=len(sample['visits']):raise ValueError('Invalid prefix length')
 out=dict(sample);out['visits']=sample['visits'][:n];out['num_visits']=n
 out['chronological_visit_ids']=[str(v['visit_id']) for v in out['visits']]
 return out

def metric_vector(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);q=p>=.5
 tp=np.sum(q & (y==1));tn=np.sum(~q & (y==0));fp=np.sum(q & (y==0));fn=np.sum(~q & (y==1))
 def div(a,b):return float(a/b) if b else 0.
 auc=np.nan
 if tp+fn and tn+fp:
  order=np.argsort(p);ys=y[order];ps=p[order]
  _,starts,counts=np.unique(ps,return_index=True,return_counts=True)
  positives=np.add.reduceat(ys,starts);negatives=counts-positives
  auc=float(np.sum(positives*(np.cumsum(negatives)-negatives+.5*negatives))/((tp+fn)*(tn+fp)))
 ece=0.;edges=np.linspace(0.,1.,11)
 for b in range(10):
  mask=(p>=edges[b]) & ((p<=edges[b+1]) if b==9 else (p<edges[b+1]))
  if mask.any():ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
 return np.array([div(tp+tn,len(y)),div(tp,tp+fn),div(tn,tn+fp),div(tp,tp+fp),div(tn,tn+fn),div(2*tp,2*tp+fp+fn),auc,np.mean((p-y)**2),ece])

def holm(values):
 p=np.asarray(values,dtype=float);result=np.full(len(p),np.nan);idx=np.flatnonzero(np.isfinite(p));order=idx[np.argsort(p[idx])]
 result[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(order)-np.arange(len(order)))))
 return result

def analyze(rows,run,nboot,nperm):
 stages=[f'first_{n}' for n in PREFIXES]+['full']
 lookup={(r['sample_id'],r['stage']):r for r in rows}
 ids=sorted({r['sample_id'] for r in rows})
 if len(lookup)!=len(rows) or len(rows)!=len(ids)*len(stages):raise ValueError('Missing/duplicate stage predictions')
 base=[lookup[(i,'full')] for i in ids];y=np.array([int(r['label']) for r in base])
 probs=np.array([[float(lookup[(i,stage)]['prob']) for i in ids] for stage in stages])
 patients=sorted({r['case_id'] for r in base});clusters=[np.array([j for j,r in enumerate(base) if r['case_id']==c]) for c in patients]
 fold_clusters={}
 for k,c in enumerate(patients):fold_clusters.setdefault(base[clusters[k][0]]['fold_id'],[]).append(k)
 point=np.array([metric_vector(y,p) for p in probs]);rng=np.random.default_rng(SEED)
 boot=np.empty((nboot,len(stages),len(METRICS)))
 for b in range(nboot):
  # Same resampled patients for every stage, preserve original fold composition.
  chosen=np.concatenate([rng.choice(v,len(v),replace=True) for v in fold_clusters.values()])
  idx=np.concatenate([clusters[c] for c in chosen])
  boot[b]=[metric_vector(y[idx],p[idx]) for p in probs]
  if (b+1)%1000==0:print(f'Bootstrap {b+1}/{nboot}',flush=True)
 summary=[]
 for j,stage in enumerate(stages):
  for m,metric in enumerate(METRICS):
   vals=boot[:,j,m];vals=vals[np.isfinite(vals)];ci=np.quantile(vals,[.025,.975]) if len(vals)>=2 else [np.nan,np.nan]
   summary.append(dict(stage=stage,metric=metric,n_patients=len(patients),n_renal_units=len(ids),positive_units=int(y.sum()),estimate=point[j,m],ci_low=ci[0],ci_high=ci[1],valid_bootstrap=len(vals)))
 write_csv(run/'stage_metrics.csv',summary)
 folds=[]
 for stage,p in zip(stages,probs):
  for fold in sorted(fold_clusters):
   idx=[j for j,r in enumerate(base) if r['fold_id']==fold]
   values=metric_vector(y[idx],p[idx]);folds.append(dict(stage=stage,fold_id=fold,n_patients=len(fold_clusters[fold]),n_renal_units=len(idx),**dict(zip(METRICS,values))))
 write_csv(run/'fold_metrics.csv',folds)
 pairs=[(j,len(stages)-1) for j in range(len(stages)-1)]+[(j,j+1) for j in range(len(stages)-2)]
 comparisons=[]
 for a,b in pairs:
  observed=point[b]-point[a];counts=np.zeros(len(METRICS));valid=np.zeros(len(METRICS))
  for iteration in range(nperm):
   swap=np.zeros(len(ids),dtype=bool)
   for chosen,idx in zip(rng.integers(0,2,len(clusters)),clusters):swap[idx]=bool(chosen)
   pa=np.where(swap,probs[b],probs[a]);pb=np.where(swap,probs[a],probs[b])
   diff=metric_vector(y,pb)-metric_vector(y,pa);finite=np.isfinite(diff)&np.isfinite(observed)
   valid+=finite;counts+=finite & (np.abs(diff)>=np.abs(observed)-1e-12)
  for m,metric in enumerate(METRICS):
   vals=boot[:,b,m]-boot[:,a,m];vals=vals[np.isfinite(vals)];ci=np.quantile(vals,[.025,.975]) if len(vals)>=2 else [np.nan,np.nan]
   comparisons.append(dict(stage_a=stages[a],stage_b=stages[b],metric=metric,difference_b_minus_a=observed[m],ci_low=ci[0],ci_high=ci[1],p_uncorrected=(counts[m]+1)/(valid[m]+1) if valid[m] else np.nan,p_holm=np.nan,valid_bootstrap=len(vals),valid_permutations=int(valid[m])))
  print(f'Permutation complete: {stages[b]} minus {stages[a]}',flush=True)
 for row,p in zip(comparisons,holm([r['p_uncorrected'] for r in comparisons])):row['p_holm']=p
 write_csv(run/'paired_comparisons.csv',comparisons)
 # Plot on a categorical axis: full history has different lengths per patient.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(12,4))
 for ax,metric in zip(axes,['AUC','Sensitivity','F1']):
  rs=[r for r in summary if r['metric']==metric];ax.plot(stages,[r['estimate'] for r in rs],'o-')
  ax.fill_between(stages,[r['ci_low'] for r in rs],[r['ci_high'] for r in rs],alpha=.15);ax.set(title=metric,ylim=(0,1));ax.tick_params(axis='x',rotation=25)
 fig.tight_layout();fig.savefig(run/'history_performance.png',dpi=200);plt.close(fig)

def main():
 import torch
 from models.longitudinal_gru_classifier import ModularGRUSideClassifier
 if not PREFIXES or max(PREFIXES)>=MIN_VISITS:raise ValueError('Prefixes must precede the last visit for every eligible patient')
 source=RESULT_DIR/'5fold_predictions.csv';all_rows=read_csv(source);selected=select_cohort(all_rows,MIN_VISITS)
 if not selected:raise ValueError('Empty eligible cohort')
 if SMOKE_TEST:
  rng=np.random.default_rng(SEED);keep=set()
  for f in sorted({r['fold_id'] for r in selected}):
   cases=sorted({r['case_id'] for r in selected if r['fold_id']==f});keep.update(rng.choice(cases,min(2,len(cases)),replace=False))
  selected=[r for r in selected if r['case_id'] in keep]
 print('Selected patients:',len({r['case_id'] for r in selected}),'renal units:',len(selected),flush=True)
 checkpoints={};meta={}
 for fold in sorted({r['fold_id'] for r in selected}):
  cp=RESULT_DIR/fold/'final_model.pt'
  if not cp.is_file():raise FileNotFoundError(f'Missing final trained checkpoint: {cp}')
  obj=torch.load(cp,map_location='cpu',weights_only=True)
  md=obj['metadata'];kwargs=md['model_kwargs']
  if md['fold_id']!=fold or md.get('checkpoint_role')!='final_outer_fold_model':raise ValueError('Wrong checkpoint role/fold')
  if md['temporal_experiment']!='baseline' or kwargs['input_modality']!='multimodal' or kwargs['image_pooling_type']!='rcc_pooling' or kwargs['temporal_backbone']!='td_conv_gru':raise ValueError('Expected full baseline model')
  train=set(md['train_case_ids']);test=set(md['test_case_ids'])
  fold_cases={r['case_id'] for r in all_rows if r['fold_id']==fold}
  if train & test or test!=fold_cases:raise ValueError('Checkpoint and OOF patient split differ')
  if set(md['test_sample_ids'])!={r['sample_id'] for r in all_rows if r['fold_id']==fold}:raise ValueError('Checkpoint sample IDs differ')
  checkpoints[fold]=cp;meta[fold]=md;del obj
 from data.dataset_online import collect_case_side_samples_online
 from fetalclip.fetalclip_wrapper import FetalCLIPFeatureExtractor
 samples={s['sample_id']:s for s in collect_case_side_samples_online(DATA_ROOT)}
 for r in selected:
  s=samples[r['sample_id']]
  if int(r['label'])!=s['label'] or r['case_id']!=s['case_id'] or r['side']!=s['side']:raise ValueError('Sample identity or label changed')
  if r['temporal_experiment']!='baseline' or r['input_modality']!='multimodal':raise ValueError('Non-baseline predictions')
  if len(s['visits'])!=int(r['num_visits']) or '|'.join(str(v['visit_id']) for v in s['visits'])!=r['visit_ids_used']:raise ValueError('Visit history changed')
  if r['visit_ids_chronological']!=r['visit_ids_used']:raise ValueError('History not chronological')
  records=json.loads(r['image_attention_json'])
  if len(records)!=len(s['visits']):raise ValueError('Missing attention records')
  for v,rec in zip(s['visits'],records):
   if str(v['visit_id'])!=str(rec['visit_id']) or [Path(p).name for p in v['image_paths']]!=rec['image_names']:raise ValueError('Image order changed')
 device=DEVICE if torch.cuda.is_available() else 'cpu';torch.manual_seed(SEED)
 run=OUTPUT_ROOT/(( 'SMOKE_' if SMOKE_TEST else 'FORMAL_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));run.mkdir(parents=True)
 write_csv(run/'cohort.csv',[{k:r[k] for k in ['sample_id','case_id','side','fold_id','label','num_visits']} for r in selected])
 manifest=dict(status='running',smoke_test=SMOKE_TEST,result_dir=str(RESULT_DIR),data_root=str(DATA_ROOT),min_visits=MIN_VISITS,prefixes=PREFIXES,seed=SEED,n_bootstrap=100 if SMOKE_TEST else N_BOOTSTRAP,n_permutations=100 if SMOKE_TEST else N_PERMUTATIONS,n_patients=len({r['case_id'] for r in selected}),n_renal_units=len(selected),prediction_sha256=sha(source),checkpoint_sha256={f:sha(p) for f,p in checkpoints.items()},fetalclip_weight_sha256=sha(FETALCLIP_WEIGHTS),fetalclip_config_sha256=sha(FETALCLIP_CONFIG),script_sha256=sha(Path(__file__)),holm_family='All 7 comparisons x 9 metrics (63 tests)',inference_scope='Fixed-model OOF prefix sensitivity analysis; conditional on fitted models, not time-to-event prediction')
 (run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 extractor=FetalCLIPFeatureExtractor(model_name='FetalCLIP',config_path=FETALCLIP_CONFIG,weight_path=FETALCLIP_WEIGHTS,device=device)
 rows=[];loaded=None;model=None;replay=[]
 with (run/'prefix_predictions.csv').open('w',encoding='utf-8-sig',newline='') as out:
  writer=None
  for count,r in enumerate(sorted(selected,key=lambda x:(x['fold_id'],x['sample_id'])),1):
   fold=r['fold_id'];s=samples[r['sample_id']];print(f'[{count}/{len(selected)}] {r["sample_id"]}',flush=True)
   if fold!=loaded:
    if model is not None:del model
    model=ModularGRUSideClassifier(**meta[fold]['model_kwargs'])
    obj=torch.load(checkpoints[fold],map_location='cpu',weights_only=True);model.load_state_dict(obj['model_state_dict'],strict=True);del obj
    model=model.to(device).eval();loaded=fold
   with torch.inference_mode():
    for v in s['visits']:
     v['text_feature']=extractor.encode_text(v.get('report_text',''));v['image_features']=extractor.encode_images(v['image_paths'])
    logits,attention=model([s],return_attention=True);full_prob=float(torch.sigmoid(logits).item())
    if not np.isfinite(full_prob) or not np.isfinite(float(r['prob'])) or abs(full_prob-float(r['prob']))>REPLAY_ATOL:raise ValueError(f'{r["sample_id"]}: full prediction does not reproduce; check preprocessing and checkpoint')
    for actual,old in zip(attention[0],json.loads(r['image_attention_json'])):
     a=actual['image_attention'];b=old['image_attention']
     if len(a)!=len(b) or not np.allclose(a,b,atol=REPLAY_ATOL,rtol=0):raise ValueError('Full attention does not reproduce')
    replay.append(dict(sample_id=r['sample_id'],fold_id=fold,saved_prob=r['prob'],replayed_prob=full_prob,absolute_error=abs(full_prob-float(r['prob']))))
    for n in PREFIXES+[None]:
     truncated=prefix_sample(s,n);stage='full' if n is None else f'first_{n}'
     if n is None:prob=full_prob
     else:
      logit,_=model([truncated],return_attention=False);prob=float(torch.sigmoid(logit).item())
     if not np.isfinite(prob):raise ValueError('Nonfinite prediction')
     row=dict(stage=stage,sample_id=r['sample_id'],case_id=r['case_id'],side=r['side'],fold_id=fold,label=int(r['label']),total_visits=len(s['visits']),used_visits=len(truncated['visits']),visit_ids_used='|'.join(str(v['visit_id']) for v in truncated['visits']),prob=prob,pred=int(prob>=.5))
     if writer is None:writer=csv.DictWriter(out,fieldnames=list(row));writer.writeheader()
     writer.writerow(row);rows.append(row)
   out.flush();write_csv(run/'full_replay.csv',replay)
   for v in s['visits']:v.pop('image_features',None);v.pop('text_feature',None)
 analyze(rows,run,manifest['n_bootstrap'],manifest['n_permutations'])
 manifest['status']='complete';(run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Complete:',run)

if __name__=='__main__':main()
