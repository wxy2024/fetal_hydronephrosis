"""Frozen OOF image perturbation evaluation. Edit configuration; no training."""
from pathlib import Path
import sys, json, csv, random, hashlib, re, tempfile
from datetime import datetime
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
DATA_ROOT = Path(r'E:\wxy_codes\dataset_hydro_1234\raw')
MODEL_DIRS = {
 'rcc_pooling': Path(r'E:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main7\output5_dataset_hydro_1234\results_69_dataset_123_fig_rcc_pooling_visit_direct_concat_td_conv_gru_modality_multimodal_temporal_baseline'),
 'attention': Path(r'E:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main8\output6_rebuttal\rebuttal_1002_dataset_1234_fig_attention_visit_direct_concat_td_conv_gru_modality_multimodal_temporal_baseline'),
}
FETALCLIP_WEIGHTS = Path(r'E:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main8\FetalCLIP_weights.pt')
FETALCLIP_CONFIG = ROOT / 'FetalCLIP_config.json'
# If automatic discovery finds multiple files, explicitly map each outer fold.
CHECKPOINTS = {'rcc_pooling': {}, 'attention': {}}  # e.g. {'fold_1': Path(r'E:\...\fold_1.pt'), ...}
OUTPUT_ROOT = ROOT / 'output6_rebuttal' / 'rcc_perturbation'
SEED = 42
MAX_PATIENTS = 0  # 0 = all eligible; set 2 for a smoke run
NOISE_REPEATS = 5
BOOTSTRAP_REPEATS = 10000
PERTURBATIONS = {'blur': [1., 2., 4.], 'noise': [.02, .05, .10]}
DEVICE = 'cuda'
# Explicit training architecture, independent of the currently active training config.
MODEL_KWARGS = dict(visit_fusion_type='direct_concat', attn_hidden_dim=256,
 fusion_hidden_dim=None, gru_hidden_dim=128, gru_num_layers=1, dropout=.3,
 bidirectional=True, temporal_backbone='td_conv_gru', input_modality='multimodal', use_explicit_time=True)
REPLAY_ATOL = 1e-4

def read_csv(path):
 with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows):
 if not rows: return
 with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def file_hash(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def discover_checkpoint(directory, fold):
 # Never guess between best/last/inner-fold checkpoints.
 matches=[p for p in Path(directory).rglob('*') if p.suffix.lower() in {'.pt','.pth','.ckpt'}
          and re.search(r'(?<![A-Za-z0-9])'+re.escape(fold)+r'(?![0-9])',str(p.relative_to(directory)))]
 if not matches: raise FileNotFoundError(f'{directory}: no checkpoint identified for {fold}. Set CHECKPOINTS explicitly. CSV predictions are not model weights.')
 if len(matches)!=1:raise ValueError(f'Ambiguous checkpoint for {fold}: {matches}. Set CHECKPOINTS explicitly to final outer-fold model.')
 return matches[0]

def perturb(image, kind, strength, seed):
 image=image.convert('RGB')
 if kind=='blur':
  # Original geometry retained; radius uses shorter edge/224 as reference scale.
  return image.filter(ImageFilter.GaussianBlur(strength*min(image.size)/224.))
 if kind=='noise':
  x=np.asarray(image,dtype=np.float32)/255.
  noise=np.random.default_rng(seed).standard_normal(x.shape).astype(np.float32)
  return Image.fromarray(np.rint(np.clip(x+strength*noise,0,1)*255).astype(np.uint8))
 raise ValueError(kind)

def rank_desc(weights,index):
 w=np.asarray(weights);v=w[index]
 return float(1+np.sum(w>v)+(np.sum(w==v)-1)/2)

def load_predictions(directory):
 rows=read_csv(directory/'5fold_predictions.csv');out={};folds={}
 for r in rows:
  if r['sample_id'] in out:raise ValueError('Duplicate OOF sample')
  if r['case_id'] in folds and folds[r['case_id']]!=r['fold_id']:raise ValueError('Patient split across folds')
  if r['input_modality']!='multimodal' or r['temporal_experiment']!='baseline':raise ValueError('Requires multimodal baseline OOF predictions')
  folds[r['case_id']]=r['fold_id'];out[r['sample_id']]=r
 if not out:raise ValueError('Empty predictions')
 return out

def state_dict(path):
 import torch
 # Load only tensor-safe checkpoints; do not fall back to unsafe pickle execution.
 obj=torch.load(path,map_location='cpu',weights_only=True)
 for key in ('model_state_dict','state_dict','model'):
  if isinstance(obj,dict) and key in obj and isinstance(obj[key],dict):obj=obj[key];break
 if not isinstance(obj,dict) or not obj or not all(torch.is_tensor(v) for v in obj.values()):
  raise ValueError(f'Unsupported checkpoint format: {path}')
 if all(k.startswith('module.') for k in obj):obj={k[7:]:v for k,v in obj.items()}
 return obj

def predict(model,sample,visit_index):
 import torch
 with torch.inference_mode():
  logits,attn=model([sample],return_attention=True)
 w=np.asarray(attn[0][visit_index]['image_attention'],dtype=float)
 if not np.all(np.isfinite(w)) or abs(w.sum()-1)>1e-4:raise ValueError('Invalid attention weights')
 return float(torch.sigmoid(logits).item()),w

def interval(values,rng):
 a=np.asarray(values,dtype=float)
 if len(a)<2:return (float(a.mean()),None,None)
 boot=np.array([rng.choice(a,len(a),replace=True).mean() for _ in range(BOOTSTRAP_REPEATS)])
 lo,hi=np.quantile(boot,[.025,.975]);return float(a.mean()),float(lo),float(hi)

def summarize(path):
 import pandas as pd
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 df=pd.read_csv(path/'perturbations.csv')
 metrics=['weight_change','rank_change','abs_probability_change','weight_decreased']
 # Repeats -> images -> patients (one selected visit per patient).
 images=df.groupby(['model','case_id','image_index','kind','level'],as_index=False)[metrics].mean()
 trends=[]
 for (model,case,index,kind),g in images.groupby(['model','case_id','image_index','kind']):
  g=g.sort_values('level')
  changes=np.array([0.]+g['weight_change'].tolist())
  trends.append(dict(model=model,case_id=case,image_index=index,kind=kind,
   nonincreasing=int(np.all(np.diff(changes)<=1e-8)),strictly_decreasing=int(np.all(np.diff(changes)<-1e-8))))
 pd.DataFrame(trends).groupby(['model','case_id','kind'],as_index=False)[['nonincreasing','strictly_decreasing']].mean().to_csv(path/'patient_trends.csv',index=False,encoding='utf-8-sig')
 patients=images.groupby(['model','case_id','kind','level'],as_index=False)[metrics].mean()
 patients.to_csv(path/'patient_effects.csv',index=False,encoding='utf-8-sig')
 rng=np.random.default_rng(SEED);summary=[];paired=[]
 for keys,g in patients.groupby(['model','kind','level']):
  for metric in metrics:
   mean,lo,hi=interval(g[metric],rng)
   summary.append(dict(model=keys[0],kind=keys[1],level=int(keys[2]),metric=metric,n_patients=len(g),mean=mean,ci_low=lo,ci_high=hi))
 for (kind,level),g in patients.groupby(['kind','level']):
  for metric in metrics:
   pivot=g.pivot(index='case_id',columns='model',values=metric)
   if set(pivot.columns)=={'rcc_pooling','attention'}:
    if pivot.isna().any().any():raise ValueError('Unpaired patients in summary')
    mean,lo,hi=interval(pivot['rcc_pooling']-pivot['attention'],rng)
    paired.append(dict(kind=kind,level=int(level),metric=metric,n_patients=len(pivot),difference_rcc_minus_attention=mean,ci_low=lo,ci_high=hi))
 write_csv(path/'summary.csv',summary);write_csv(path/'paired_differences.csv',paired)
 fig,axes=plt.subplots(1,len(PERTURBATIONS),figsize=(10,4),squeeze=False)
 for ax,kind in zip(axes[0],PERTURBATIONS):
  for model in MODEL_DIRS:
   rows=sorted([r for r in summary if r['kind']==kind and r['model']==model and r['metric']=='weight_change'],key=lambda r:r['level'])
   x=[0]+[r['level'] for r in rows];y=[0]+[r['mean'] for r in rows]
   ax.plot(x,y,'o-',label=model)
   if all(r['ci_low'] is not None for r in rows):ax.fill_between(x,[0]+[r['ci_low'] for r in rows],[0]+[r['ci_high'] for r in rows],alpha=.15)
  ax.axhline(0,color='gray',linewidth=.7);ax.set(title=kind,xlabel='Severity level',ylabel='Target weight change');ax.legend()
 fig.tight_layout();fig.savefig(path/'weight_change.png',dpi=200);plt.close(fig)

def main():
 import torch
 from data.dataset_online import collect_case_side_samples_online
 from models.longitudinal_gru_classifier import ModularGRUSideClassifier
 predictions={name:load_predictions(directory) for name,directory in MODEL_DIRS.items()}
 reference=predictions['rcc_pooling']
 for name,pred in predictions.items():
  if set(pred)!=set(reference):raise ValueError(f'{name}: OOF cohorts differ; align before comparison')
  for key,row in reference.items():
   for field in ('case_id','side','label','fold_id','visit_ids_used'):
    if pred[key][field]!=row[field]:raise ValueError(f'{name}/{key}: mismatch in {field}; no automatic intersection')
 checkpoints={name:{fold:Path(CHECKPOINTS[name].get(fold) or discover_checkpoint(directory,fold))
                    for fold in sorted({r['fold_id'] for r in reference.values()})} for name,directory in MODEL_DIRS.items()}
 for folds in checkpoints.values():
  for p in folds.values():
   if not p.is_file():raise FileNotFoundError(p)
 for p in (DATA_ROOT,FETALCLIP_CONFIG,FETALCLIP_WEIGHTS):
  if not p.exists():raise FileNotFoundError(p)
 from fetalclip.fetalclip_wrapper import FetalCLIPFeatureExtractor
 device=DEVICE if torch.cuda.is_available() else 'cpu'
 torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
 samples={s['sample_id']:s for s in collect_case_side_samples_online(DATA_ROOT)}
 groups={}
 for key,row in reference.items():
  if key not in samples:raise ValueError('Missing raw sample: '+key)
  s=samples[key]
  if str(s['label'])!=row['label']:raise ValueError('Label mismatch: '+key)
  if '|'.join(str(v['visit_id']) for v in s['visits'])!=row['visit_ids_used']:raise ValueError('Visit history mismatch: '+key)
  for name,pred in predictions.items():
   records=json.loads(pred[key]['image_attention_json'])
   if len(records)!=len(s['visits']):raise ValueError('Missing saved image attention')
   for v,rec in zip(s['visits'],records):
    if str(v['visit_id'])!=str(rec['visit_id']) or [Path(p).name for p in v['image_paths']]!=rec['image_names']:
     raise ValueError(f'{name}/{key}: image order or identities changed')
  for j,v in enumerate(s['visits']):
   if len(v['image_paths'])>=2:groups.setdefault(s['case_id'],[]).append((key,j))
 rng=random.Random(SEED);cases=sorted(groups);rng.shuffle(cases)
 if MAX_PATIENTS:cases=cases[:MAX_PATIENTS]
 selection=[(case,*rng.choice(groups[case])) for case in cases]
 if not selection:raise ValueError('No eligible patients')
 run=OUTPUT_ROOT/datetime.now().strftime('%Y%m%d_%H%M%S_%f');run.mkdir(parents=True)
 write_csv(run/'selection.csv',[dict(case_id=c,sample_id=k,visit_index=j,visit_id=samples[k]['visits'][j]['visit_id'],fold_id=reference[k]['fold_id']) for c,k,j in selection])
 manifest=dict(seed=SEED,max_patients=MAX_PATIENTS,selected_patients=len(selection),perturbations=PERTURBATIONS,noise_repeats=NOISE_REPEATS,model_kwargs=MODEL_KWARGS,data_root=str(DATA_ROOT),fetalclip_weights=str(FETALCLIP_WEIGHTS),fetalclip_sha256=file_hash(FETALCLIP_WEIGHTS),prediction_sha256={n:file_hash(d/'5fold_predictions.csv') for n,d in MODEL_DIRS.items()},checkpoints={n:{f:dict(path=str(p),sha256=file_hash(p)) for f,p in folds.items()} for n,folds in checkpoints.items()},status='running')
 (run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 extractor=FetalCLIPFeatureExtractor(model_name='FetalCLIP',config_path=FETALCLIP_CONFIG,weight_path=FETALCLIP_WEIGHTS,device=device)
 models={};loaded_fold=None;baseline_rows=[]
 with tempfile.TemporaryDirectory(prefix='rcc_perturb_') as temp, (run/'perturbations.csv').open('w',encoding='utf-8-sig',newline='') as output:
  writer=None
  for count,(case,key,vi) in enumerate(sorted(selection,key=lambda x:(reference[x[1]]['fold_id'],x[0])),1):
   print(f'[{count}/{len(selection)}] {key}',flush=True)
   s=samples[key];fold=reference[key]['fold_id']
   for visit in s['visits']:
    visit['text_feature']=extractor.encode_text(visit.get('report_text',''))
    visit['image_features']=extractor.encode_images(visit['image_paths'])
   dim=int(s['visits'][0]['text_feature'].shape[0])
   if fold!=loaded_fold:
    models.clear()
    for name in MODEL_DIRS:
     model=ModularGRUSideClassifier(feature_dim=dim,image_pooling_type=name,**MODEL_KWARGS)
     model.load_state_dict(state_dict(checkpoints[name][fold]),strict=True)
     models[name]=model.to(device).eval()
    loaded_fold=fold
   visit=s['visits'][vi];original=visit['image_features'];base={}
   for name,model in models.items():
    prob,w=predict(model,s,vi);saved=predictions[name][key]
    saved_w=json.loads(saved['image_attention_json'])[vi]['image_attention']
    if abs(prob-float(saved['prob']))>REPLAY_ATOL or not np.allclose(w,saved_w,atol=REPLAY_ATOL,rtol=0):
     raise ValueError(f'{name}/{key}: original prediction/attention does not reproduce; check weights, preprocessing and architecture. No perturbation inference accepted.')
    if len(w)!=len(visit['image_paths']):raise ValueError('Image/weight count mismatch')
    base[name]=(prob,w)
    baseline_rows.append(dict(model=name,case_id=case,sample_id=key,fold_id=fold,probability=prob,saved_probability=float(saved['prob'])))
   write_csv(run/'baseline_replay.csv',baseline_rows)
   for index,image_path in enumerate(visit['image_paths']):
    with Image.open(image_path) as im:clean=im.convert('RGB')
    for kind,strengths in PERTURBATIONS.items():
     for level,strength in enumerate(strengths,1):
      for repeat in range(NOISE_REPEATS if kind=='noise' else 1):
       seed=int.from_bytes(hashlib.sha256(f'{SEED}|{key}|{vi}|{index}|{kind}|{repeat}'.encode()).digest()[:4],'little')
       # Unique file paths prevent a wrapper's path cache from reusing features.
       tmp=Path(temp)/f'{count}_{index}_{kind}_{level}_{repeat}.png'
       degraded=perturb(clean,kind,strength,seed);degraded.save(tmp)
       if count==1 and index==0 and repeat==0:
        (run/'examples').mkdir(exist_ok=True);clean.save(run/'examples/original.png');degraded.save(run/f'examples/{kind}_{level}.png')
       feature=extractor.encode_images([str(tmp)])
       try:
        visit['image_features']=original.clone();visit['image_features'][index]=feature[0].to(original.device)
        for name,model in models.items():
         prob,w=predict(model,s,vi);bp,bw=base[name]
         row=dict(model=name,case_id=case,sample_id=key,fold_id=fold,visit_id=visit['visit_id'],image_index=index,image_name=Path(image_path).name,n_images=len(bw),kind=kind,level=level,strength=strength,repeat=repeat,seed=seed,baseline_weight=float(bw[index]),weight=float(w[index]),weight_change=float(w[index]-bw[index]),rank_change=rank_desc(w,index)-rank_desc(bw,index),baseline_probability=bp,probability=prob,abs_probability_change=abs(prob-bp),weight_decreased=int(w[index]<bw[index]-1e-8))
         if writer is None:writer=csv.DictWriter(output,fieldnames=list(row));writer.writeheader()
         writer.writerow(row)
       finally:visit['image_features']=original
       tmp.unlink()
   output.flush()
   for v in s['visits']:v.pop('image_features',None);v.pop('text_feature',None)
 write_csv(run/'baseline_replay.csv',baseline_rows)
 summarize(run)
 manifest['status']='complete';(run/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('Completed:',run)

if __name__=='__main__':main()
