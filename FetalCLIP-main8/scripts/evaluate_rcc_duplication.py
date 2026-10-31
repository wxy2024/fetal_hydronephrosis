"""Fixed-cohort exact-image duplication experiment. Edit configuration below."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Server paths. No command-line questions; use the existing training environment.
SERVER_ROOT = Path(r'E:\wxy_codes\Fetal_hydronephrosis\FetalCLIP-main8')
DATA_ROOT = Path(r'E:\wxy_codes\dataset_hydro_1234\raw')
MODEL_DIRS = {
    'rcc_pooling': SERVER_ROOT / 'output6_rebuttal/rebuttal_69_dataset_1234_fig_rcc_pooling_visit_direct_concat_td_conv_gru_modality_multimodal_temporal_baseline',
    'attention': SERVER_ROOT / 'output6_rebuttal/rebuttal_1013_dataset_1234_fig_attention_visit_direct_concat_td_conv_gru_modality_multimodal_temporal_baseline',
}
SOURCE_RUN = SERVER_ROOT / 'output6_rebuttal/rcc_perturbation/20261031_183355_665139'
FETALCLIP_WEIGHTS = SERVER_ROOT / 'FetalCLIP_weights.pt'
FETALCLIP_CONFIG = SERVER_ROOT / 'FetalCLIP_config.json'
OUTPUT_ROOT = SERVER_ROOT / 'output6_rebuttal/rcc_duplication'
SMOKE_TEST = False  # First run: two of the fixed 185 patients. False: all 185.
EXPECTED_PATIENTS = 185
ADDED_COPIES = [1, 3, 5]
N_BOOTSTRAP = 10000
SEED = 42
DEVICE = 'cuda'
REPLAY_ATOL = 1e-4
METRICS = ['group_weight_change', 'relative_feature_change',
           'cosine_distance', 'abs_probability_change']


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_predictions(directory):
    result, folds = {}, {}
    for r in read_csv(directory / '5fold_predictions.csv'):
        key = r['sample_id']
        if key in result:
            raise ValueError(f'Duplicate prediction: {key}')
        if r['case_id'] in folds and folds[r['case_id']] != r['fold_id']:
            raise ValueError('Same patient in different test folds')
        if r['input_modality'] != 'multimodal' or r['temporal_experiment'] != 'baseline':
            raise ValueError('Requires multimodal baseline OOF predictions')
        if not np.isfinite(float(r['prob'])) or not 0 <= float(r['prob']) <= 1:
            raise ValueError('Invalid saved probability')
        folds[r['case_id']] = r['fold_id']
        result[key] = r
    if not result:
        raise ValueError('Empty predictions')
    return result


def validate_selection(selection, predictions):
    """Check the exact existing patient/renal-unit/visit selection, no resampling."""
    seen = set()
    for row in selection:
        case, key, vi = row['case_id'], row['sample_id'], int(row['visit_index'])
        if case in seen:
            raise ValueError(f'Multiple selected visits for patient {case}')
        seen.add(case)
        r = predictions[key]
        if r['case_id'] != case or r['fold_id'] != row['fold_id']:
            raise ValueError(f'Selection identity/fold mismatch: {key}')
        records = json.loads(r['image_attention_json'])
        ids = r['visit_ids_used'].split('|')
        if len(records) != int(r['num_visits']) or len(ids) != len(records):
            raise ValueError(f'Incomplete saved visit records: {key}')
        if not 0 <= vi < len(records) or str(records[vi]['visit_id']) != row['visit_id'] or ids[vi] != row['visit_id']:
            raise ValueError(f'Selected visit mismatch: {key}')
        if len(records[vi]['image_names']) < 2:
            raise ValueError(f'Selected visit needs at least two images: {key}')


def effective_weights(pool, alpha):
    """RCC beta=(1-lambda)/N+lambda*alpha; standard attention beta=alpha."""
    if alpha.ndim != 1 or len(alpha) == 0 or not torch.isfinite(alpha).all():
        raise ValueError('Invalid attention vector')
    if (alpha < 0).any() or not torch.isclose(alpha.sum(), alpha.new_tensor(1.), atol=1e-5):
        raise ValueError('Attention weights do not sum to one')
    if hasattr(pool, 'residual_logit'):
        ratio = torch.sigmoid(pool.residual_logit).detach()
        return (1 - ratio) / len(alpha) + ratio * alpha, float(ratio)
    return alpha, None


def add_copies(sample, visit_index, image_index, copies):
    """Append exact frozen feature copies; never mutate the original sample."""
    v = sample['visits'][visit_index]
    x = v['image_features']
    if copies < 1 or not 0 <= image_index < len(x):
        raise ValueError('Invalid duplication request')
    if len(x) != len(v['image_paths']):
        raise ValueError('Image feature/path count mismatch')
    changed = dict(v)
    changed['image_features'] = torch.cat([x, x[image_index:image_index + 1].expand(copies, -1)], dim=0)
    changed['image_paths'] = list(v['image_paths']) + [v['image_paths'][image_index]] * copies
    changed['num_images'] = len(changed['image_paths'])
    out = dict(sample)
    out['visits'] = list(sample['visits'])
    out['visits'][visit_index] = changed
    return out


def measure_effect(before, after, beta_before, beta_after, image_index, copies, prob_before, prob_after):
    def array(x):
        return x.detach().cpu().double().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=float)
    x, z, a, b = map(array, (before, after, beta_before, beta_after))
    if len(b) != len(a) + copies or not all(np.isfinite(t).all() for t in (x, z, a, b)):
        raise ValueError('Invalid representation or weight sizes')
    norm_x, norm_z = np.linalg.norm(x), np.linalg.norm(z)
    if min(norm_x, norm_z) <= 1e-12:
        raise ValueError('Near-zero pooled feature norm; relative/cosine change is undefined')
    # Count the original target AND its appended copies, not one copy alone.
    mass = float(b[image_index] + b[len(a):].sum())
    return {
        'baseline_target_weight': float(a[image_index]),
        'duplicate_group_weight': mass,
        'group_weight_change': mass - float(a[image_index]),
        'relative_feature_change': float(np.linalg.norm(z - x) / norm_x),
        'cosine_distance': float(1 - np.clip(np.dot(x, z) / (norm_x * norm_z), -1, 1)),
        'baseline_feature_norm': float(norm_x),
        'baseline_probability': prob_before,
        'probability': prob_after,
        'abs_probability_change': abs(prob_after - prob_before),
    }


@torch.inference_mode()
def evaluate_sample(model, sample, visit_index):
    """Capture exactly the pooled feature used by the downstream classifier."""
    call = 0
    target = {}

    def capture(module, inputs, output):
        nonlocal call
        if call == visit_index:
            target['pooled'] = output[0].detach().clone()
            target['alpha'] = output[2].detach().clone()
        call += 1

    handle = model.image_pool.register_forward_hook(capture)
    try:
        logits, all_attention = model([sample], return_attention=True)
    finally:
        handle.remove()
    if call != len(sample['visits']) or not target:
        raise ValueError('Unexpected image-pooling calls')
    probability = float(torch.sigmoid(logits).item())
    if not np.isfinite(probability):
        raise ValueError('Nonfinite probability')
    beta, ratio = effective_weights(model.image_pool, target['alpha'])
    features = sample['visits'][visit_index]['image_features'].to(beta.device).float()
    reconstructed = (features * beta[:, None]).sum(0)
    if not torch.allclose(reconstructed, target['pooled'], atol=1e-5, rtol=1e-4):
        raise ValueError('Effective coefficients fail to reconstruct pooled feature')
    return dict(probability=probability, pooled=target['pooled'], alpha=target['alpha'],
                beta=beta, mixture_lambda=ratio, all_attention=all_attention[0])


def aggregate_patients(rows):
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r['model'], r['case_id'], int(r['added_copies']))
        identity = key + (int(r['image_index']),)
        if identity in seen:
            raise ValueError('Duplicate per-image result')
        seen.add(identity)
        groups[key].append(r)
    result = []
    for (model, case, copies), rs in sorted(groups.items()):
        if len({r['sample_id'] for r in rs}) != 1 or len({r['fold_id'] for r in rs}) != 1:
            raise ValueError('More than one renal-unit visit per patient')
        if {int(r['image_index']) for r in rs} != set(range(len(rs))):
            raise ValueError('Incomplete target-image experiments')
        values = np.array([[float(r[m]) for m in METRICS] for r in rs])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite metrics')
        result.append(dict(model=model, case_id=case, fold_id=rs[0]['fold_id'],
                           added_copies=copies, n_target_images=len(rs),
                           **dict(zip(METRICS, values.mean(axis=0)))))
    return result


def bootstrap_summary(patients, repeats, seed):
    """Equal-weight patients; preserve fold sizes and pair all model/count cells."""
    cases = sorted({r['case_id'] for r in patients})
    settings = sorted({(r['model'], int(r['added_copies'])) for r in patients})
    lookup = {(r['model'], int(r['added_copies']), r['case_id']): r for r in patients}
    if len(lookup) != len(patients) or len(patients) != len(cases) * len(settings):
        raise ValueError('Missing/duplicate model-patient-count results')
    cube = np.array([[[lookup[(model, copies, c)][m] for m in METRICS] for model, copies in settings] for c in cases], dtype=float)
    by_fold = defaultdict(list)
    for i, c in enumerate(cases):
        rs = [lookup[(model, copies, c)] for model, copies in settings]
        if len({r['fold_id'] for r in rs}) != 1 or len({r['n_target_images'] for r in rs}) != 1:
            raise ValueError('Unpaired folds or target-image counts')
        by_fold[rs[0]['fold_id']].append(i)
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, len(settings), len(METRICS)))
    for b in range(repeats):
        indices = np.concatenate([rng.choice(v, len(v), replace=True) for _, v in sorted(by_fold.items())])
        boot[b] = cube[indices].mean(axis=0)
    point = cube.mean(axis=0)

    def ci(values):
        if len(cases) < 2:
            return (None, None)
        return tuple(float(v) for v in np.quantile(values, [.025, .975]))

    summary, paired = [], []
    for j, (model, copies) in enumerate(settings):
        for m, metric in enumerate(METRICS):
            lo, hi = ci(boot[:, j, m])
            summary.append(dict(model=model, added_copies=copies, metric=metric,
                                n_patients=len(cases), mean=float(point[j, m]), ci_low=lo, ci_high=hi))
    for copies in sorted({s[1] for s in settings}):
        a = settings.index(('attention', copies))
        r = settings.index(('rcc_pooling', copies))
        for m, metric in enumerate(METRICS):
            lo, hi = ci(boot[:, r, m] - boot[:, a, m])
            paired.append(dict(added_copies=copies, metric=metric, n_patients=len(cases),
                               difference_rcc_minus_attention=float(point[r, m] - point[a, m]), ci_low=lo, ci_high=hi))
    return summary, paired


def save_plot(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ['group_weight_change', 'relative_feature_change', 'abs_probability_change']):
        for model in ['rcc_pooling', 'attention']:
            rs = sorted([r for r in summary if r['metric'] == metric and r['model'] == model], key=lambda r: r['added_copies'])
            x = [0] + [r['added_copies'] for r in rs]
            ax.plot(x, [0] + [r['mean'] for r in rs], 'o-', label=model)
            if all(r['ci_low'] is not None for r in rs):
                ax.fill_between(x, [0] + [r['ci_low'] for r in rs], [0] + [r['ci_high'] for r in rs], alpha=.15)
        ax.set(title=metric.replace('_', ' '), xlabel='Additional exact copies')
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def verify_raw_sample(sample, row):
    if any(str(sample[k]) != row[k] for k in ['sample_id', 'case_id', 'side', 'label']):
        raise ValueError('Raw sample identity/label changed')
    visits = sample['visits']
    ids = '|'.join(str(v['visit_id']) for v in visits)
    if ids != row['visit_ids_used'] or ids != row['visit_ids_chronological'] or len(visits) != int(row['num_visits']):
        raise ValueError('Raw chronological visit history changed')
    saved = json.loads(row['image_attention_json'])
    if len(saved) != len(visits):
        raise ValueError('Missing saved attention records')
    for v, record in zip(visits, saved):
        names = [Path(p).name for p in v['image_paths']]
        if str(v['visit_id']) != str(record['visit_id']) or names != record['image_names']:
            raise ValueError('Raw image order differs from saved predictions')
        for image_path in v['image_paths']:
            if not Path(image_path).is_file():
                raise FileNotFoundError(image_path)


def check_replay(result, saved):
    error = abs(result['probability'] - float(saved['prob']))
    if error > REPLAY_ATOL:
        raise ValueError(f"Original prediction cannot be reproduced: {saved['sample_id']}, error={error}")
    previous = json.loads(saved['image_attention_json'])
    current = result['all_attention']
    if len(previous) != len(current):
        raise ValueError('Attention replay visit count differs')
    for a, b in zip(previous, current):
        if str(a['visit_id']) != str(b['visit_id']) or a['image_names'] != b['image_names']:
            raise ValueError('Attention replay image IDs differ')
        if len(a['image_attention']) != len(b['image_attention']) or not np.allclose(a['image_attention'], b['image_attention'], atol=REPLAY_ATOL, rtol=0):
            raise ValueError('Original attention cannot be reproduced')
    return error


def main():
    from models.longitudinal_gru_classifier import ModularGRUSideClassifier

    if ADDED_COPIES != [1, 3, 5]:
        raise ValueError('This protocol prespecifies 1, 3, 5 additional copies')
    selection_file = SOURCE_RUN / 'selection.csv'
    source_manifest = json.loads((SOURCE_RUN / 'manifest.json').read_text(encoding='utf-8'))
    selection = read_csv(selection_file)
    if source_manifest.get('status') != 'complete' or len(selection) != EXPECTED_PATIENTS:
        raise ValueError('Expected the completed 185-patient perturbation selection')
    predictions = {name: load_predictions(path) for name, path in MODEL_DIRS.items()}
    reference = predictions['rcc_pooling']
    for name, pred in predictions.items():
        if set(pred) != set(reference):
            raise ValueError('OOF cohorts differ')
        for key, row in reference.items():
            for field in ['case_id', 'side', 'label', 'fold_id', 'visit_ids_used', 'visit_ids_chronological', 'num_visits']:
                if pred[key][field] != row[field]:
                    raise ValueError(f'{name}/{key}: mismatch in {field}')
        if sha256(MODEL_DIRS[name] / '5fold_predictions.csv') != source_manifest['prediction_sha256'][name]:
            raise ValueError('Prediction file differs from the original perturbation experiment')
        validate_selection(selection, pred)

    checkpoints, metadata = {}, {}
    folds = sorted({r['fold_id'] for r in selection})
    for name, directory in MODEL_DIRS.items():
        checkpoints[name], metadata[name] = {}, {}
        for fold in folds:
            path = directory / fold / 'final_model.pt'
            if not path.is_file():
                raise FileNotFoundError(f'Missing final trained weight: {path}')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            md = payload['metadata']
            kw = md['model_kwargs']
            if md['checkpoint_role'] != 'final_outer_fold_model' or md['fold_id'] != fold:
                raise ValueError('Checkpoint is not the requested final outer-fold model')
            if kw['image_pooling_type'] != name or kw['input_modality'] != 'multimodal' or kw['temporal_backbone'] != 'td_conv_gru' or md['temporal_experiment'] != 'baseline':
                raise ValueError('Unexpected checkpoint architecture/experiment')
            test = {r['case_id'] for r in reference.values() if r['fold_id'] == fold}
            if set(md['test_case_ids']) != test or set(md['train_case_ids']) != {r['case_id'] for r in reference.values()} - test:
                raise ValueError('Checkpoint train/test patients differ from saved OOF split')
            if set(md['test_sample_ids']) != {k for k, r in reference.items() if r['fold_id'] == fold}:
                raise ValueError('Checkpoint test sample IDs differ')
            if sha256(path) != source_manifest['checkpoints'][name][fold]['sha256']:
                raise ValueError('Checkpoint differs from previous perturbation experiment')
            checkpoints[name][fold], metadata[name][fold] = path, md
            del payload
    for fold in folds:
        left = {k: v for k, v in metadata['rcc_pooling'][fold]['model_kwargs'].items() if k != 'image_pooling_type'}
        right = {k: v for k, v in metadata['attention'][fold]['model_kwargs'].items() if k != 'image_pooling_type'}
        if left != right:
            raise ValueError('Models differ beyond pooling type')
    if sha256(FETALCLIP_WEIGHTS) != source_manifest['fetalclip_sha256']:
        raise ValueError('FetalCLIP checkpoint differs from previous experiment')

    if SMOKE_TEST:
        indices = np.random.default_rng(SEED).choice(len(selection), 2, replace=False)
        selection = [selection[int(i)] for i in indices]
    print(f'Selected {len(selection)} fixed patient/visit pairs. Loading raw data...', flush=True)
    from data.dataset_online import collect_case_side_samples_online
    from fetalclip.fetalclip_wrapper import FetalCLIPFeatureExtractor
    all_samples = {s['sample_id']: s for s in collect_case_side_samples_online(DATA_ROOT)}
    for row in selection:
        sample = all_samples[row['sample_id']]
        for pred in predictions.values():
            verify_raw_sample(sample, pred[row['sample_id']])
        if len(sample['visits'][int(row['visit_index'])]['image_paths']) < 2:
            raise ValueError('Selected raw visit no longer has at least two images')
    total_images = sum(len(all_samples[r['sample_id']]['visits'][int(r['visit_index'])]['image_paths']) for r in selection)
    repeats = 100 if SMOKE_TEST else N_BOOTSTRAP
    run = OUTPUT_ROOT / (('SMOKE_' if SMOKE_TEST else 'FORMAL_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run.mkdir(parents=True, exist_ok=False)
    write_csv(run / 'selection.csv', selection)
    manifest = dict(status='running', smoke_test=SMOKE_TEST, n_patients=len(selection), n_target_images=total_images,
                    expected_rows=total_images * len(ADDED_COPIES) * 2, added_copies=ADDED_COPIES,
                    bootstrap_repeats=repeats, seed=SEED, source_run=str(SOURCE_RUN),
                    selection_sha256=sha256(selection_file), script_sha256=sha256(Path(__file__)),
                    data_root=str(DATA_ROOT), model_dirs={k: str(v) for k, v in MODEL_DIRS.items()},
                    model_metadata=metadata, checkpoints=source_manifest['checkpoints'],
                    prediction_sha256=source_manifest['prediction_sha256'],
                    fetalclip_sha256=source_manifest['fetalclip_sha256'], fetalclip_config_sha256=sha256(FETALCLIP_CONFIG),
                    primary_endpoint='RCC minus attention: patient-mean relative pooled-feature change after 5 added copies',
                    method='Exact frozen feature duplication; effective RCC beta=(1-lambda)/N+lambda*alpha. Equal patient weighting, paired percentile bootstrap within folds. Conditional on fitted models. Secondary intervals unadjusted; no P values.')

    def save_manifest():
        (run / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    save_manifest()
    device = DEVICE if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(SEED)
    rows, replay, effective_rows, lambda_rows = [], [], [], []
    models, loaded = {}, None
    try:
        extractor = FetalCLIPFeatureExtractor(model_name='FetalCLIP', config_path=FETALCLIP_CONFIG,
                                             weight_path=FETALCLIP_WEIGHTS, device=device)
        with (run / 'duplication_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = None
            for count, selected in enumerate(sorted(selection, key=lambda r: (r['fold_id'], r['case_id'])), 1):
                key, fold, vi = selected['sample_id'], selected['fold_id'], int(selected['visit_index'])
                sample = all_samples[key]
                print(f'[{count}/{len(selection)}] {key}', flush=True)
                if fold != loaded:
                    models.clear()
                    for name in MODEL_DIRS:
                        payload = torch.load(checkpoints[name][fold], map_location='cpu', weights_only=True)
                        model = ModularGRUSideClassifier(**metadata[name][fold]['model_kwargs'])
                        model.load_state_dict(payload['model_state_dict'], strict=True)
                        models[name] = model.to(device).eval()
                        del payload
                        ratio = float(torch.sigmoid(model.image_pool.residual_logit).detach()) if name == 'rcc_pooling' else None
                        lambda_rows.append(dict(model=name, fold_id=fold, mixture_lambda=ratio,
                                                mean_branch_fraction=1 - ratio if ratio is not None else 0.))
                    loaded = fold
                    write_csv(run / 'mixture_coefficients.csv', lambda_rows)
                with torch.inference_mode():
                    for visit in sample['visits']:
                        visit['text_feature'] = extractor.encode_text(visit.get('report_text', ''))
                        visit['image_features'] = extractor.encode_images(visit['image_paths'])
                target = sample['visits'][vi]
                n = len(target['image_paths'])
                baselines = {}
                for name, model in models.items():
                    base = evaluate_sample(model, sample, vi)
                    error = check_replay(base, predictions[name][key])
                    baselines[name] = base
                    replay.append(dict(model=name, case_id=selected['case_id'], sample_id=key, fold_id=fold,
                                       probability=base['probability'], saved_probability=predictions[name][key]['prob'], absolute_error=error))
                    for i, (a, b) in enumerate(zip(base['alpha'].cpu().tolist(), base['beta'].cpu().tolist())):
                        effective_rows.append(dict(model=name, case_id=selected['case_id'], sample_id=key,
                                                   fold_id=fold, visit_id=selected['visit_id'], image_index=i,
                                                   image_name=Path(target['image_paths'][i]).name,
                                                   mixture_lambda=base['mixture_lambda'], attention_alpha=a, effective_beta=b))
                write_csv(run / 'baseline_replay.csv', replay)
                write_csv(run / 'baseline_weights.csv', effective_rows)
                for i in range(n):
                    for copies in ADDED_COPIES:
                        duplicated = add_copies(sample, vi, i, copies)
                        for name, model in models.items():
                            base = baselines[name]
                            after = evaluate_sample(model, duplicated, vi)
                            effect = measure_effect(base['pooled'], after['pooled'], base['beta'], after['beta'],
                                                    i, copies, base['probability'], after['probability'])
                            alpha = after['alpha'].detach().cpu().numpy()
                            row = dict(model=name, case_id=selected['case_id'], sample_id=key, fold_id=fold,
                                       visit_id=selected['visit_id'], image_index=i, image_name=Path(target['image_paths'][i]).name,
                                       original_n_images=n, added_copies=copies, new_n_images=n + copies,
                                       label=int(sample['label']), mixture_lambda=base['mixture_lambda'],
                                       baseline_attention_alpha=float(base['alpha'][i]),
                                       duplicated_group_attention_alpha=float(alpha[i] + alpha[n:].sum()), **effect)
                            if writer is None:
                                writer = csv.DictWriter(f, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append(row)
                f.flush()
                for visit in sample['visits']:
                    visit.pop('image_features', None)
                    visit.pop('text_feature', None)
            if len(rows) != manifest['expected_rows']:
                raise ValueError('Incomplete duplication results')
        patients = aggregate_patients(rows)
        write_csv(run / 'patient_effects.csv', patients)
        summary, paired = bootstrap_summary(patients, repeats, SEED)
        write_csv(run / 'summary.csv', summary)
        write_csv(run / 'paired_differences.csv', paired)
        save_plot(summary, run / 'duplication_effects.png')
        manifest['status'] = 'complete'
        save_manifest()
        print('Completed:', run, flush=True)
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        save_manifest()
        raise


if __name__ == '__main__':
    main()
