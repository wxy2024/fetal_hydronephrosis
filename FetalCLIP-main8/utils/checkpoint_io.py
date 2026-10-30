"""Final outer-fold inference checkpoints (not optimizer/resume snapshots)."""
import json
import os
import tempfile
from pathlib import Path
import torch


def save_final_checkpoint(model, path, metadata):
    path = Path(path)
    # Normalize metadata to safe JSON primitives, also accepted by weights_only.
    metadata = json.loads(json.dumps(metadata, ensure_ascii=False))
    if set(metadata.get('train_case_ids', [])) & set(metadata.get('test_case_ids', [])):
        raise ValueError('Train and test patients overlap.')
    sidecar = path.with_suffix('.json')
    if path.exists() or sidecar.exists():
        raise FileExistsError(f'Checkpoint already exists: {path}; use a new experiment ID.')
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 1,
        'model_state_dict': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        'metadata': metadata,
    }
    fd, temp_name = tempfile.mkstemp(prefix='.checkpoint_', suffix='.tmp', dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp_name)
        # Windows rename fails if the destination exists; POSIX link does too.
        if os.name == 'nt':
            os.rename(temp_name, path)
        else:
            os.link(temp_name, path)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    with sidecar.open('x', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return path
