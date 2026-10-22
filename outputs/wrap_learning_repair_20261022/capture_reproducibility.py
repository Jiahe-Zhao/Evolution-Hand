"""Record external dependencies and selected weights without copying large assets."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parent.parent
paths = set((root.parent / 'contact_control_calibration_20261021').glob('*.py'))
paths.add(root.parent / 'contact_control_calibration_20261021/plans_wrap_cover_selected.json')
paths.add(root.parent / 'strike_v3_training_20261020/bc_seeds/strike/ppo/params/agent.yaml')
paths.add(repo / 'Isaaclab_other/isaaclab_tool.py')
paths.update((root.parent / 'motion_collision_revision_20261021/matrix/baseline/right').rglob('*'))
for run in ['bc_position01', 'bc_contraction20', 'bc_contraction29', 'ppo_completion31']:
    paths.update((root / run).glob('*.pth'))
for run in ['bc_position01', 'bc_contraction20', 'bc_contraction29']:
    manifest = root / run / 'dataset.json'
    if manifest.exists():
        for item in json.loads(manifest.read_text())['sources']:
            folder = root.parent / 'wrap_robustness_20261021' / item['run']
            paths.update([folder / 'trace.json', folder / 'audit.json'])
paths.add(root.parent / 'wrap_robustness_20261021/perturbation_results.json')
rows = []
for path in sorted(paths):
    if not path.is_file():
        continue
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    rows.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest()))
versions = {}
for name in ['torch', 'rl-games', 'numpy', 'gym', 'tensorboard']:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = 'not found'
(root / 'reproducibility_manifest.json').write_text(json.dumps(dict(
    python='/home/zjh/miniconda3/envs/evolution_isaaclab/bin/python',
    versions=versions, files=rows,
    scope='External dependencies and available selected weights. Raw assets remain on the remote host; this Git archive is not a standalone asset bundle.'
), indent=2))
print(len(rows), 'dependency hashes recorded')
