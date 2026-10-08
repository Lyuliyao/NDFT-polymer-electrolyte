from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / 'work' / 'runtime'

def link(source, target):
    if target.is_symlink() or target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())

def prepare():
    roots = {}
    for system in ('eps75', 'eps2', 'lj1', 'bigbox', 'field2d'):
        root = RUNTIME / ('salt_in_polymer_' + system)
        roots[system] = root
        runs = root / 'runs'
        runs.mkdir(parents=True, exist_ok=True)
        md = ROOT / 'data' / 'md' / system
        for path in md.iterdir() if md.exists() else []:
            link(path, runs / path.name)
        models = ROOT / 'data' / 'models' / system
        learned = runs / 'learn'
        learned.mkdir(exist_ok=True)
        for path in models.iterdir() if models.exists() else []:
            link(path, learned / path.name)
        link(ROOT / 'code', root / 'code')
        split = md / ('heldout.json' if system == 'lj1' else 'heldout_split.json')
        if split.exists():
            link(split, root / 'splits' / 'heldout.json')
        excluded = md / 'excluded.json'
        if excluded.exists():
            link(excluded, root / 'splits' / 'excluded.json')
        predictions = ROOT / 'data' / 'predictions' / system
        for path in predictions.iterdir() if predictions.exists() else []:
            link(path, root / path.name)
    link(ROOT / 'data/md/field2d', roots['eps75'] / 'field2d')
    link(ROOT / 'data/md/bigbox', roots['eps75'] / 'bigbox')
    print(json.dumps({k: str(v.relative_to(ROOT)) for k,v in roots.items()},indent=2))
    return roots

if __name__ == '__main__':
    prepare()
