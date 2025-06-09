import os
import json
import pickle
from pathlib import Path

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def save_json(data, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(data, filepath):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_files_with_extensions(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(f'**/*{ext}'))
    return [str(f) for f in files]