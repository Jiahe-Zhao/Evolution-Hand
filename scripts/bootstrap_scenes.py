#!/usr/bin/env python3
"""Verify and prepare all versioned Newton scene inputs in this checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def prepare_inputs():
    folder = ROOT / 'assets/releases/nov20'
    manifest = json.loads((folder/'nov20_inputs_v1.json').read_text())
    archive_hash = hashlib.sha256()
    with tempfile.TemporaryFile() as archive:
        for entry in manifest['parts']:
            data = (folder/entry['file']).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise RuntimeError('Scene input checksum mismatch: '+entry['file'])
            archive_hash.update(data);archive.write(data)
        if archive_hash.hexdigest() != manifest['archive_sha256']:
            raise RuntimeError('Scene input archive checksum mismatch')
        archive.seek(0)
        with zipfile.ZipFile(archive) as z:
            for name, info in manifest['files'].items():
                dest = (ROOT/name).resolve()
                if not dest.is_relative_to(ROOT):raise RuntimeError('Unsafe path: '+name)
                data=z.read(name)
                if hashlib.sha256(data).hexdigest()!=info['sha256']:
                    raise RuntimeError('Scene input file checksum mismatch: '+name)
                if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=info['sha256']:
                    raise RuntimeError('Existing modified input would be overwritten: '+name+'. Use a fresh checkout.')
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.write_bytes(data)
    seed_folder=ROOT/'assets/scene_seed_overrides'
    seeds=json.loads((seed_folder/'manifest.json').read_text())
    seed_archive=zipfile.ZipFile(seed_folder/'seeds.zip')
    for name, expected in seeds.items():
        data=seed_archive.read(name)
        if hashlib.sha256(data).hexdigest()!=expected:
            raise RuntimeError('Seed checksum mismatch: '+name)
        dest=ROOT/name
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Modified input seed: '+name+'. Use a fresh checkout.')
        dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():dest.write_bytes(data)
    seed_archive.close()
    print('Scene input files verified:',len(manifest['files']),'; named seeds:',len(seeds))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true',help='Install Python dependencies into the invoking interpreter/environment.')
    args=parser.parse_args()
    package=ROOT/'outputs/ax50_visualskin_v3_0_0'
    if not package.exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/unpack_ax50.py'),'--dest',str(ROOT/'outputs')],check=True)
    else:
        checks=json.loads((package/'SHA256SUMS.json').read_text())
        for name,expected in checks.items():
            if hashlib.sha256((package/name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Existing AX50 package was modified: '+name+'. Use a fresh checkout.')
    prepare_inputs()
    if args.install:
        subprocess.run([sys.executable,'-m','pip','install','-r',str(ROOT/'requirements-scenes.txt'),str(package/'dist/ax50_visualskin-3.0.0-py3-none-any.whl')],check=True)
    print('Ready. List scenes: python scripts/run_scene.py --list')

if __name__=='__main__':main()
