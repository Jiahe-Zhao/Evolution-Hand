"""Build the relocatable Nov20 input snapshot from a source experiment directory."""
import argparse, ast, hashlib, json, os, tempfile, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


def build(source, dest):
    names = set()
    for p in (source/'scripts/nov20').rglob('*.py'):
        try: tree = ast.parse(p.read_text())
        except (SyntaxError, UnicodeError): continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value,str) and n.value.endswith('.npz') and '*' not in n.value:
                names.add(Path(n.value).name)
    inputs = set()
    for p in (source/'outputs/nov20').rglob('*'):
        if p.is_file() and not p.name.startswith('._') and (p.suffix in {'.xml','.npy','.json','.obj','.mjb'} or (p.suffix=='.npz' and p.name in names)):
            inputs.add(p)
    inputs.update(p for p in (source/'outputs/v5_acceptance/hand_111').rglob('*') if p.is_file() and p.suffix in {'.urdf','.stl','.obj','.json'})
    xml_data={}; missing=[]; excluded=[]
    for p in list(inputs):
        if p.suffix not in {'.xml','.urdf'}:continue
        try: tree=ET.parse(p)
        except ET.ParseError:
            inputs.remove(p); excluded.append(str(p.relative_to(source))); continue
        for node in tree.iter():
            for key in ('file','filename'):
                val=node.get(key)
                if not val:continue
                ref=Path(val) if Path(val).is_absolute() else p.parent/val
                if not ref.exists():
                    alternate=source/'outputs/v5_acceptance/hand_111/meshes'/Path(val).name
                    if alternate.exists():ref=alternate
                if not ref.exists():missing.append((str(p.relative_to(source)),val));continue
                ref=ref.resolve()
                if not ref.is_relative_to(source):raise ValueError(f'External asset: {ref}')
                inputs.add(ref)
                node.set(key,os.path.relpath(ref,p.parent))
        xml_data[p]=ET.tostring(tree.getroot(),encoding='utf-8',xml_declaration=True)
    if missing:raise ValueError(f'Missing asset references: {missing[:15]}')
    dest.mkdir(parents=True,exist_ok=True)
    files={}
    with tempfile.TemporaryDirectory() as temp:
        archive=Path(temp)/'scene_inputs.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in sorted(inputs):
                data=xml_data.get(p)
                if data is None:data=p.read_bytes()
                relative=str(p.relative_to(source))
                files[relative]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
                z.writestr(relative,data)
        parts=[];digest=hashlib.sha256()
        with archive.open('rb') as f:
            index=0
            while data:=f.read(24*1024*1024):
                name=f'nov20_inputs_v1.zip.part{index:03d}';(dest/name).write_bytes(data);digest.update(data)
                parts.append({'file':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)});index+=1
        manifest={'version':'nov20-inputs-20261128','archive_sha256':digest.hexdigest(),'parts':parts,'files':files,'excluded_invalid_xml':excluded,'scope':'Model sources and fixed reference inputs. Not training checkpoints or complete experiment history.'}
        (dest/'nov20_inputs_v1.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(files),'parts':len(parts),'compressed_bytes':sum(x['bytes'] for x in parts)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args();build(a.source.resolve(),a.dest.resolve())
