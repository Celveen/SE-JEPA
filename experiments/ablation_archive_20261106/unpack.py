"""Verify and extract the archived experiments using only Python's standard library."""
from pathlib import Path, PurePosixPath
import argparse,hashlib,json,os,shutil,tempfile,zipfile

ROOT=Path(__file__).resolve().parent
BLOCK=4*1024**2
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(BLOCK),b''):h.update(block)
 return h.hexdigest()

def safe(base,name):
 p=PurePosixPath(name)
 if p.is_absolute() or not p.parts or any(x in ('','..','.') for x in name.split('/')) or '\\' in name or ':' in name:
  raise ValueError('Unsafe archive path: '+name)
 out=base.joinpath(*p.parts)
 if not out.resolve().is_relative_to(base.resolve()):raise ValueError('Path escapes destination')
 cursor=out
 while cursor!=base:
  if cursor.is_symlink():raise ValueError('Symlink destination rejected')
  cursor=cursor.parent
 return out

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--verify-only',action='store_true')
 parser.add_argument('--output',type=Path,default=ROOT/'extracted')
 parser.add_argument('--package',action='append',help='Package name from packages.json; default: all')
 args=parser.parse_args()
 destination=args.output.resolve()
 index=json.loads((ROOT/'packages.json').read_text(encoding='utf-8'))
 packages=index['packages']
 if args.package:
  assert set(args.package)<={p['name'] for p in packages},'Unknown package'
  packages=[p for p in packages if p['name'] in args.package]
 scratch=ROOT/'.unpack_work';scratch.mkdir(exist_ok=True)
 verified=0
 with tempfile.TemporaryDirectory(dir=scratch) as temp:
  for package in packages:
   archive=Path(temp)/package['archive']
   with archive.open('wb') as target:
    for part in package['parts']:
     source=safe(ROOT,part['file'])
     assert source.stat().st_size==part['bytes'] and sha(source)==part['sha256'],str(source)
     with source.open('rb') as stream:shutil.copyfileobj(stream,target,BLOCK)
   assert archive.stat().st_size==package['archive_bytes'] and sha(archive)==package['archive_sha256']
   with zipfile.ZipFile(archive) as z:
    manifest_name='_package_manifests/'+package['name']+'.json'
    manifest=json.loads(z.read(manifest_name))
    expected={row['path']:row for row in manifest['files']}
    assert len(expected)==len(manifest['files'])
    assert set(z.namelist())==set(expected)|{manifest_name}
    assert len(z.namelist())==len(expected)+1
    for name in z.namelist():
     target=safe(destination,name)
     if name!=manifest_name:
      entry=expected[name];h=hashlib.sha256();size=0
      with z.open(name) as stream:
       for block in iter(lambda:stream.read(BLOCK),b''):h.update(block);size+=len(block)
      assert size==entry['bytes'] and h.hexdigest()==entry['sha256'],name
      verified+=1
     if not args.verify_only:
      payload_hash=expected[name]['sha256'] if name in expected else hashlib.sha256(z.read(name)).hexdigest()
      if target.exists():
       assert target.is_file() and sha(target)==payload_hash,'Existing file differs: '+str(target)
       continue
      target.parent.mkdir(parents=True,exist_ok=True)
      temporary=target.with_name(target.name+'.partial')
      assert not temporary.exists(),'Partial extraction already exists: '+str(temporary)
      with z.open(name) as source,temporary.open('xb') as stream:shutil.copyfileobj(source,stream,BLOCK)
      os.replace(temporary,target)
   archive.unlink()
   print(('VERIFIED ' if args.verify_only else 'EXTRACTED ')+package['name'],flush=True)
 print(f'OK: {len(packages)} packages, {verified} verified files.')
 if not args.verify_only:print('Output: '+str(destination))

if __name__=='__main__':main()
