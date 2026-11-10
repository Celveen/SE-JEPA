"""Insert two archived simulator observations into the native Figure 2 assets."""
from pathlib import Path
import base64,hashlib,json,re,shutil,xml.etree.ElementTree as ET
import pymupdf as fitz
import matplotlib
R=Path(__file__).resolve().parents[1]
FIG=R/'figures';DATA=R/'data/revision_assets/architecture_tasks';DATA.mkdir(parents=True,exist_ok=True)
original=FIG/'source/architecture_before_deform.pdf'
if not original.exists():shutil.copy2(FIG/'architecture.pdf',original)
source=FIG/'source/deform_qualitative_original.pdf'
with fitz.open(source) as d:
 for name,xref in [('rope',13),('granular',21)]:
  im=d.extract_image(xref);assert im['ext']=='png';(DATA/f'{name}.png').write_bytes(im['image'])
font_path=Path(matplotlib.get_data_path())/'fonts/ttf/STIXGeneral.ttf'
font=fitz.Font(fontfile=str(font_path))
with fitz.open(original) as doc:
 p=doc[0]
 p.replace_image(11,filename=str(DATA/'rope.png'));p.replace_image(12,filename=str(DATA/'granular.png'))
 for rect in [fitz.Rect(12,257,60,283),fitz.Rect(110,257,123,283)]:p.add_redact_annot(rect,fill=(1,1,1))
 p.apply_redactions(images=0,graphics=0)
 for label,cx in [('Rope',36),('Granular',117)]:
  x=cx-font.text_length(label,fontsize=16.5)/2
  p.insert_text((x,274.5),label,fontsize=16.5,fontfile=str(font_path),fontname='STIXTasks',color=(41/255,45/255,50/255))
 doc.save(FIG/'architecture.pdf',garbage=4,deflate=True)
# Keep the editable SVG and its embedded draw.io document consistent with the PDF.
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
svg=FIG/'source/architecture.svg';tree=ET.parse(svg);root=tree.getroot();images=root.findall('.//{'+NS+'}image')
for i,name in [(2,'rope'),(3,'granular')]:
 images[i].set('{'+XL+'}href','data:image/png;base64,'+base64.b64encode((DATA/f'{name}.png').read_bytes()).decode())
for node in root.findall('.//{'+NS+'}text'):
 if node.get('y')=='368' and node.get('x') in ('48','158'):
  for child in list(node):node.remove(child)
  node.text='Rope' if node.get('x')=='48' else 'Granular'
mx=ET.fromstring(root.get('content'))
for cell in mx.findall('.//mxCell'):
 cid=cell.get('id')
 if cid in ('19','21'):
  name='rope' if cid=='19' else 'granular';encoded=base64.b64encode((DATA/f'{name}.png').read_bytes()).decode()
  cell.set('style',re.sub(r'image=[^;]*;',f'image=data:image/png,{encoded};',cell.get('style')))
 if cid in ('20','22'):
  label='Rope' if cid=='20' else 'Granular';cx=85 if cid=='20' else 195;g=cell.find('mxGeometry')
  cell.set('value',label);g.set('width','100');g.set('x',str(cx-50))
root.set('content',ET.tostring(mx,encoding='unicode'))
tree.write(svg,encoding='utf-8',xml_declaration=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(DATA/'provenance.json').write_text(json.dumps({'source':'figures/source/deform_qualitative_original.pdf','source_sha256':sha(source),'selection':'Initial observations from the first rope and granular examples; embedded images extracted verbatim, including robot arm and goal overlays.','replaced_panels':['Square','T'],'new_panels':['Rope','Granular'],'unchanged_panels':['I','L'],'original_pdf':'figures/source/architecture_before_deform.pdf','images':{n:{'xref':x,'file':n+'.png','sha256':sha(DATA/(n+'.png'))} for n,x in [('rope',13),('granular',21)]}},indent=2)+'\n')
print('Updated native PDF/SVG with original rope and granular simulator pixels.')
