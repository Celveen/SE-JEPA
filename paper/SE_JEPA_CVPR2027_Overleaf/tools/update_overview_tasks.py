"""Insert archived simulator observations into the author's revised Figure 2."""
from pathlib import Path
import base64,hashlib,json,re,xml.etree.ElementTree as ET
import pymupdf as fitz
from export_embedded_svg import export
R=Path(__file__).resolve().parents[1];FIG=R/'figures';DATA=R/'data/revision_assets/architecture_tasks'
SOURCE=FIG/'source/architecture_author_20261110.svg';SIM=FIG/'source/deform_qualitative_original.pdf'
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
with fitz.open(SIM) as doc:
 for name,xref in [('rope',13),('granular',21)]:
  im=doc.extract_image(xref);assert im['ext']=='png';(DATA/f'{name}.png').write_bytes(im['image'])
tree=ET.parse(SOURCE);root=tree.getroot();images=root.findall('.//{'+NS+'}image');assert len(images)==14
for index,name in [(2,'rope'),(3,'granular')]:images[index].set('{'+XL+'}href','data:image/png;base64,'+base64.b64encode((DATA/f'{name}.png').read_bytes()).decode())
for node in root.findall('.//{'+NS+'}text'):
 if (node.get('x'),node.get('y')) in [('48','363'),('156','363')]:
  assert node.text in ['Square','T'];node.text='Rope' if node.get('x')=='48' else 'Granular'
# Update the embedded draw.io source while retaining the author's cell centers.
mx=ET.fromstring(root.get('content'))
for cell in mx.findall('.//mxCell'):
 cid=cell.get('id')
 if cid in ('19','21'):
  name='rope' if cid=='19' else 'granular';encoded=base64.b64encode((DATA/f'{name}.png').read_bytes()).decode()
  cell.set('style',re.sub(r'image=[^;]*;',f'image=data:image/png,{encoded};',cell.get('style')))
 if cid in ('20','22'):
  g=cell.find('mxGeometry');center=float(g.get('x'))+float(g.get('width'))/2
  cell.set('value','Rope' if cid=='20' else 'Granular');g.set('width','100');g.set('x',str(center-50))
root.set('content',ET.tostring(mx,encoding='unicode'));svg=FIG/'source/architecture.svg';tree.write(svg,encoding='utf-8',xml_declaration=True)
font_export=export(svg,FIG/'architecture.pdf')
with fitz.open(FIG/'architecture.pdf') as doc:doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(FIG/'architecture.png')
(DATA/'provenance.json').write_text(json.dumps({
 'source':str(SIM.relative_to(R)),'source_sha256':sha(SIM),
 'selection':'Initial observations from the first rope and granular examples; embedded images extracted verbatim, including robot arm and goal overlays.',
 'replaced_panels':['Square','T'],'new_panels':['Rope','Granular'],'unchanged_panels':['I','L'],
 'author_revision':str(SOURCE.relative_to(R)),'author_revision_sha256':sha(SOURCE),'author_colors_positions_and_fonts_preserved':True,
 'font_export':font_export,'canvas':{'width':root.get('width'),'height':root.get('height'),'viewBox':root.get('viewBox')},
 'images':{n:{'xref':x,'file':n+'.png','sha256':sha(DATA/(n+'.png'))} for n,x in [('rope',13),('granular',21)]}
 },indent=2)+'\n')
print('Updated the author revision with original rope and granular simulator observations.')
