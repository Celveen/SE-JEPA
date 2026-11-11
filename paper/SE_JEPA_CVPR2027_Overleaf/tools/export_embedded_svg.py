"""Export embedded SVG fonts to vector outlines without font substitution.

The editable original is retained; only the print copy outlines glyphs.
Dependencies: fonttools, brotli, uharfbuzz, cairosvg.
"""
from pathlib import Path
from io import BytesIO
import argparse,base64,re,xml.etree.ElementTree as ET
import cairosvg
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
import uharfbuzz as hb
NS='http://www.w3.org/2000/svg';TAG='{'+NS+'}'
ET.register_namespace('',NS);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')

def export(source,destination):
 root=ET.parse(source).getroot();fonts={}
 for style in root.iter(TAG+'style'):
  for family,encoded in re.findall(r'font-family:\s*"([^"]+)";src:\s*url\("data:font/woff2;base64,([^\"]+)"\)',style.text or ''):
   font=TTFont(BytesIO(base64.b64decode(encoded)));font.flavor=None;ttf=BytesIO();font.save(ttf)
   shaper=hb.Font(hb.Face(ttf.getvalue()));upm=font['head'].unitsPerEm;shaper.scale=(upm,upm)
   fonts[family]=(font,shaper,upm,{})
 assert len(fonts)==3,'Expected the three embedded typefaces'
 count=0
 def visit(node,inherited):
  nonlocal count
  attrs=dict(inherited);attrs.update(node.attrib)
  # Print the light-mode colors already present in explicit SVG attributes.
  if 'style' in node.attrib:
   node.set('style',';'.join(s for s in node.get('style').split(';') if 'light-dark(' not in s and 'color-scheme' not in s))
  if node.tag!=TAG+'text':
   for child in list(node):
    if child.tag==TAG+'style':node.remove(child)
    else:visit(child,attrs)
   return
  assert not list(node),'Nested text spans need explicit handling'
  label=node.text or '';family=attrs['font-family'].strip('\'"');font,shaper,upm,cache=fonts[family]
  scale=float(attrs['font-size'].removesuffix('px'))/upm
  buf=hb.Buffer();buf.add_str(label);buf.guess_segment_properties();hb.shape(shaper,buf)
  assert all(i.codepoint!=0 for i in buf.glyph_infos),label
  width=sum(p.x_advance for p in buf.glyph_positions)*scale
  x=float(node.get('x','0'))-{'start':0,'middle':width/2,'end':width}[attrs.get('text-anchor','start')];y=float(node.get('y','0'))
  node.tag=TAG+'g';node.text=None;transform=node.get('transform','')
  for key in ['x','y']:node.attrib.pop(key,None)
  node.set('transform',f'{transform} translate({x:.9f},{y:.9f}) scale({scale:.12f},{-scale:.12f})'.strip());node.set('aria-label',label)
  cursor=0;glyphs=font.getGlyphSet();order=font.getGlyphOrder()
  for info,pos in zip(buf.glyph_infos,buf.glyph_positions):
   name=order[info.codepoint]
   if name not in cache:
    pen=SVGPathPen(glyphs);glyphs[name].draw(pen);cache[name]=pen.getCommands()
   if cache[name]:ET.SubElement(node,TAG+'path',{'d':cache[name],'transform':f'translate({cursor+pos.x_offset},{pos.y_offset})'})
   cursor+=pos.x_advance
  count+=1
 visit(root,{});root.attrib.pop('content',None)
 cairosvg.svg2pdf(bytestring=ET.tostring(root,encoding='utf-8'),write_to=str(destination),background_color='white')
 return {'embedded_fonts':sorted(fonts),'outlined_text_nodes':count}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();print(export(a.source,a.destination))
