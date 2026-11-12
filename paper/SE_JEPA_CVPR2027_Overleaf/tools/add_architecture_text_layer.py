"""Restore selectable SVG labels over the exact outlined print artwork.

Original vector outlines remain visible. Embedded original fonts form a matching
invisible Unicode text layer, so copy/search work without font substitution.
"""
from pathlib import Path
from io import BytesIO
import base64, json, re, xml.etree.ElementTree as ET
import pymupdf as fitz
from fontTools.ttLib import TTFont
import uharfbuzz as hb

ROOT=Path(__file__).resolve().parents[1]
SVG=ROOT/'figures/source/architecture.svg'
PDF=ROOT/'figures/architecture.pdf'
TAG='{http://www.w3.org/2000/svg}'

def main():
    root=ET.parse(SVG).getroot(); fonts={}
    for style in root.iter(TAG+'style'):
        for family,encoded in re.findall(r'font-family:\s*"([^"]+)";src:\s*url\("data:font/woff2;base64,([^\"]+)"\)',style.text or ''):
            f=TTFont(BytesIO(base64.b64decode(encoded))); f.flavor=None
            b=BytesIO(); f.save(b); data=b.getvalue(); shaper=hb.Font(hb.Face(data))
            upm=f['head'].unitsPerEm; shaper.scale=(upm,upm)
            fonts[family]=(data,shaper,upm)
    d=fitz.open(PDF); p=d[0]
    if p.get_text().strip():
        print('Architecture already has a text layer.'); return
    before=p.get_pixmap(matrix=fitz.Matrix(1,1)).samples
    scale=p.rect.width/float(root.get('viewBox').split()[2])
    aliases={f:'svg'+str(i) for i,f in enumerate(fonts)}
    for family,(data,_,_) in fonts.items():
        p.insert_font(fontname=aliases[family],fontbuffer=data)
    labels=[]
    def visit(node,inherited):
        attrs=dict(inherited); attrs.update(node.attrib)
        if node.get('transform'):
            assert not list(node.iter(TAG+'text')), 'Text transform handling needed'
        if node.tag==TAG+'text':
            assert not list(node)
            label=node.text or ''; family=attrs['font-family'].strip('\'"')
            _,shaper,upm=fonts[family]; size=float(attrs['font-size'].removesuffix('px'))
            b=hb.Buffer();b.add_str(label);b.guess_segment_properties();hb.shape(shaper,b)
            width=sum(pos.x_advance for pos in b.glyph_positions)*size/upm
            anchor={'start':0,'middle':0.5,'end':1}[attrs.get('text-anchor','start')]
            x=(float(attrs.get('x',0))-anchor*width)*scale; y=float(attrs.get('y',0))*scale
            p.insert_text((x,y),label,fontname=aliases[family],fontsize=size*scale,render_mode=3)
            labels.append(label)
        for c in node: visit(c,attrs)
    visit(root,{})
    d.subset_fonts(); output=PDF.with_suffix('.searchable.pdf');d.save(output,garbage=4,deflate=True)
    d.close(); result=fitz.open(output)
    assert result[0].get_pixmap(matrix=fitz.Matrix(1,1)).samples==before
    recovered=result[0].get_text()
    assert all(label in recovered for label in labels),[x for x in labels if x not in recovered]
    result.close(); output.replace(PDF)
    (ROOT/'data/revision_assets/architecture_tasks/text_layer_audit.json').write_text(json.dumps(dict(
        selectable_labels=len(labels),visible_render_pixel_identical=True,
        font_families=list(fonts),source='figures/source/architecture.svg',
        method='Original outlined artwork plus aligned invisible text using the original embedded fonts.'),indent=2)+'\n')
    print(f'Restored {len(labels)} selectable labels; rendered pixels unchanged.')

if __name__=='__main__':main()
