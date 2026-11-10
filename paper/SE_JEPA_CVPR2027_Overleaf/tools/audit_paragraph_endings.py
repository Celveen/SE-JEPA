"""Check rendered prose/caption endings and bibliography entries against 40% width.

Usage: python tools/audit_paragraph_endings.py main.pdf main.aux report.json
The source uses one paragraph per long TeX line. Headings, equations, algorithm
steps and table cells are not paragraphs. All source candidates must be matched.
"""
from pathlib import Path
import re,json,unicodedata,argparse
import pymupdf as fitz
R=Path(__file__).resolve().parents[1]
def norm(s):
 s=unicodedata.normalize('NFKC',s).lower()
 return ''.join(c for c in s if c.isalnum())
def clean(s,labels):
 s=re.sub(r'(?<!\\)%.*','',s)
 for a,b in [(r'\ours','SE-JEPA'),(r'\emph',''),(r'\textbf',''),(r'\textit',''),(r'\texttt',''),(r'\newblock',''),(r'\looseness=-1',''),(r'\looseness=1','')]:s=s.replace(a,b)
 s=re.sub(r'\\(?:mainref|ref)\{([^}]+)\}',lambda m:labels.get(m[1],''),s)
 s=re.sub(r'\\supref\{([^}]+)\}',r'Appendix \1',s)
 s=re.sub(r'\\cite\{[^}]+\}','',s)
 s=re.sub(r'\\[a-zA-Z]+\*?', '',s)
 return s

def audit_references(doc):
 first=next(i for i,p in enumerate(doc) if 'References' in p.get_text())
 last=next((i for i,p in enumerate(doc) if i>first and 'Supplementary Material' in p.get_text()),len(doc))
 entries=[];current=None
 for i in range(first,last):
  for low,high in [(58,300),(317,560)]:
   spans=[s for b in doc[i].get_text('dict')['blocks'] for l in b.get('lines',[]) for s in l['spans'] if 'NimbusRom' in s['font'] and 8<s['size']<9.5 and low<=s['bbox'][0]<high and s['bbox'][1]<735]
   spans.sort(key=lambda s:(s['origin'][1],s['bbox'][0]));groups=[]
   for s in spans:
    if not groups or abs(s['origin'][1]-groups[-1][0]['origin'][1])>.8:groups.append([s])
    else:groups[-1].append(s)
   for g in groups:
    g.sort(key=lambda s:s['bbox'][0]);text=''.join(s['text'] for s in g);m=re.match(r'\[(\d+)\]',text)
    if m:
     if current:entries.append(current)
     current={'number':int(m[1]),'page':i+1}
    if current:
     current['last_line']=text;current['last_line_page']=i+1;current['bbox']=[min(s['bbox'][0] for s in g),min(s['bbox'][1] for s in g),max(s['bbox'][2] for s in g),max(s['bbox'][3] for s in g)];current['ratio']=round((current['bbox'][2]-current['bbox'][0])/(238.5-19.924),4)
 if current:entries.append(current)
 return {'checked':len(entries),'short':[e for e in entries if e['ratio']<.4],'entries':entries}

def audit(pdf,aux,out):
 labels={l:n for l,n in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}',aux.read_text())}
 doc=fitz.open(pdf);lines=[];stream='';spans=[]
 for pno,p in enumerate(doc):
  for b in p.get_text('dict')['blocks']:
   for line in b.get('lines',[]):
    ts=[s for s in line['spans'] if not (s['font'].startswith('NimbusSan') or s['size']<7.8)]
    if not ts or not any('NimbusRom' in s['font'] for s in ts):continue
    # Main paper prose/captions use Times; embedded figure typography is excluded.
    if min(s['bbox'][0] for s in ts)<45:continue
    txt=''.join(s['text'] for s in ts)
    if not norm(txt):continue
    box=[min(s['bbox'][0] for s in ts),min(s['bbox'][1] for s in ts),max(s['bbox'][2] for s in ts),max(s['bbox'][3] for s in ts)]
    n=norm(txt);lo=len(stream);stream+=n
    lines.append(dict(page=pno+1,text=txt,bbox=box,stream_start=lo,stream_end=len(stream)))
 candidates=[]
 for p in sorted((R/'sec').glob('*.tex')):
  in_math=False
  for i,s in enumerate(p.read_text().splitlines()):
   s=s.strip()
   if s.startswith(r'\begin{equation}'):in_math=True
   if s.startswith(r'\end{equation}'):in_math=False;continue
   if in_math:continue
   if len(s)<55 or s.startswith('%'):continue
   cap=s.startswith(r'\caption{')
   if not cap and not (s.endswith('.') or s.endswith('}') and not s.startswith('\\')):continue
   if any(s.startswith(x) for x in [r'\label',r'\include',r'\State',r'\Require',r'\Ensure']):continue
   if cap:s=s[len(r'\caption{'):-1]
   cleaned=norm(clean(s,labels));suffix=cleaned[-65:]
   pos=stream.find(suffix)
   if pos<0:
    # Inline math and references can vary in PDF extraction; use a shorter unique ending.
    suffix=cleaned[-35:];pos=stream.find(suffix)
   if pos<0 and stream.count(cleaned[-20:])==1:
    suffix=cleaned[-20:];pos=stream.find(suffix)
   if pos<0 and stream.count(cleaned[-16:])==1:
    suffix=cleaned[-16:];pos=stream.find(suffix)
   if pos<0:
    candidates.append(dict(file=p.name,line=i+1,status='unmatched',ending=s[-100:]));continue
   end=pos+len(suffix)
   line=next(l for l in lines if l['stream_start']<end<=l['stream_end'])
   x0,y0,x1,y1=line['bbox']
   # Full-width captions are 495pt; narrow captions and prose are 238.5pt.
   width=495 if cap and x1>305 and x0<310 else 238.5
   # Determine full-width captions from earlier matched text, even if the final line is short.
   if cap:
    starts=stream.rfind(norm(clean(s,labels))[:45],0,end)
    first=next((l for l in lines if l['stream_start']<=starts<l['stream_end']),None)
    if first and first['bbox'][2]-first['bbox'][0]>300:width=495
   ratio=(x1-x0)/width
   candidates.append(dict(file=p.name,line=i+1,status='ok' if ratio>=.4 else 'short',page=line['page'],ratio=round(ratio,4),last_line=line['text'],bbox=line['bbox'],ending=s[-100:],kind='caption' if cap else 'prose'))
 report=dict(pdf=str(pdf),checked=len(candidates),matched=sum(x['status']!='unmatched' for x in candidates),short=[x for x in candidates if x['status']=='short'],unmatched=[x for x in candidates if x['status']=='unmatched'],paragraphs=candidates)
 report['bibliography']=audit_references(doc)
 report['threshold']=.4
 report['scope']='Abstract, body prose, figure/table captions, appendices, and bibliography; excludes headings, equations, algorithms, and table cells.'
 report['pdf']=pdf.name
 out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
 print('Paragraphs',report['checked'],'matched',report['matched'],'short',len(report['short']),'unmatched',len(report['unmatched']))
 print('Bibliography',report['bibliography']['checked'],'short',len(report['bibliography']['short']))
 for x in report['short']:print(f"{x['file']}:{x['line']} p{x['page']} {x['ratio']:.0%} | {x['last_line']}")
 for x in report['unmatched']:print('UNMATCHED',x['file'],x['line'],x['ending'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('pdf',type=Path);p.add_argument('aux',type=Path);p.add_argument('out',type=Path);a=p.parse_args();audit(a.pdf,a.aux,a.out)
