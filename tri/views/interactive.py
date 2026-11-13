"""Interactive 3-D page for a Scene: one self-contained HTML file (no network,
no libraries). Drag to turn, wheel to zoom, double-click to reset, hover for
names, click a legend entry to hide its group (alt-click: solo).

It draws what figure.draw() draws, from the same Scene: fields (hull faces of
figure.field_hulls with crease edges) or dots, locators as diamonds, group
centres, stars, the origin with labelled vectors, and the three axes."""
from __future__ import annotations

import html as _html
import json
import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .figure import AXIS_COLOURS, FAINT, Scene, colours, field_hulls

BG = "#0e1015"
FG = "#e6e6e6"


def _r(x) -> list:
    return [round(float(v), 4) for v in np.asarray(x, float).ravel()[:3]]


def scene_data(sc: Scene, title: str = "plot", axes: Optional[Sequence[str]] = None,
               view: Tuple[float, float] = (22, -58)) -> dict:
    """The page's data: points, hulls, vectors and legend, all in data units."""
    cols = colours(len(sc.names))
    P = np.asarray(sc.P, float)
    grp = np.asarray(sc.group, int)
    pc = list(sc.point_colour) if sc.point_colour is not None else [cols[g] if g >= 0 else FAINT for g in grp]
    lab = list(sc.labels) if sc.labels is not None else [str(sc.names[g]) if g >= 0 else "" for g in grp]
    pts = []
    fields = sc.style == "fields"
    for i, p in enumerate(P):
        pts.append(dict(p=_r(p), c=pc[i], s=round(sc.point_size ** 0.5 * 0.9, 2), n=str(lab[i]),
                        g="build", r=int(grp[i]) if grp[i] >= 0 else None))
    if sc.locators is not None:
        ll = list(sc.locator_labels) if sc.locator_labels is not None else ["locator"] * len(sc.locators)
        for i, p in enumerate(np.asarray(sc.locators, float)):
            pts.append(dict(p=_r(p), c="#ffffff", s=3.5, n=str(ll[i]), g="locator", r=None))
    if sc.centres is not None:
        for g, p in enumerate(np.asarray(sc.centres, float)):
            if np.all(np.isfinite(p)):
                pts.append(dict(p=_r(p), c=cols[g], s=6, n=str(sc.names[g]), g="centre", r=g, t=str(sc.names[g])[:24]))
    for lbl, p in sc.stars:
        pts.append(dict(p=_r(p), c="#ffd23f", s=7, n=str(lbl), g="star", r=None, t=str(lbl)[:30]))
    vec = []
    o = None
    if sc.origin is not None:
        o = _r(sc.origin)
        pts.append(dict(p=o, c="#ffffff", s=4.5, n="origin", g="origin", r=None))
        vec = [dict(a=o, b=_r(t), c=c, t=str(l)[:30]) for l, t, c in sc.vectors]
    hulls = [dict(r=h["g"], c=cols[h["g"]], v=[_r(x) for x in h["V"]], f=h["F"], e=[list(map(int, e)) for e in h["E"]])
             for h in (field_hulls(sc) if fields else [])]
    Q = np.array([q["p"] for q in pts] + [v["b"] for v in vec] or [[0, 0, 0]], float)
    lo, hi = Q.min(0), Q.max(0)
    at = o if o is not None else _r(np.clip(np.zeros(3), lo, hi))
    return dict(title=str(title), points=pts, hulls=hulls, vectors=vec, fields=fields, at=at,
                legend=[[str(n), _group_colour(g, grp, pc, cols)] for g, n in enumerate(sc.names)],
                axes=list(axes or ["x", "y", "z"]), axc=list(AXIS_COLOURS),
                el=math.radians(view[0]), az=math.radians(view[1]))


def _group_colour(g: int, grp: np.ndarray, pc: list, cols: list) -> str:
    i = np.flatnonzero(grp == g)
    return pc[int(i[0])] if len(i) else cols[g]


_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>html,body{margin:0;height:100%;background:__BG__;color:__FG__;font:13px system-ui,sans-serif;overflow:hidden}
#c{display:block;width:100vw;height:100vh;cursor:grab}
.pn{position:fixed;background:#15181fe6;border:1px solid #2a2f3a;border-radius:6px;padding:6px 9px}
#leg{right:10px;top:10px;max-height:70vh;overflow:auto}#leg div{white-space:nowrap;font-size:12px;cursor:pointer;user-select:none}
#leg div.off{opacity:.35}#leg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
#ax{position:fixed;left:12px;bottom:10px;color:#9a9a9a;font-size:11.5px;pointer-events:none}
#tip{position:fixed;pointer-events:none;display:none;max-width:380px;font-size:12px}</style></head><body>
<canvas id="c"></canvas><div id="leg" class="pn"></div><div id="ax"></div><div id="tip" class="pn"></div>
<script>
const D=__DATA__;const $=i=>document.getElementById(i);const cv=$('c'),g=cv.getContext('2d'),tip=$('tip');
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));const hidden=new Set();
const all=D.points.map(o=>o.p).concat(D.vectors.map(v=>v.b));let m=[0,0,0],r=1e-9;
for(let k=0;k<3;k++){let a=Infinity,b=-Infinity;for(const p of all){a=Math.min(a,p[k]);b=Math.max(b,p[k])}m[k]=(a+b)/2;r=Math.max(r,(b-a)/2)}
const nq=p=>p.map((x,k)=>(x-m[k])/r);const P=D.points.map(o=>Object.assign({q:nq(o.p)},o));
const H=D.hulls.map(h=>({r:h.r,c:h.c,v:h.v.map(nq),f:h.f,e:h.e}));const V=D.vectors.map(v=>({a:nq(v.a),b:nq(v.b),c:v.c,t:v.t}));
const AT=nq(D.at);const EL0=D.el,AZ0=D.az+Math.PI/2;let az=AZ0,el=EL0,zoom=1,drag=null,hover=-1;
function size(){cv.width=innerWidth*devicePixelRatio;cv.height=innerHeight*devicePixelRatio}addEventListener('resize',()=>{size();draw()});size();
function proj(q){const ca=Math.cos(az),sa=Math.sin(az),ce=Math.cos(el),se=Math.sin(el);
 const x=ca*q[0]-sa*q[1],y0=sa*q[0]+ca*q[1];const y=ce*q[2]-se*y0,z=se*q[2]+ce*y0;
 const s=Math.min(cv.width,cv.height)*0.42*zoom,f=3.2/(3.2+z);return[cv.width/2+x*s*f,cv.height/2-y*s*f,z,f]}
const vis=o=>!(o.r!=null&&hidden.has(o.r));
function line(a,b){g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke()}
function draw(){g.clearRect(0,0,cv.width,cv.height);const dp=devicePixelRatio;g.lineWidth=dp;g.font=`${11*dp}px system-ui`;
 for(let k=0;k<3;k++){const a=AT.slice(),b=AT.slice();a[k]=-1.05;b[k]=1.05;const A=proj(a),B=proj(b);g.globalAlpha=0.55;g.strokeStyle=D.axc[k];line(A,B)}
 g.globalAlpha=1;
 if(D.fields){const T=[];for(const h of H){if(hidden.has(h.r))continue;const W=h.v.map(proj);for(const f of h.f)T.push([(W[f[0]][2]+W[f[1]][2]+W[f[2]][2])/3,W,f,h.c])}
  T.sort((a,b)=>b[0]-a[0]);for(const[,W,f,c]of T){g.globalAlpha=0.16;g.fillStyle=c;g.beginPath();g.moveTo(W[f[0]][0],W[f[0]][1]);g.lineTo(W[f[1]][0],W[f[1]][1]);g.lineTo(W[f[2]][0],W[f[2]][1]);g.closePath();g.fill()}
  g.globalAlpha=0.9;g.lineWidth=1.2*dp;for(const h of H){if(hidden.has(h.r))continue;const W=h.v.map(proj);g.strokeStyle=h.c;for(const[a,b]of h.e)line(W[a],W[b])}}
 const S=[];P.forEach((o,i)=>{if(vis(o)&&!(D.fields&&o.g==='build'&&i!==hover))S.push([proj(o.q),i])});S.sort((a,b)=>b[0][2]-a[0][2]);
 for(const[[x,y,z,f],i]of S){const o=P[i];const big=o.g!=='build';const rad=o.s*dp*f;g.globalAlpha=big?0.95:0.6;g.fillStyle=o.c;g.beginPath();
  if(o.g==='locator'){g.moveTo(x,y-rad);g.lineTo(x+rad,y);g.lineTo(x,y+rad);g.lineTo(x-rad,y);g.closePath()}
  else if(o.g==='star'){for(let k=0;k<10;k++){const t=-Math.PI/2+k*Math.PI/5,q=k%2?rad*0.45:rad*1.2;k?g.lineTo(x+q*Math.cos(t),y+q*Math.sin(t)):g.moveTo(x+q*Math.cos(t),y+q*Math.sin(t))}g.closePath()}
  else g.arc(x,y,rad,0,7);g.fill();
  if(big||i===hover){g.globalAlpha=1;g.strokeStyle=i===hover?'#ffffff':'#202020';g.lineWidth=dp;g.stroke()}
  if(o.t){g.globalAlpha=0.85;g.fillStyle='#9a9a9a';g.fillText(o.t,x+rad+3*dp,y+4*dp)}}
 g.globalAlpha=0.95;g.lineWidth=1.6*dp;for(const v of V){const A=proj(v.a),B=proj(v.b);g.strokeStyle=v.c;line(A,B);
  const ang=Math.atan2(B[1]-A[1],B[0]-A[0]),h=7*dp;g.fillStyle=v.c;g.beginPath();g.moveTo(B[0],B[1]);g.lineTo(B[0]-h*Math.cos(ang-0.4),B[1]-h*Math.sin(ang-0.4));g.lineTo(B[0]-h*Math.cos(ang+0.4),B[1]-h*Math.sin(ang+0.4));g.closePath();g.fill();
  g.fillText(v.t,B[0]+4*dp,B[1]-4*dp)}g.globalAlpha=1;
 $('ax').innerHTML=D.axes.map((a,i)=>`<span style="color:${D.axc[i]}">&#9632;</span> ${esc(a)}`).join('<br>')}
function pick(ex,ey){const dp=devicePixelRatio,X=ex*dp,Y=ey*dp;let best=-1,bd=1e9;P.forEach((o,i)=>{if(!vis(o))return;const[x,y]=proj(o.q);const d=(x-X)**2+(y-Y)**2;if(d<bd){bd=d;best=i}});return bd<(9*dp)**2?best:-1}
function legend(){if(!D.legend.length){$('leg').style.display='none';return}
 $('leg').innerHTML=D.legend.map(([n,c],k)=>`<div data-r="${k}" class="${hidden.has(k)?'off':''}"><i style="background:${c}"></i>${esc(n)}</div>`).join('');
 for(const d of $('leg').querySelectorAll('div[data-r]'))d.onclick=e=>{const k=+d.dataset.r;if(e.altKey){const solo=hidden.size===D.legend.length-1&&!hidden.has(k);hidden.clear();if(!solo)D.legend.forEach((_,j)=>{if(j!==k)hidden.add(j)})}else hidden.has(k)?hidden.delete(k):hidden.add(k);legend();draw()}}
cv.onmousedown=e=>{drag=[e.clientX,e.clientY,az,el]};addEventListener('mouseup',()=>{drag=null});
cv.onmousemove=e=>{if(drag){az=drag[2]+(e.clientX-drag[0])*0.008;el=Math.max(-1.55,Math.min(1.55,drag[3]+(e.clientY-drag[1])*0.008));tip.style.display='none';draw();return}
 const h=pick(e.clientX,e.clientY);if(h!==hover){hover=h;draw()}if(h>=0&&P[h].n){tip.textContent=P[h].n;tip.style.display='block';tip.style.left=e.clientX+14+'px';tip.style.top=e.clientY+10+'px'}else tip.style.display='none'};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(0.3,Math.min(12,zoom*Math.exp(-e.deltaY*0.0015)));draw()};
cv.ondblclick=()=>{az=AZ0;el=EL0;zoom=1;draw()};
let t0=null;cv.addEventListener('touchstart',e=>{const t=e.touches[0];t0=[t.clientX,t.clientY,az,el]},{passive:true});
cv.addEventListener('touchmove',e=>{const t=e.touches[0];az=t0[2]+(t.clientX-t0[0])*0.008;el=Math.max(-1.55,Math.min(1.55,t0[3]+(t.clientY-t0[1])*0.008));draw()},{passive:true});
legend();draw();
</script></body></html>"""


def page(sc: Scene, title: str = "plot", axes: Optional[Sequence[str]] = None,
         view: Tuple[float, float] = (22, -58)) -> str:
    """The interactive page for a Scene as an HTML string."""
    data = scene_data(sc, title, axes, view)
    js = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return (_PAGE.replace("__TITLE__", _html.escape(str(title))).replace("__BG__", BG)
            .replace("__FG__", FG).replace("__DATA__", js))
