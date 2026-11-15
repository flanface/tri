"""Interactive 3-D page for a Scene: one self-contained HTML file (no network,
no libraries), orthographic.

It draws what figure.draw() draws from the same Scene (fields or dots,
locators, centres, stars, origin and vectors, axes) and adds tools:

    view     drag to turn, right/shift-drag to pan, wheel to zoom
    note     text anchored at a point of the plot
    pen      strokes on the screen plane through the centre; they turn with the plot
    point    a marker (snaps to nearby points)
    vector   tail to head; |v| shown
    plane    through three clicked points
    measure  two vectors: |u|, |w|, cos, angle, orthogonality and the projection
             length of each on the other; two points: distance

Depth: points are shaded spheres that shrink and fade with distance, field
faces are lit (Lambert, light over the viewer's shoulder), vectors cast a
shadow on the floor of the scene with a drop line from the head. PNG export
is transparent.

The command box takes a name (a build, item, locator or centre; in a scene
with an origin it adds that point's vector from the origin), x,y,z (a point),
a > b (a vector; either end a name or x,y,z; empty tail = origin),
plane a,b,c = d, plane x = c, note text @ place. Annotations persist in the
browser, export to JSON and load back; PNG saves the view. Legend entries hide
a group (alt-click: solo).
"""
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
    return dict(title=str(title), points=pts, hulls=hulls, vectors=vec, fields=fields, at=at, origin=o,
                legend=[[str(n), _group_colour(g, grp, pc, cols)] for g, n in enumerate(sc.names)],
                axes=list(axes or ["x", "y", "z"]), axc=list(AXIS_COLOURS),
                el=math.radians(view[0]), az=math.radians(view[1]))


def _group_colour(g: int, grp: np.ndarray, pc: list, cols: list) -> str:
    i = np.flatnonzero(grp == g)
    return pc[int(i[0])] if len(i) else cols[g]


_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>html,body{margin:0;height:100%;background:__BG__;color:__FG__;font:13px system-ui,sans-serif;overflow:hidden}
#c{display:block;width:100vw;height:100vh}
.pn{position:fixed;background:#15181fe6;border:1px solid #2a2f3a;border-radius:6px;padding:6px 9px}
#left{position:fixed;left:10px;top:10px;width:min(640px,calc(100vw - 260px));display:flex;flex-direction:column;gap:6px;align-items:flex-start}
#left .pn{position:static}
#bar{display:flex;flex-wrap:wrap;gap:4px;align-items:center}
#bar button{background:#20242d;color:__FG__;border:1px solid #2f3542;border-radius:4px;padding:3px 8px;font:12px system-ui;cursor:pointer}
#bar button.on{background:#3a4a66;border-color:#5a72a0}#bar .sep{width:1px;height:18px;background:#2f3542;margin:0 3px}
#cmd{width:340px;padding:4px}#cmd input{width:100%;box-sizing:border-box;background:#0e1015;color:__FG__;border:1px solid #2f3542;border-radius:4px;padding:4px 6px;font:12px ui-monospace,monospace}
#hint{color:#9a9a9a;font-size:11.5px;max-width:420px}
#leg{right:10px;top:10px;max-height:44vh;overflow:auto;width:220px}#leg div{white-space:nowrap;font-size:12px;cursor:pointer;user-select:none;overflow:hidden;text-overflow:ellipsis}
#leg div.off{opacity:.35}#leg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
#obj{right:10px;bottom:10px;max-height:40vh;overflow:auto;width:220px;font-size:12px}#obj:empty{display:none}
#obj div{display:flex;gap:5px;align-items:center;white-space:nowrap}#obj span{flex:1;overflow:hidden;text-overflow:ellipsis;cursor:text}
#obj input[type=color]{width:16px;height:16px;border:0;padding:0;background:none}#obj b{cursor:pointer;color:#9a9a9a}
#out{left:50%;bottom:10px;transform:translateX(-50%);font:12px ui-monospace,monospace;white-space:pre;display:none}
#ax{position:fixed;left:12px;bottom:10px;color:#9a9a9a;font-size:11.5px;pointer-events:none}
#tip{position:fixed;pointer-events:none;display:none;max-width:380px;font-size:12px}</style></head><body>
<canvas id="c"></canvas>
<div id="left"><div id="bar" class="pn"></div>
<div id="cmd" class="pn"><input id="q" list="names" placeholder="name · x,y,z · a > b · plane a,b,c = d" autocomplete="off"><datalist id="names"></datalist></div>
<div id="hint"></div></div><div id="leg" class="pn"></div><div id="obj" class="pn"></div><div id="out" class="pn"></div>
<div id="ax"></div><div id="tip" class="pn"></div><input id="file" type="file" accept=".json" style="display:none">
<script>
const D=__DATA__;const $=i=>document.getElementById(i);const cv=$('c'),g=cv.getContext('2d'),tip=$('tip');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));const hidden=new Set(),layer={build:1,locator:1,centre:1,star:1,origin:1,field:1,vector:1};
const all=D.points.map(o=>o.p).concat(D.vectors.map(v=>v.b));let m=[0,0,0],r=1e-9,lo=[0,0,0],hi=[0,0,0];
for(let k=0;k<3;k++){let a=Infinity,b=-Infinity;for(const p of all){a=Math.min(a,p[k]);b=Math.max(b,p[k])}m[k]=(a+b)/2;r=Math.max(r,(b-a)/2);lo[k]=a;hi[k]=b}
const nq=p=>p.map((x,k)=>(x-m[k])/r),dq=q=>q.map((x,k)=>x*r+m[k]);
lo=nq(lo).map(x=>x-0.08);hi=nq(hi).map(x=>x+0.08);
const P=D.points.map(o=>Object.assign({q:nq(o.p)},o));
const H=D.hulls.map(h=>{const v=h.v.map(nq);return{r:h.r,c:h.c,v,f:h.f,e:h.e,n:h.f.map(f=>fnorm(v[f[0]],v[f[1]],v[f[2]]))}});
function fnorm(a,b,c){const u=[0,1,2].map(k=>b[k]-a[k]),w=[0,1,2].map(k=>c[k]-a[k]);return[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]]}const V=D.vectors.map(v=>({a:v.a,b:v.b,c:v.c,t:v.t}));
const AT=nq(D.at),O0=D.origin||[0,0,0];const EL0=D.el,AZ0=D.az+Math.PI/2;
let az=AZ0,el=EL0,zoom=1,pan=[0,0],drag=null,hover=-1,mode='view',picks=[],pen=null;
const PAL=['#ffd23f','#5ec8ff','#ff6b8b','#8ce99a','#c3a6ff','#ffa94d','#63e6be','#f783ac'];
let U=[],undo=[],nid=1;const KEY='tri:'+D.title+':'+D.points.length;
const sub=(a,b)=>a.map((x,k)=>x-b[k]),dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2],nrm=a=>Math.sqrt(dot(a,a));
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const f3=a=>a.map(x=>(+x).toFixed(3)).join(', ');
const RGB={};const hx=c=>RGB[c]||(RGB[c]=(()=>{let h=String(c).replace('#','');if(h.length===3)h=h.split('').map(x=>x+x).join('');const n=parseInt(h.slice(0,6),16)||0;return[n>>16&255,n>>8&255,n&255]})());
const mix=(c,t)=>{const f=x=>Math.round(t>=0?x+(255-x)*t:x*(1+t));return 'rgb('+hx(c).map(f).join(',')+')'};
const dz=z=>Math.max(0,Math.min(1,(z+1.7)/3.4)),LT=(()=>{const l=[-0.35,0.55,-0.75],n=Math.hypot(...l);return l.map(x=>x/n)})();
function size(){cv.width=innerWidth*devicePixelRatio;cv.height=innerHeight*devicePixelRatio}addEventListener('resize',()=>{size();draw()});size();
function rot(){return[Math.cos(az),Math.sin(az),Math.cos(el),Math.sin(el)]}
function scale(){return Math.min(cv.width,cv.height)*0.42*zoom}
function proj(q){const[ca,sa,ce,se]=rot();const x=ca*q[0]-sa*q[1],y0=sa*q[0]+ca*q[1];const y=ce*q[2]-se*y0,z=se*q[2]+ce*y0;
 const s=scale(),dp=devicePixelRatio;return[cv.width/2+pan[0]*dp+x*s,cv.height/2+pan[1]*dp-y*s,z]}
function unproj(X,Y,z){const[ca,sa,ce,se]=rot(),s=scale(),dp=devicePixelRatio;const x=(X*dp-cv.width/2-pan[0]*dp)/s,y=-(Y*dp-cv.height/2-pan[1]*dp)/s;
 const q2=ce*y+se*z,y0=-se*y+ce*z;return[ca*x+sa*y0,-sa*x+ca*y0,q2]}
const pd=p=>proj(nq(p));
function rv(q){const[ca,sa,ce,se]=rot();const x=ca*q[0]-sa*q[1],y0=sa*q[0]+ca*q[1];return[x,ce*q[2]-se*y0,se*q[2]+ce*y0]}
const FL=(()=>{if(!D.vectors.length)return null;const E=D.vectors.flatMap(v=>[nq(v.a),nq(v.b)]);let a=Infinity,s=0;
 for(const p of E){a=Math.min(a,p[2]);for(const q of E)s=Math.max(s,Math.abs(p[0]-q[0]),Math.abs(p[1]-q[1]),Math.abs(p[2]-q[2]))}return Math.max(lo[2],a-0.25*s)})();
function shadow(a,b,c){const dp=devicePixelRatio,qa=nq(a),qb=nq(b);const fz=FL!=null?Math.min(FL,qa[2],qb[2]):lo[2],fa=[qa[0],qa[1],fz],fb=[qb[0],qb[1],fz];
 const A=proj(fa),B=proj(fb),T=proj(qb);g.save();g.globalAlpha=0.22;g.strokeStyle='#000';g.lineWidth=3*dp;g.lineCap='round';line(A,B);
 g.globalAlpha=0.45;g.strokeStyle=c;g.lineWidth=dp;g.setLineDash([2*dp,3*dp]);line(T,B);g.restore()}
const vis=o=>layer[o.g]&&!(o.r!=null&&hidden.has(o.r));
function line(a,b){g.beginPath();g.moveTo(a[0],a[1]);g.lineTo(b[0],b[1]);g.stroke()}
function arrow(A,B,c,w){const dp=devicePixelRatio;g.strokeStyle=c;g.fillStyle=c;g.lineWidth=w*dp;line(A,B);
 const t=Math.atan2(B[1]-A[1],B[0]-A[0]),h=8*dp;g.beginPath();g.moveTo(B[0],B[1]);g.lineTo(B[0]-h*Math.cos(t-0.4),B[1]-h*Math.sin(t-0.4));g.lineTo(B[0]-h*Math.cos(t+0.4),B[1]-h*Math.sin(t+0.4));g.closePath();g.fill()}
function label(x,y,t,c){if(!t)return;const dp=devicePixelRatio;g.globalAlpha=1;g.fillStyle=c||'#c8c8c8';g.fillText(t,x+5*dp,y-5*dp)}
function planePoly(n,d){const Q=nq,N=[n[0]*r,n[1]*r,n[2]*r],dd=d-dot(n,m),pts=[];const B=[lo,hi];
 for(let a=0;a<3;a++){const b=(a+1)%3,c=(a+2)%3;for(const i of[0,1])for(const j of[0,1]){const p=[0,0,0];p[b]=B[i][b];p[c]=B[j][c];
  if(Math.abs(N[a])<1e-12)continue;const t=(dd-N[b]*p[b]-N[c]*p[c])/N[a];if(t>=lo[a]-1e-9&&t<=hi[a]+1e-9){p[a]=t;pts.push(p)}}}
 if(pts.length<3)return[];const C=[0,1,2].map(k=>pts.reduce((s,p)=>s+p[k],0)/pts.length);let u=sub(pts[0],C);if(nrm(u)<1e-12)u=sub(pts[1],C);
 const w=cross(N,u);return pts.map(p=>[p,Math.atan2(dot(sub(p,C),w)/nrm(w),dot(sub(p,C),u)/nrm(u))]).sort((a,b)=>a[1]-b[1]).map(x=>x[0])}
function tipOf(o){return o.k==='vector'?o.b:o.p}
function draw(){const dp=devicePixelRatio;g.clearRect(0,0,cv.width,cv.height);
 g.lineWidth=dp;g.font=`${11*dp}px system-ui`;
 for(let k=0;k<3;k++){const a=AT.slice(),b=AT.slice();a[k]=-1.05;b[k]=1.05;const A=proj(a),B=proj(b);g.globalAlpha=0.55;g.strokeStyle=D.axc[k];line(A,B);g.globalAlpha=0.8;g.fillStyle=D.axc[k];g.fillText(D.axes[k],B[0]+3*dp,B[1])}
 g.globalAlpha=1;
 for(const o of U)if(o.on&&o.k==='plane'){const W=planePoly(o.n,o.d).map(proj);if(W.length<3)continue;g.globalAlpha=0.16;g.fillStyle=o.c;g.beginPath();W.forEach((w,i)=>i?g.lineTo(w[0],w[1]):g.moveTo(w[0],w[1]));g.closePath();g.fill();
  g.globalAlpha=0.8;g.strokeStyle=o.c;g.lineWidth=1.2*dp;g.stroke();label(W[0][0],W[0][1],o.t,o.c)}
 const fields=D.fields&&layer.field;
 if(fields){const T=[];for(const h of H){if(hidden.has(h.r))continue;const W=h.v.map(proj);h.f.forEach((f,j)=>{const n=rv(h.n[j]),l=Math.abs(dot(n,LT))/(nrm(n)||1);T.push([(W[f[0]][2]+W[f[1]][2]+W[f[2]][2])/3,W,f,h.c,l])})}
  T.sort((a,b)=>b[0]-a[0]);for(const[z,W,f,c,l]of T){g.globalAlpha=(0.1+0.16*l)*(1-0.35*dz(z));g.fillStyle=mix(c,0.45*l-0.25);g.beginPath();g.moveTo(W[f[0]][0],W[f[0]][1]);g.lineTo(W[f[1]][0],W[f[1]][1]);g.lineTo(W[f[2]][0],W[f[2]][1]);g.closePath();g.fill()}
  g.globalAlpha=0.9;g.lineWidth=1.2*dp;for(const h of H){if(hidden.has(h.r))continue;const W=h.v.map(proj);g.strokeStyle=h.c;for(const[a,b]of h.e)line(W[a],W[b])}}
 const S=[];P.forEach((o,i)=>{if(vis(o)&&!(fields&&o.g==='build'&&i!==hover&&!o.hl))S.push([proj(o.q),i])});S.sort((a,b)=>b[0][2]-a[0][2]);
 for(const[[x,y,z],i]of S){const o=P[i];const big=o.g!=='build',t=dz(z);const rad=o.s*dp*(1.15-0.35*t);g.globalAlpha=(big?0.95:0.7)*(1-0.45*t);
  const gr=g.createRadialGradient(x-rad*0.35,y-rad*0.35,rad*0.1,x,y,rad*1.1);gr.addColorStop(0,mix(o.c,0.55));gr.addColorStop(0.55,o.c);gr.addColorStop(1,mix(o.c,-0.45));g.fillStyle=gr;g.beginPath();
  if(o.g==='locator'){g.moveTo(x,y-rad);g.lineTo(x+rad,y);g.lineTo(x,y+rad);g.lineTo(x-rad,y);g.closePath()}
  else if(o.g==='star'){for(let k=0;k<10;k++){const t=-Math.PI/2+k*Math.PI/5,q=k%2?rad*0.45:rad*1.2;k?g.lineTo(x+q*Math.cos(t),y+q*Math.sin(t)):g.moveTo(x+q*Math.cos(t),y+q*Math.sin(t))}g.closePath()}
  else g.arc(x,y,rad,0,7);g.fill();
  if(big||i===hover){g.globalAlpha=1;g.strokeStyle=i===hover?'#ffffff':'#202020';g.lineWidth=dp;g.stroke()}
  if(o.t){g.globalAlpha=0.85;g.fillStyle='#9a9a9a';g.fillText(o.t,x+rad+3*dp,y+4*dp)}}
 if(layer.vector)for(const v of V)shadow(v.a,v.b,v.c);
 g.globalAlpha=0.95;if(layer.vector)for(const v of V){const A=pd(v.a),B=pd(v.b);arrow(A,B,v.c,1.6*(1.2-0.4*dz(B[2])));label(B[0],B[1],v.t,v.c)}
 for(const o of U){if(!o.on)continue;g.globalAlpha=1;
  if(o.k==='vector'){shadow(o.a,o.b,o.c);g.globalAlpha=1;const A=pd(o.a),B=pd(o.b);arrow(A,B,o.c,1.8);label(B[0],B[1],(o.t?o.t+' ':'')+'|v| '+nrm(sub(o.b,o.a)).toFixed(3),o.c)}
  else if(o.k==='point'){const[x,y]=pd(o.p);g.fillStyle=o.c;g.beginPath();g.arc(x,y,5*dp,0,7);g.fill();g.strokeStyle='#000';g.lineWidth=dp;g.stroke();label(x,y,o.t,o.c)}
  else if(o.k==='note'){const[x,y]=pd(o.p);g.fillStyle=o.c;g.beginPath();g.arc(x,y,2.5*dp,0,7);g.fill();g.font=`${12.5*dp}px system-ui`;
   const w=g.measureText(o.t).width;g.globalAlpha=0.82;g.fillStyle='#15181f';g.fillRect(x+6*dp,y-20*dp,w+8*dp,17*dp);g.globalAlpha=1;g.fillStyle=o.c;g.fillText(o.t,x+10*dp,y-7*dp);g.font=`${11*dp}px system-ui`}
  else if(o.k==='pen'&&o.s.length>1){g.strokeStyle=o.c;g.lineWidth=2*dp;g.lineJoin='round';g.beginPath();o.s.forEach((p,i)=>{const[x,y]=pd(p);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}}
 if(pen&&pen.length>1){g.strokeStyle=PAL[(nid-1)%PAL.length];g.lineWidth=2*dp;g.beginPath();pen.forEach((p,i)=>{const[x,y]=pd(p);i?g.lineTo(x,y):g.moveTo(x,y)});g.stroke()}
 for(const p of picks){const[x,y]=pd(p.p);g.strokeStyle='#ffffff';g.lineWidth=1.5*dp;g.beginPath();g.arc(x,y,8*dp,0,7);g.stroke()}
 g.globalAlpha=1;$('ax').innerHTML=D.axes.map((a,i)=>`<span style="color:${D.axc[i]}">&#9632;</span> ${esc(a)}`).join('<br>')}
function near(ex,ey){const dp=devicePixelRatio,X=ex*dp,Y=ey*dp,R=(10*dp)**2;let best=null,bd=R;
 const see=(p,o)=>{const[x,y]=pd(p);const d=(x-X)**2+(y-Y)**2;if(d<bd){bd=d;best=o}};
 P.forEach((o,i)=>{if(vis(o)&&!(D.fields&&layer.field&&o.g==='build'))see(o.p,{p:o.p,t:o.n,i})});
 if(layer.vector)V.forEach(v=>see(v.b,{p:v.b,t:v.t,v:[v.a,v.b]}));
 for(const o of U)if(o.on&&o.k!=='pen'&&o.k!=='plane')see(tipOf(o),{p:tipOf(o),t:o.t,v:o.k==='vector'?[o.a,o.b]:null,u:o.id});return best}
function at(ex,ey){return near(ex,ey)||{p:dq(unproj(ex,ey,0)),t:''}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(U))}catch(e){}}
function push(o){undo.push(JSON.stringify(U));o.id=nid++;o.on=true;o.c=o.c||PAL[(o.id-1)%PAL.length];U.push(o);save();list();draw();return o}
function edit(f){undo.push(JSON.stringify(U));f();save();list();draw()}
function list(){$('obj').innerHTML=U.map(o=>`<div data-id="${o.id}"><input type="checkbox" ${o.on?'checked':''}><input type="color" value="${o.c}"><span title="${esc(desc(o))}">${esc(o.k+(o.t?' · '+o.t:''))}</span><b>&times;</b></div>`).join('');
 for(const d of $('obj').children){const o=U.find(x=>x.id==d.dataset.id);const[cb,cl]=d.querySelectorAll('input');
  cb.onchange=()=>edit(()=>o.on=cb.checked);cl.oninput=()=>{o.c=cl.value;save();draw()};
  d.querySelector('span').ondblclick=()=>{const t=prompt('label',o.t||'');if(t!==null)edit(()=>o.t=t)};
  d.querySelector('span').onclick=()=>show(desc(o));d.querySelector('b').onclick=()=>edit(()=>U.splice(U.indexOf(o),1))}}
function desc(o){if(o.k==='vector'){const v=sub(o.b,o.a);return `vector ${o.t||''}\n tail (${f3(o.a)})\n head (${f3(o.b)})\n v    (${f3(v)})\n |v|  ${nrm(v).toFixed(3)}`}
 if(o.k==='plane'){const[x,y,z]=o.n.map(v=>(+v).toFixed(3));return `plane ${o.t||''}\n ${x} x + ${y} y + ${z} z = ${(+o.d).toFixed(3)}`}
 if(o.k==='pen')return `stroke, ${o.s.length} points`;return `${o.k} ${o.t||''}\n (${f3(o.p)})`}
function show(t){const e=$('out');e.textContent=t;e.style.display=t?'block':'none'}
function measure(a,b){if(a.v&&b.v){const u=sub(a.v[1],a.v[0]),w=sub(b.v[1],b.v[0]),c=dot(u,w)/(nrm(u)*nrm(w)||1e-12);
  return `${a.t||'u'} · ${b.t||'w'}\n |u| ${nrm(u).toFixed(3)}   |w| ${nrm(w).toFixed(3)}\n cos ${c.toFixed(3)}   angle ${(Math.acos(Math.max(-1,Math.min(1,c)))*180/Math.PI).toFixed(1)}°   orthogonality ${Math.sqrt(Math.max(0,1-c*c)).toFixed(3)}\n proj ${a.t||'u'} on ${b.t||'w'} ${(dot(u,w)/(nrm(w)||1e-12)).toFixed(3)}   proj ${b.t||'w'} on ${a.t||'u'} ${(dot(u,w)/(nrm(u)||1e-12)).toFixed(3)}`}
 return `${a.t||'a'} to ${b.t||'b'}\n distance ${nrm(sub(a.p,b.p)).toFixed(3)}\n difference (${f3(sub(b.p,a.p))})`}
const HINT={view:'drag: turn · right/shift-drag: pan · wheel: zoom · double-click: reset',note:'click a point or empty space, then type the note',
 pen:'drag to draw; strokes lie in the screen plane through the centre and turn with the plot',point:'click to place a point (snaps to nearby points)',
 vector:'click the tail, then the head (snaps to points; click the origin for an origin vector)',plane:'click three points to put a plane through them',
 measure:'click two vectors (their heads) for angle, orthogonality and projection lengths, or two points for distance'};
function setMode(k){mode=k;picks=[];pen=null;for(const b of $('bar').querySelectorAll('button[data-m]'))b.classList.toggle('on',b.dataset.m===k);$('hint').textContent=HINT[k];cv.style.cursor=k==='view'?'grab':'crosshair';draw()}
function click(e){const h=at(e.clientX,e.clientY);
 if(mode==='note'){const t=prompt('note');if(t)push({k:'note',p:h.p,t})}
 else if(mode==='point')push({k:'point',p:h.p,t:h.t||''});
 else if(mode==='vector'){picks.push(h);if(picks.length===2){push({k:'vector',a:picks[0].p,b:picks[1].p,t:picks[1].t||''});picks=[]}draw()}
 else if(mode==='plane'){picks.push(h);if(picks.length===3){const[a,b,c]=picks.map(x=>x.p),n=cross(sub(b,a),sub(c,a));picks=[];
  if(nrm(n)<1e-12)show('the three points are collinear');else push({k:'plane',n,d:dot(n,a),t:''})}draw()}
 else if(mode==='measure'){const n=near(e.clientX,e.clientY);if(!n)return;picks.push(n);if(picks.length===2){show(measure(picks[0],picks[1]));picks=[]}draw()}}
function vec3(s){const v=s.split(/[ ,]+/).filter(Boolean).map(Number);return v.length===3&&v.every(isFinite)?v:null}
function ref(s){s=s.trim();if(!s||s==='0'||s==='origin')return{p:O0,t:'origin'};const v=vec3(s);if(v)return{p:v,t:''};const l=s.toLowerCase();
 const o=P.find(o=>String(o.n).toLowerCase()===l)||P.find(o=>String(o.n).toLowerCase().includes(l));if(o)return{p:o.p,t:o.n,i:P.indexOf(o)};
 const u=U.find(o=>(o.t||'').toLowerCase()===l&&o.k!=='pen'&&o.k!=='plane');return u?{p:tipOf(u),t:u.t}:null}
function run(s){s=s.trim();if(!s)return;let m;
 if(m=s.match(/^plane\s+([xyz])\s*=\s*(\S+)$/i)){const n=[0,0,0];n['xyz'.indexOf(m[1].toLowerCase())]=1;return push({k:'plane',n,d:+m[2],t:s.slice(6)})}
 if(m=s.match(/^plane\s+(.+?)\s*=\s*(\S+)$/i)){const n=vec3(m[1]);if(!n||!isFinite(+m[2]))return show('plane a, b, c = d');return push({k:'plane',n,d:+m[2],t:''})}
 if(m=s.match(/^note\s+(.+?)\s*@\s*(.+)$/i)){const a=ref(m[2]);if(!a)return show('unknown place: '+m[2]);return push({k:'note',p:a.p,t:m[1]})}
 if(s.includes('>')){const[x,y]=s.split('>');const a=x.trim()?ref(x):{p:O0,t:'origin'},b=ref(y);if(!a||!b)return show('unknown: '+(!a?x:y).trim());return push({k:'vector',a:a.p,b:b.p,t:b.t||''})}
 const v=vec3(s);if(v)return push({k:'point',p:v,t:''});const a=ref(s);if(!a)return show('no point named '+s);
 if(D.origin)return push({k:'vector',a:O0,b:a.p,t:a.t});push({k:'point',p:a.p,t:a.t})}
const BTN=[['view','View','v'],['note','Note','n'],['pen','Pen','p'],['point','Point','o'],['vector','Vector','a'],['plane','Plane','l'],['measure','Measure','m']];
const ACT=[['Undo',()=>{if(undo.length){U=JSON.parse(undo.pop());save();list();draw()}}],['Clear',()=>{if(U.length)edit(()=>U=[])}],null,
 ['Iso',()=>setView(AZ0,EL0)],['Top',()=>setView(Math.PI/2,Math.PI/2)],['Front',()=>setView(Math.PI/2,0)],['Side',()=>setView(0,0)],null,
 ['Save',()=>dl(new Blob([JSON.stringify({title:D.title,axes:D.axes,objects:U},null,1)],{type:'application/json'}),'annotations.json')],
 ['Load',()=>$('file').click()],['PNG',()=>{draw();cv.toBlob(b=>dl(b,'plot.png'))}]];
function setView(a,e){az=a;el=e;pan=[0,0];zoom=1;draw()}
function dl(b,n){const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=n;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
$('bar').innerHTML=BTN.map(([k,t,s])=>`<button data-m="${k}" title="${s}">${t}</button>`).join('')+'<span class="sep"></span>'+
 ACT.map((a,i)=>a?`<button data-a="${i}">${a[0]}</button>`:'<span class="sep"></span>').join('')+'<span class="sep"></span>'+
 Object.keys(layer).filter(k=>D.points.some(o=>o.g===k)||(k==='field'&&D.fields)||(k==='vector'&&D.vectors.length)).map(k=>`<button data-l="${k}" class="on">${{build:'points',origin:'origin'}[k]||k+'s'}</button>`).join('');
for(const b of $('bar').querySelectorAll('button')){if(b.dataset.m)b.onclick=()=>setMode(b.dataset.m);else if(b.dataset.a)b.onclick=ACT[+b.dataset.a][1];
 else b.onclick=()=>{layer[b.dataset.l]^=1;b.classList.toggle('on',!!layer[b.dataset.l]);draw()}}
$('file').onchange=()=>{const f=$('file').files[0];if(!f)return;f.text().then(t=>{const j=JSON.parse(t);edit(()=>{U=(j.objects||j).map(o=>Object.assign(o,{id:nid++}))})}).catch(()=>show('not an annotations file'));$('file').value=''};
$('names').innerHTML=[...new Set(P.filter(o=>o.n&&o.g!=='origin').map(o=>o.n))].slice(0,5000).map(n=>`<option value="${esc(n)}">`).join('');
$('q').onkeydown=e=>{e.stopPropagation();if(e.key==='Enter'){run($('q').value);$('q').value=''}};
function legend(){if(!D.legend.length){$('leg').style.display='none';return}
 $('leg').innerHTML=D.legend.map(([n,c],k)=>`<div data-r="${k}" class="${hidden.has(k)?'off':''}" title="${esc(n)}"><i style="background:${c}"></i>${esc(n)}</div>`).join('');
 for(const d of $('leg').querySelectorAll('div[data-r]'))d.onclick=e=>{const k=+d.dataset.r;if(e.altKey){const solo=hidden.size===D.legend.length-1&&!hidden.has(k);hidden.clear();if(!solo)D.legend.forEach((_,j)=>{if(j!==k)hidden.add(j)})}else hidden.has(k)?hidden.delete(k):hidden.add(k);legend();draw()}}
cv.oncontextmenu=e=>e.preventDefault();
cv.onmousedown=e=>{const p=e.button===2||e.shiftKey;if(mode==='pen'&&!p){pen=[dq(unproj(e.clientX,e.clientY,0))];return}
 drag={x:e.clientX,y:e.clientY,az,el,pan:pan.slice(),p,moved:false}};
addEventListener('mouseup',e=>{if(pen){if(pen.length>1)push({k:'pen',s:pen,t:''});pen=null;draw();return}
 if(drag&&!drag.moved&&e.target===cv&&mode!=='view'&&!drag.p)click(e);drag=null});
cv.onmousemove=e=>{if(pen){pen.push(dq(unproj(e.clientX,e.clientY,0)));draw();return}
 if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)drag.moved=true;if(!drag.moved)return;
  if(drag.p)pan=[drag.pan[0]+dx,drag.pan[1]+dy];else{az=drag.az+dx*0.008;el=Math.max(-1.57,Math.min(1.57,drag.el+dy*0.008))}tip.style.display='none';draw();return}
 const n=near(e.clientX,e.clientY),h=n&&n.i!=null?n.i:-1;if(h!==hover){hover=h;draw()}
 if(n){tip.textContent=(n.t?n.t+'  ':'')+'('+f3(n.p)+')';tip.style.display='block';tip.style.left=e.clientX+14+'px';tip.style.top=e.clientY+10+'px'}else tip.style.display='none'};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(0.3,Math.min(40,zoom*Math.exp(-e.deltaY*0.0015)));draw()};
cv.ondblclick=()=>{if(mode==='view')setView(AZ0,EL0)};
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if((e.ctrlKey||e.metaKey)&&e.key==='z'){ACT[0][1]();e.preventDefault();return}
 if(e.key==='Escape'){picks=[];show('');setMode('view');return}const b=BTN.find(x=>x[2]===e.key);if(b)setMode(b[0])});
let t0=null;cv.addEventListener('touchstart',e=>{const t=e.touches[0];t0=[t.clientX,t.clientY,az,el]},{passive:true});
cv.addEventListener('touchmove',e=>{const t=e.touches[0];az=t0[2]+(t.clientX-t0[0])*0.008;el=Math.max(-1.57,Math.min(1.57,t0[3]+(t.clientY-t0[1])*0.008));draw()},{passive:true});
try{const s=JSON.parse(localStorage.getItem(KEY)||'[]');if(Array.isArray(s)){U=s;nid=1+Math.max(0,...U.map(o=>o.id||0))}}catch(e){}
legend();list();setMode('view');
</script></body></html>"""


def page(sc: Scene, title: str = "plot", axes: Optional[Sequence[str]] = None,
         view: Tuple[float, float] = (22, -58)) -> str:
    """The interactive page for a Scene as an HTML string."""
    data = scene_data(sc, title, axes, view)
    js = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    return (_PAGE.replace("__TITLE__", _html.escape(str(title))).replace("__BG__", BG)
            .replace("__FG__", FG).replace("__DATA__", js))
