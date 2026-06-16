"""Browser-based mask-review app for SAM3_new — no terminal needed.

A biologist opens the image in a browser, sees the green center-plant mask
overlaid on the photo, and cleans it up by clicking unwanted blobs (COMPONENT
mode) or painting over leaks that are physically merged into the center plant
(BRUSH mode). "Keep seed" and "Snap-separate" one-shot helpers are there too.
Saving overwrites the persisted mask in place (`masks/<stem>.mask.npy`), marks
the meta edited, and writes a review preview — exactly like the matplotlib
`mask_review` editor, so all downstream use consumes the corrected mask.

All the mask math runs server-side (reusing the GUI-free functions in
`mask_review.py`); the browser only sends small click/stroke coordinates and
receives small JPEG overlays, so it stays responsive even though the source
photos are ~20 MB. The server binds to 127.0.0.1 only — nothing leaves the
machine.

Run:   python -m SAM3_new.review_app <output_dir>
       (e.g. python -m SAM3_new.review_app output_Testimages)
Then open http://127.0.0.1:7860  (the launcher opens it for you).
"""

from __future__ import annotations

import argparse
import base64
import threading
import webbrowser
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import mask_review as mr

DISPLAY_W = 1200          # working width of the main editor image (px)
THUMB_W = 240             # sidebar thumbnail width (px)
MASK_RGB = np.array([0, 210, 0], np.float32)   # overlay colour
MAX_LOADED = 4            # cap full-res states held in memory (LRU)


def _overlay(disp_rgb: np.ndarray, full_mask: np.ndarray, alpha: float = 0.5) -> bytes:
    """Composite the mask (resized to the display image) over disp_rgb -> JPEG bytes."""
    dh, dw = disp_rgb.shape[:2]
    m = cv2.resize(full_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    vis = disp_rgb.astype(np.float32).copy()
    vis[m] = vis[m] * (1 - alpha) + MASK_RGB * alpha
    bgr = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes()


def create_app(output_dir: Path) -> FastAPI:
    output_dir = Path(output_dir)
    app = FastAPI(title="SAM3_new Mask Review")

    states: "OrderedDict[str, dict]" = OrderedDict()   # stem -> full-res working state (LRU)
    thumbs: dict[str, bytes] = {}                       # stem -> small overlay JPEG cache

    # ----------------------------------------------------------------- state
    def get_state(stem: str) -> dict:
        st = states.get(stem)
        if st is None:
            rgb, mask, meta, _, _ = mr.load(output_dir, stem)
            if rgb is None:
                rgb = np.zeros((*mask.shape, 3), np.uint8)
            scale = DISPLAY_W / rgb.shape[1]
            disp = cv2.resize(rgb, (DISPLAY_W, max(1, round(rgb.shape[0] * scale))))
            st = {
                "rgb": rgb, "disp": disp, "mask": mask.copy(),
                "original": mask.copy(), "meta": meta, "history": [],
            }
            states[stem] = st
            while len(states) > MAX_LOADED:
                states.popitem(last=False)   # evict least-recently loaded
        states.move_to_end(stem)
        return st

    def push_history(st: dict) -> None:
        st["history"].append(st["mask"].copy())
        if len(st["history"]) > 15:
            st["history"].pop(0)

    def status_of(stem: str) -> str:
        meta_path = output_dir / mr.MASKS_SUBDIR / f"{stem}.meta.json"
        if meta_path.is_file():
            import json
            try:
                if json.loads(meta_path.read_text()).get("manually_edited"):
                    return "saved"
            except Exception:
                pass
        return "pending"

    # ----------------------------------------------------------------- pages
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/stems")
    def api_stems():
        out = []
        for stem in mr.list_stems(output_dir):
            conf = None
            mp = output_dir / mr.MASKS_SUBDIR / f"{stem}.meta.json"
            if mp.is_file():
                import json
                try:
                    conf = json.loads(mp.read_text()).get("mask_confidence")
                except Exception:
                    conf = None
            out.append({"stem": stem, "conf": conf, "status": status_of(stem)})
        return JSONResponse(out)

    @app.get("/api/thumbnail/{stem}")
    def api_thumbnail(stem: str):
        buf = thumbs.get(stem)
        if buf is None:
            rgb, mask, _, _, _ = mr.load(output_dir, stem)
            if rgb is None:
                rgb = np.zeros((*mask.shape, 3), np.uint8)
            h = max(1, round(rgb.shape[0] * THUMB_W / rgb.shape[1]))
            disp = cv2.resize(rgb, (THUMB_W, h))
            buf = _overlay(disp, mask)
            thumbs[stem] = buf
        return Response(buf, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})

    @app.get("/api/load/{stem}")
    def api_load(stem: str):
        st = get_state(stem)
        h, w = st["mask"].shape
        meta = st["meta"]
        img = base64.b64encode(_overlay(st["disp"], st["mask"])).decode()
        return JSONResponse({
            "stem": stem,
            "image": "data:image/jpeg;base64," + img,
            "full_w": w, "full_h": h,
            "conf": meta.get("mask_confidence"),
            "area": float(st["mask"].mean()),
            "blobs": int(cv2.connectedComponents(st["mask"].astype(np.uint8))[0] - 1),
            "status": status_of(stem),
        })

    @app.post("/api/edit/{stem}")
    async def api_edit(stem: str, request: Request):
        body = await request.json()
        st = get_state(stem)
        fh, fw = st["mask"].shape
        dw = float(body.get("dw") or DISPLAY_W)
        dh = float(body.get("dh") or st["disp"].shape[0])
        sx, sy = fw / dw, fh / dh
        action = body.get("action")
        push_history(st)
        if action == "component":
            new = mr.remove_component_at(st["mask"], round(body["x"] * sx), round(body["y"] * sy))
            if int(new.sum()) == int(st["mask"].sum()):
                st["history"].pop()            # clicked empty space — no-op, drop undo step
            else:
                st["mask"] = new
        elif action == "brush":
            r = max(1, round(float(body.get("radius", 14)) * sx))
            work = st["mask"].astype(np.uint8)
            for px, py in body.get("points", []):
                cv2.circle(work, (round(px * sx), round(py * sy)), r, 0, -1)
            st["mask"] = work.astype(bool)
        elif action == "keepseed":
            st["mask"] = mr.keep_only_seed(st["mask"], st["meta"].get("seed_xy"))
        elif action == "separate":
            r = max(3, round(float(body.get("radius", 14)) * sx) * 3)
            st["mask"] = mr.open_separate(st["mask"], st["meta"].get("seed_xy"), r)
        else:
            st["history"].pop()
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
        return _edit_response(st)

    @app.post("/api/undo/{stem}")
    def api_undo(stem: str):
        st = get_state(stem)
        if st["history"]:
            st["mask"] = st["history"].pop()
        return _edit_response(st)

    @app.post("/api/reset/{stem}")
    def api_reset(stem: str):
        st = get_state(stem)
        push_history(st)
        st["mask"] = st["original"].copy()
        return _edit_response(st)

    @app.post("/api/save/{stem}")
    def api_save(stem: str):
        st = get_state(stem)
        mr.save(output_dir, stem, st["mask"], st["rgb"])
        thumbs.pop(stem, None)          # refresh sidebar thumbnail next request
        return JSONResponse({"ok": True, "status": "saved", "area": int(st["mask"].sum())})

    def _edit_response(st: dict):
        return Response(
            _overlay(st["disp"], st["mask"]),
            media_type="image/jpeg",
            headers={
                "X-Blobs": str(int(cv2.connectedComponents(st["mask"].astype(np.uint8))[0] - 1)),
                "X-Area": f"{float(st['mask'].mean()):.4f}",
                "Cache-Control": "no-cache",
            },
        )

    return app


# ----------------------------------------------------------------- front-end
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mask Review — SAM3</title>
<style>
  :root { --bg:#0f1115; --panel:#181b22; --line:#2a2f3a; --txt:#e6e9ef; --mut:#9aa3b2;
          --accent:#22c55e; --warn:#f59e0b; --red:#ef4444; --blue:#3b82f6; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--txt); background:var(--bg); height:100vh; display:flex; overflow:hidden; }
  /* sidebar */
  #side { width:300px; flex:0 0 300px; background:var(--panel); border-right:1px solid var(--line);
          display:flex; flex-direction:column; height:100vh; }
  #side h1 { font-size:15px; margin:0; padding:14px 16px; border-bottom:1px solid var(--line); }
  #side h1 small { color:var(--mut); font-weight:400; display:block; font-size:12px; margin-top:3px; }
  #progress { padding:8px 16px; color:var(--mut); font-size:12px; border-bottom:1px solid var(--line); }
  #list { overflow-y:auto; flex:1; padding:6px; }
  .item { display:flex; gap:8px; align-items:center; padding:6px; border-radius:8px; cursor:pointer; }
  .item:hover { background:#20242e; }
  .item.active { background:#283041; outline:1px solid var(--blue); }
  .item img { width:64px; height:42px; object-fit:cover; border-radius:4px; background:#000; flex:0 0 64px; }
  .item .meta { min-width:0; }
  .item .nm { font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .item .sub { font-size:11px; color:var(--mut); }
  .dot { width:9px; height:9px; border-radius:50%; flex:0 0 9px; }
  .dot.saved { background:var(--accent); } .dot.pending { background:var(--red); }
  .dot.editing { background:var(--warn); }
  /* main */
  #main { flex:1; display:flex; flex-direction:column; height:100vh; min-width:0; }
  #bar { padding:10px 16px; border-bottom:1px solid var(--line); display:flex; gap:10px;
         align-items:center; flex-wrap:wrap; background:var(--panel); }
  #bar .hdr { font-weight:600; margin-right:auto; }
  #bar .hdr small { color:var(--mut); font-weight:400; margin-left:8px; }
  button { font:inherit; color:var(--txt); background:#252a35; border:1px solid var(--line);
           padding:7px 12px; border-radius:8px; cursor:pointer; }
  button:hover { background:#2e3543; } button:disabled { opacity:.4; cursor:default; }
  button.on { background:var(--blue); border-color:var(--blue); }
  button.save { background:var(--accent); border-color:var(--accent); color:#06210f; font-weight:600; }
  .sep { width:1px; height:24px; background:var(--line); }
  #brushwrap { display:flex; align-items:center; gap:6px; color:var(--mut); font-size:12px; }
  #stage { flex:1; overflow:auto; display:flex; align-items:flex-start; justify-content:center;
           padding:18px; background:#0b0d11; position:relative; }
  #holder { position:relative; line-height:0; }
  #view { display:block; max-width:100%; height:auto; border-radius:6px; box-shadow:0 6px 30px rgba(0,0,0,.5); }
  #paint { position:absolute; inset:0; cursor:crosshair; touch-action:none; }
  #hint { position:absolute; bottom:10px; left:50%; transform:translateX(-50%); color:var(--mut);
          font-size:12px; background:rgba(0,0,0,.55); padding:5px 12px; border-radius:20px; }
  #empty { color:var(--mut); margin:auto; text-align:center; max-width:420px; }
  #toast { position:fixed; bottom:20px; right:20px; background:var(--accent); color:#06210f;
           padding:10px 16px; border-radius:10px; font-weight:600; opacity:0; transition:opacity .25s;
           pointer-events:none; }
  #toast.show { opacity:1; }
  kbd { background:#252a35; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px;
        padding:1px 6px; font-size:11px; }
</style>
</head>
<body>
  <aside id="side">
    <h1>🌿 Mask Review<small>Click a plant photo, clean the green mask, Save.</small></h1>
    <div id="progress">Loading…</div>
    <div id="list"></div>
  </aside>

  <section id="main">
    <div id="bar">
      <span class="hdr" id="hdr">No image loaded</span>
      <button id="mClick" class="on" title="Click a green blob to delete it (C)">🖱 Click-delete</button>
      <button id="mBrush" title="Drag to erase merged leaks (B)">🖌 Brush</button>
      <span id="brushwrap">size <input id="brush" type="range" min="4" max="60" value="14"/><span id="bval">14</span></span>
      <span class="sep"></span>
      <button id="bKeep" title="Keep only the stem plant (K)">✂ Keep seed</button>
      <button id="bSep" title="Snap-separate thin bridges (O)">⛓ Snap-separate</button>
      <span class="sep"></span>
      <button id="bUndo" title="Undo (U)">↶ Undo</button>
      <button id="bReset" title="Reset to original (R)">⟳ Reset</button>
      <button id="bSave" class="save" title="Save (S)">💾 Save</button>
      <button id="bNext" title="Next image (N)">Next ▸</button>
    </div>
    <div id="stage">
      <div id="empty">Select an image from the left to begin.<br/><br/>
        <b>Click-delete</b>: click a stray green blob to remove it.<br/>
        <b>Brush</b>: drag over leaks that touch the main plant.<br/>
        Then press <b>Save</b> — the cleaned mask overwrites the file.</div>
      <div id="holder" style="display:none">
        <img id="view" alt="mask overlay"/>
        <canvas id="paint"></canvas>
        <div id="hint"></div>
      </div>
    </div>
  </section>
  <div id="toast"></div>

<script>
const $ = s => document.querySelector(s);
let stem=null, mode='click', brush=14, fullW=0, fullH=0;
let drawing=false, pts=[], lastBlob=null, dirty=false;
let headerConf=null, headerArea=0;
const view=$('#view'), paint=$('#paint'), ctx=paint.getContext('2d');

function toast(msg){ const t=$('#toast'); t.textContent=msg; t.classList.add('show');
  clearTimeout(t._t); t._t=setTimeout(()=>t.classList.remove('show'),1600); }

async function loadStems(){
  const r=await fetch('/api/stems'); const d=await r.json();
  window._stems=d.map(x=>x.stem);
  const saved=d.filter(x=>x.status==='saved').length;
  $('#progress').textContent=`${saved} / ${d.length} reviewed`;
  const L=$('#list'); L.innerHTML='';
  for(const it of d){
    const el=document.createElement('div'); el.className='item'; el.dataset.stem=it.stem;
    el.innerHTML=`<span class="dot ${it.status}"></span>
      <img loading="lazy" src="/api/thumbnail/${it.stem}"/>
      <div class="meta"><div class="nm">${it.stem}</div>
      <div class="sub">${it.conf!=null?('conf '+it.conf.toFixed(2)):''}</div></div>`;
    el.onclick=()=>loadStem(it.stem);
    L.appendChild(el);
  }
}

function setActive(s){ document.querySelectorAll('.item').forEach(e=>
  e.classList.toggle('active', e.dataset.stem===s)); }
function setDot(s,cls){ const e=document.querySelector(`.item[data-stem="${s}"] .dot`);
  if(e) e.className='dot '+cls; }

async function loadStem(s){
  stem=s; dirty=false;
  const r=await fetch('/api/load/'+s); const d=await r.json();
  fullW=d.full_w; fullH=d.full_h; lastBlob=d.blobs;
  headerConf=d.conf; headerArea=d.area;
  $('#empty').style.display='none'; $('#holder').style.display='';
  view.src=d.image;
  setActive(s);
  await view.decode().catch(()=>{}); sizeCanvas();
  header(d.conf, d.area, d.blobs, d.status);
}
function header(conf, area, blobs, status){
  $('#hdr').innerHTML=`${stem} <small>${conf!=null?('conf '+conf.toFixed(2)+' · '):''}`+
    `area ${(area*100).toFixed(1)}% · ${blobs} blob${blobs===1?'':'s'}${dirty?' · <b style="color:#f59e0b">unsaved</b>':''}</small>`;
  $('#hint').textContent = mode==='click'
    ? 'Click a green blob to delete it' : 'Drag over the leak to erase  ·  [ ] resize brush';
}
function sizeCanvas(){
  paint.width=view.clientWidth; paint.height=view.clientHeight;
  paint.style.width=view.clientWidth+'px'; paint.style.height=view.clientHeight+'px';
}
window.addEventListener('resize', ()=>{ if(stem) sizeCanvas(); });

function applyJpeg(blob, blobs, area){
  if(view.src.startsWith('blob:')) URL.revokeObjectURL(view.src);
  view.src=URL.createObjectURL(blob);
  dirty=true; if(blobs!=null) lastBlob=blobs;
  header(headerConf, headerArea=(area!=null?area:headerArea), lastBlob, 'editing');
  setDot(stem,'editing');
}

async function edit(payload){
  if(!stem) return;
  payload.dw=paint.width; payload.dh=paint.height;
  const r=await fetch('/api/edit/'+stem,{method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
  const blob=await r.blob();
  applyJpeg(blob, +r.headers.get('X-Blobs'), parseFloat(r.headers.get('X-Area')));
}
async function simple(url){
  if(!stem) return;
  const r=await fetch(url+stem,{method:'POST'});
  const blob=await r.blob();
  applyJpeg(blob, +r.headers.get('X-Blobs'), parseFloat(r.headers.get('X-Area')));
}

// pointer interaction — brush shows a live ring so you can see its size + position
let hoverX=0, hoverY=0, hovering=false;
paint.addEventListener('pointerenter', ()=>{ hovering=true; redraw(); });
paint.addEventListener('pointerleave', ()=>{ if(!drawing){ hovering=false; redraw(); } });
paint.addEventListener('pointerdown', e=>{
  if(!stem) return; paint.setPointerCapture(e.pointerId);
  hoverX=e.offsetX; hoverY=e.offsetY; hovering=true;
  if(mode==='click'){ edit({action:'component', x:e.offsetX, y:e.offsetY}); }
  else { drawing=true; pts=[[e.offsetX,e.offsetY]]; redraw(); }
});
paint.addEventListener('pointermove', e=>{
  hoverX=e.offsetX; hoverY=e.offsetY; hovering=true;
  if(drawing) pts.push([e.offsetX,e.offsetY]);
  redraw();
});
function endStroke(){ if(!drawing) return; drawing=false;
  if(pts.length) edit({action:'brush', points:pts, radius:brush});
  pts=[]; redraw(); }
paint.addEventListener('pointerup', endStroke);
paint.addEventListener('pointercancel', endStroke);
function redraw(){
  ctx.clearRect(0,0,paint.width,paint.height);
  if(mode!=='brush') return;
  if(drawing){                                   // erased area painted so far
    ctx.fillStyle='rgba(255,80,80,0.5)';
    for(const [x,y] of pts){ ctx.beginPath(); ctx.arc(x,y,brush,0,7); ctx.fill(); }
  }
  if(hovering){                                  // brush outline that follows the cursor
    ctx.beginPath(); ctx.arc(hoverX,hoverY,brush,0,7);
    ctx.lineWidth=2; ctx.strokeStyle='rgba(255,255,255,0.95)'; ctx.stroke();
    ctx.beginPath(); ctx.arc(hoverX,hoverY,brush,0,7);
    ctx.lineWidth=1; ctx.strokeStyle='rgba(0,0,0,0.65)'; ctx.stroke();
  }
}

// toolbar
function setMode(m){ mode=m;
  $('#mClick').classList.toggle('on', m==='click');
  $('#mBrush').classList.toggle('on', m==='brush');
  paint.style.cursor='crosshair';
  redraw();
  if(stem) header(headerConf, headerArea, lastBlob, 'editing'); }
$('#mClick').onclick=()=>setMode('click');
$('#mBrush').onclick=()=>setMode('brush');
$('#brush').oninput=e=>{ brush=+e.target.value; $('#bval').textContent=brush; redraw(); };
$('#bKeep').onclick=()=>edit({action:'keepseed'});
$('#bSep').onclick=()=>edit({action:'separate', radius:brush});
$('#bUndo').onclick=()=>simple('/api/undo/');
$('#bReset').onclick=()=>simple('/api/reset/');
$('#bNext').onclick=()=>nextStem();
$('#bSave').onclick=save;

async function save(){
  if(!stem) return;
  const r=await fetch('/api/save/'+stem,{method:'POST'}); const d=await r.json();
  if(d.ok){ dirty=false; setDot(stem,'saved'); toast('Saved ✓');
    const saved=document.querySelectorAll('.dot.saved').length;
    $('#progress').textContent=`${saved} / ${window._stems.length} reviewed`;
    header(headerConf, headerArea, lastBlob, 'saved'); }
}
function nextStem(){ const a=window._stems||[]; const i=a.indexOf(stem);
  if(i>=0 && a.length) loadStem(a[(i+1)%a.length]); }

document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  const k=e.key.toLowerCase();
  if(k==='c') setMode('click');
  else if(k==='b') setMode('brush');
  else if(k==='k') edit({action:'keepseed'});
  else if(k==='o') edit({action:'separate', radius:brush});
  else if(k==='u') simple('/api/undo/');
  else if(k==='r') simple('/api/reset/');
  else if(k==='s'){ e.preventDefault(); save(); }
  else if(k==='n') nextStem();
  else if(k==='['){ brush=Math.max(4,Math.round(brush/1.3)); $('#brush').value=brush; $('#bval').textContent=brush; redraw(); }
  else if(k===']'){ brush=Math.min(60,Math.round(brush*1.3)+1); $('#brush').value=brush; $('#bval').textContent=brush; redraw(); }
});

loadStems();
</script>
</body>
</html>"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="SAM3_new.review_app",
        description="Browser-based mask-review UI for SAM3_new center-plant masks.")
    p.add_argument("output_dir", help="A SAM3_new segment output dir containing masks/.")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (keep 127.0.0.1 — local only).")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    if not (out / mr.MASKS_SUBDIR).is_dir():
        p.error(f"No '{mr.MASKS_SUBDIR}/' folder in {out} — pass a SAM3_new segment output dir.")
    n = len(mr.list_stems(out))

    app = create_app(out)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Mask Review — {n} image(s) from {out}")
    print(f"  Open:  {url}\n  (Ctrl-C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
