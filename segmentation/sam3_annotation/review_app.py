"""Browser-based mask-review / editor for SAM3 center-plant masks — no terminal needed.

A biologist opens an image in a browser, sees the green center-plant mask over the
photo, and edits it:
  - remove  : click a stray blob (COMPONENT), or drag the ERASE brush over a leak
              that is physically merged into the center plant; keep-seed / snap-separate.
  - add     : drag the ADD/PAINT brush to complete a missed region, or use SMART-ADD
              (click / box) to let SAM 3 segment a missed region and union it in.
Precision aids: zoom & pan, an opacity slider, and hold-to-peek the photo underneath.

Saving overwrites the persisted mask in place (`masks/<stem>.mask.npy`) and regenerates
the pipeline's derived images, so all downstream use consumes the corrected mask.

Mask math runs server-side (reusing the GUI-free functions in `mask_review.py`); the
browser sends small click/stroke coordinates and receives a small photo JPEG once plus
a green-on-transparent mask PNG per edit, so it stays responsive on ~20 MB photos. The
SAM "smart add" backend loads lazily on first use only, so erase/paint stays model-free
and fast. The server binds to 127.0.0.1 only — nothing leaves the machine.

Run:   python -m segmentation.sam3_annotation.review_app <output_dir> [--device cpu]
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
from starlette.concurrency import run_in_threadpool

from . import mask_review as mr

DISPLAY_W = 1200          # working width of the main editor image (px)
THUMB_W = 240             # sidebar thumbnail width (px)
MASK_RGB = np.array([0, 210, 0], np.float32)   # thumbnail overlay colour
MASK_BGRA = (0, 210, 0, 255)                   # mask-layer colour (cv2 BGRA)
MAX_LOADED = 4            # cap full-res states held in memory (LRU)


def _overlay(disp_rgb: np.ndarray, full_mask: np.ndarray, alpha: float = 0.5) -> bytes:
    """Composite the mask (resized to the display image) over disp_rgb -> JPEG bytes.
    Used for the sidebar thumbnails (where adjustable opacity isn't needed)."""
    dh, dw = disp_rgb.shape[:2]
    m = cv2.resize(full_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    vis = disp_rgb.astype(np.float32).copy()
    vis[m] = vis[m] * (1 - alpha) + MASK_RGB * alpha
    bgr = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes()


def _photo_jpeg(disp_rgb: np.ndarray) -> bytes:
    """The plain display photo (no mask) -> JPEG bytes; sent once per image."""
    bgr = cv2.cvtColor(disp_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes()


def _mask_png(disp_shape, full_mask: np.ndarray) -> bytes:
    """Green-on-transparent mask layer at display resolution -> PNG bytes; the browser
    stacks it over the photo so opacity / peek are instant and zoom stays crisp."""
    dh, dw = disp_shape[:2]
    m = cv2.resize(full_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST).astype(bool)
    bgra = np.zeros((dh, dw, 4), np.uint8)
    bgra[m] = MASK_BGRA
    ok, buf = cv2.imencode(".png", bgra)
    return buf.tobytes()


def create_app(output_dir: Path, device: str = "cpu") -> FastAPI:
    output_dir = Path(output_dir)
    app = FastAPI(title="SAM3 Mask Review")

    states: "OrderedDict[str, dict]" = OrderedDict()   # stem -> full-res working state (LRU)
    thumbs: dict[str, bytes] = {}                       # stem -> small overlay JPEG cache
    sam_holder: dict = {"sam": None}                    # lazy InteractiveSAM singleton

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
                "sam_state": None,
            }
            states[stem] = st
            while len(states) > MAX_LOADED:
                states.popitem(last=False)   # evict least-recently loaded (drops its sam_state)
        states.move_to_end(stem)
        return st

    def get_sam():
        if sam_holder["sam"] is None:
            from .interactive import InteractiveSAM
            sam_holder["sam"] = InteractiveSAM(device=device)
        return sam_holder["sam"]

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
        dh, dw = st["disp"].shape[:2]
        meta = st["meta"]
        photo = base64.b64encode(_photo_jpeg(st["disp"])).decode()
        maskpng = base64.b64encode(_mask_png(st["disp"].shape, st["mask"])).decode()
        return JSONResponse({
            "stem": stem,
            "photo": "data:image/jpeg;base64," + photo,
            "mask": "data:image/png;base64," + maskpng,
            "full_w": w, "full_h": h, "disp_w": dw, "disp_h": dh,
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
        dw = float(body.get("dw") or st["disp"].shape[1])
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
        elif action in ("brush", "paint"):
            r = max(1, round(float(body.get("radius", 14)) * sx))
            work = st["mask"].astype(np.uint8)
            fill = 1 if action == "paint" else 0
            for px, py in body.get("points", []):
                cv2.circle(work, (round(px * sx), round(py * sy)), r, fill, -1)
            st["mask"] = work.astype(bool)
        elif action == "keepseed":
            st["mask"] = mr.keep_only_seed(st["mask"], st["meta"].get("seed_xy"))
        elif action == "separate":
            r = max(3, round(float(body.get("radius", 14)) * sx) * 3)
            st["mask"] = mr.open_separate(st["mask"], st["meta"].get("seed_xy"), r)
        elif action == "fillholes":
            m = st["mask"].astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(m)
            cv2.drawContours(filled, cnts, -1, 1, -1)
            st["mask"] = filled.astype(bool)
        else:
            st["history"].pop()
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
        return _edit_response(st)

    @app.post("/api/smartadd/{stem}")
    async def api_smartadd(stem: str, request: Request):
        body = await request.json()
        st = get_state(stem)
        try:
            added = await run_in_threadpool(_smartadd_compute, st, body)
        except Exception as e:  # model build / inference failure
            return JSONResponse({"error": f"Smart-add failed: {e}"}, status_code=500)
        if added is not None and added.shape == st["mask"].shape and added.any():
            push_history(st)
            st["mask"] = st["mask"] | added
        return _edit_response(st)

    def _smartadd_compute(st: dict, body: dict):
        fh, fw = st["mask"].shape
        dw = float(body.get("dw") or st["disp"].shape[1])
        dh = float(body.get("dh") or st["disp"].shape[0])
        sx, sy = fw / dw, fh / dh
        sam = get_sam()
        if st.get("sam_state") is None:
            st["sam_state"] = sam.set_image(st["rgb"])
        box = body.get("box")
        if box:
            x0, x1 = sorted((float(box[0]) * sx, float(box[2]) * sx))
            y0, y1 = sorted((float(box[1]) * sy, float(box[3]) * sy))
            if (x1 - x0) < 4 or (y1 - y0) < 4:   # tiny drag — treat as a click
                return sam.add_from_point(st["sam_state"], (x0 + x1) / 2, (y0 + y1) / 2)
            return sam.add_from_box(st["sam_state"], [x0, y0, x1, y1])
        return sam.add_from_point(st["sam_state"], float(body["x"]) * sx, float(body["y"]) * sy)

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
            _mask_png(st["disp"].shape, st["mask"]),
            media_type="image/png",
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
          --accent:#22c55e; --warn:#f59e0b; --red:#ef4444; --blue:#3b82f6; --cyan:#06b6d4; }
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
  #bar { padding:8px 14px; border-bottom:1px solid var(--line); display:flex; gap:7px;
         align-items:center; flex-wrap:wrap; background:var(--panel); }
  #bar .hdr { font-weight:600; margin-right:auto; }
  #bar .hdr small { color:var(--mut); font-weight:400; margin-left:8px; }
  button { font:inherit; color:var(--txt); background:#252a35; border:1px solid var(--line);
           padding:6px 10px; border-radius:8px; cursor:pointer; }
  button:hover { background:#2e3543; } button:disabled { opacity:.4; cursor:default; }
  button.on { background:var(--blue); border-color:var(--blue); }
  button.add.on { background:var(--accent); border-color:var(--accent); color:#06210f; }
  button.smart.on { background:var(--cyan); border-color:var(--cyan); color:#04222a; }
  button.save { background:var(--accent); border-color:var(--accent); color:#06210f; font-weight:600; }
  .sep { width:1px; height:22px; background:var(--line); }
  .ctl { display:flex; align-items:center; gap:6px; color:var(--mut); font-size:12px; }
  .ctl input[type=range] { width:84px; }
  #zlabel { min-width:38px; text-align:center; }
  #stage { flex:1; overflow:auto; display:flex; align-items:flex-start; justify-content:center;
           padding:18px; background:#0b0d11; position:relative; }
  #holder { position:relative; line-height:0; margin:auto; }
  #photo, #mask { display:block; width:100%; height:auto; border-radius:6px; }
  #holder img { position:absolute; inset:0; }
  #photo { position:relative; box-shadow:0 6px 30px rgba(0,0,0,.5); }
  #mask { pointer-events:none; }
  #paint { position:absolute; inset:0; cursor:crosshair; touch-action:none; width:100%; height:auto; }
  #empty { color:var(--mut); margin:auto; text-align:center; max-width:440px; }
  #hint { position:fixed; bottom:14px; left:320px; color:var(--mut); font-size:12px;
          background:rgba(0,0,0,.55); padding:5px 12px; border-radius:20px; pointer-events:none; }
  #busy { position:fixed; inset:0; display:none; align-items:center; justify-content:center;
          background:rgba(8,10,14,.55); z-index:50; }
  #busy.show { display:flex; }
  #busy .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
                padding:18px 24px; display:flex; gap:14px; align-items:center; }
  .spin { width:20px; height:20px; border:3px solid #3a4252; border-top-color:var(--cyan);
          border-radius:50%; animation:sp 0.8s linear infinite; }
  @keyframes sp { to { transform:rotate(360deg); } }
  #toast { position:fixed; bottom:20px; right:20px; background:var(--accent); color:#06210f;
           padding:10px 16px; border-radius:10px; font-weight:600; opacity:0; transition:opacity .25s;
           pointer-events:none; max-width:340px; }
  #toast.show { opacity:1; } #toast.err { background:var(--red); color:#fff; }
</style>
</head>
<body>
  <aside id="side">
    <h1>🌿 Mask Review<small>Click a photo, clean or complete the green mask, Save.</small></h1>
    <div id="progress">Loading…</div>
    <div id="list"></div>
  </aside>

  <section id="main">
    <div id="bar">
      <span class="hdr" id="hdr">No image loaded</span>
      <button id="mClick" class="on" title="Click a green blob to delete it (C)">🖱 Click-delete</button>
      <button id="mErase" title="Drag to erase merged leaks (E)">🩹 Erase</button>
      <button id="mPaint" class="add" title="Drag to paint a missed region (A)">🖊 Add</button>
      <button id="mSmart" class="smart" title="Click/box a missed region — SAM segments it (G)">✨ Smart-add</button>
      <span class="ctl">size <input id="brush" type="range" min="4" max="80" value="14"/><span id="bval">14</span></span>
      <span class="sep"></span>
      <span class="ctl">opacity <input id="op" type="range" min="10" max="100" value="60"/></span>
      <button id="bPeek" title="Hold to see the photo underneath (Space)">👁 Peek</button>
      <span class="sep"></span>
      <span class="ctl"><button id="zOut" title="Zoom out (-)">➖</button><span id="zlabel">Fit</span>
        <button id="zIn" title="Zoom in (+)">➕</button><button id="zFit" title="Fit (0)">⤢</button></span>
      <span class="sep"></span>
      <button id="bKeep" title="Keep only the stem plant (K)">✂ Keep seed</button>
      <button id="bSep" title="Snap-separate thin bridges (O)">⛓ Separate</button>
      <button id="bFill" title="Fill interior holes (F)">⬚ Fill holes</button>
      <span class="sep"></span>
      <button id="bUndo" title="Undo (U)">↶ Undo</button>
      <button id="bReset" title="Reset to original (R)">⟳ Reset</button>
      <button id="bSave" class="save" title="Save (S)">💾 Save</button>
      <button id="bNext" title="Next image (N)">Next ▸</button>
    </div>
    <div id="stage">
      <div id="empty">Select an image from the left to begin.<br/><br/>
        <b>Click-delete</b>: remove a stray green blob.<br/>
        <b>Erase</b> / <b>Add</b>: drag to erase a leak or paint a missed region.<br/>
        <b>Smart-add</b>: click or box a missed region and SAM segments it.<br/>
        Then press <b>Save</b> — the corrected mask overwrites the file.</div>
      <div id="holder" style="display:none">
        <img id="photo" alt="plant"/>
        <img id="mask" alt="mask"/>
        <canvas id="paint"></canvas>
      </div>
    </div>
    <div id="hint"></div>
  </section>

  <div id="busy"><div class="card"><div class="spin"></div><div id="busytxt">Segmenting…</div></div></div>
  <div id="toast"></div>

<script>
const $ = s => document.querySelector(s);
let stem=null, mode='click', brush=14, fullW=0, fullH=0, dispW=0, dispH=0;
let zoom=1, fitZoom=1;
let drawing=false, pts=[], boxStart=null, boxNow=null, lastBlob=null, dirty=false, peeking=false;
let headerConf=null, headerArea=0;
const stage=$('#stage'), holder=$('#holder'), photo=$('#photo'), maskImg=$('#mask'),
      paint=$('#paint'), ctx=paint.getContext('2d');

function toast(msg, err){ const t=$('#toast'); t.textContent=msg; t.classList.toggle('err',!!err);
  t.classList.add('show'); clearTimeout(t._t); t._t=setTimeout(()=>t.classList.remove('show'),2200); }
function busy(on,txt){ if(txt) $('#busytxt').textContent=txt; $('#busy').classList.toggle('show',on); }

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
  fullW=d.full_w; fullH=d.full_h; dispW=d.disp_w; dispH=d.disp_h; lastBlob=d.blobs;
  headerConf=d.conf; headerArea=d.area;
  $('#empty').style.display='none'; holder.style.display='';
  photo.src=d.photo; setMaskSrc(d.mask);
  setActive(s);
  await photo.decode().catch(()=>{});
  fitView(); header(d.conf, d.area, d.blobs, d.status);
}
function setMaskSrc(src){
  if(maskImg.src.startsWith('blob:')) URL.revokeObjectURL(maskImg.src);
  maskImg.src=src;
}
function header(conf, area, blobs, status){
  $('#hdr').innerHTML=`${stem} <small>${conf!=null?('conf '+conf.toFixed(2)+' · '):''}`+
    `area ${(area*100).toFixed(1)}% · ${blobs} blob${blobs===1?'':'s'}${dirty?' · <b style="color:#f59e0b">unsaved</b>':''}</small>`;
  const hints={click:'Click a green blob to delete it',
    erase:'Drag to erase  ·  [ ] brush size  ·  hold Space to peek',
    paint:'Drag to paint a missed region  ·  [ ] brush size',
    smart:'Click a missed region, or drag a box — SAM segments it'};
  $('#hint').textContent=hints[mode]||'';
}

/* ---- zoom / pan : holder width = dispW * zoom; canvas internal stays dispW x dispH ---- */
function applyZoom(){
  const W=dispW*zoom, H=dispH*zoom;
  // The canvas is a *replaced* element: with auto-height holder, a CSS height:100%
  // collapses to 0 (it has no definite basis), so the canvas would cover nothing and
  // swallow no pointer events. Pin holder AND canvas to explicit px so all three
  // layers (photo/mask/paint) line up and the canvas actually receives pointer input.
  holder.style.width=W+'px'; holder.style.height=H+'px';
  paint.style.width=W+'px'; paint.style.height=H+'px';
  paint.width=dispW; paint.height=dispH;     // fixed internal res (constant memory)
  $('#zlabel').textContent = Math.abs(zoom-fitZoom)<1e-3 ? 'Fit' : Math.round(zoom*100)+'%';
  redraw();
}
function fitView(){
  const availW=stage.clientWidth-36, availH=stage.clientHeight-36;
  fitZoom=Math.min(availW/dispW, availH/dispH, 1);
  zoom=fitZoom; applyZoom();
}
function setZoom(z, cx, cy){
  z=Math.max(fitZoom*0.5, Math.min(8, z));
  const prevW=dispW*zoom;
  const sl=stage.scrollLeft, st_=stage.scrollTop;
  const px=(cx==null? stage.clientWidth/2:cx), py=(cy==null? stage.clientHeight/2:cy);
  const fx=(sl+px)/Math.max(1,prevW), fy=(st_+py)/Math.max(1,dispH*zoom);
  zoom=z; applyZoom();
  const newW=dispW*zoom, newH=dispH*zoom;
  stage.scrollLeft=fx*newW-px; stage.scrollTop=fy*newH-py;
}
$('#zIn').onclick=()=>setZoom(zoom*1.25);
$('#zOut').onclick=()=>setZoom(zoom/1.25);
$('#zFit').onclick=()=>fitView();
stage.addEventListener('wheel', e=>{ if(e.ctrlKey||e.metaKey){ e.preventDefault();
  setZoom(zoom*(e.deltaY<0?1.12:0.89), e.clientX-stage.getBoundingClientRect().left,
          e.clientY-stage.getBoundingClientRect().top); } }, {passive:false});
window.addEventListener('resize', ()=>{ if(stem) applyZoom(); });

/* ---- pointer -> canvas-internal coords (independent of zoom) ---- */
function pos(e){ const r=paint.width/paint.clientWidth; return [e.offsetX*r, e.offsetY*r]; }
let hover=null;
paint.addEventListener('pointerenter', e=>{ hover=pos(e); redraw(); });
paint.addEventListener('pointerleave', ()=>{ if(!drawing){ hover=null; redraw(); } });
paint.addEventListener('pointerdown', e=>{
  if(!stem) return;
  if(e.button===1){ startPan(e); return; }     // middle-drag = pan
  paint.setPointerCapture(e.pointerId);
  const p=pos(e); hover=p;
  if(mode==='click'){ edit({action:'component', x:p[0], y:p[1]}); }
  else if(mode==='smart'){ boxStart=p; boxNow=p; }
  else { drawing=true; pts=[p]; }
  redraw();
});
paint.addEventListener('pointermove', e=>{
  const p=pos(e); hover=p;
  if(drawing) pts.push(p);
  else if(boxStart) boxNow=p;
  redraw();
});
function endStroke(e){
  if(drawing){ drawing=false;
    if(pts.length) edit({action: mode==='paint'?'paint':'brush', points:pts, radius:brush});
    pts=[]; }
  else if(boxStart){
    const a=boxStart, b=boxNow||boxStart;
    const dx=Math.abs(a[0]-b[0]), dy=Math.abs(a[1]-b[1]);
    if(dx<5 && dy<5) smartAdd({x:a[0], y:a[1]});
    else smartAdd({box:[a[0],a[1],b[0],b[1]]});
    boxStart=boxNow=null;
  }
  redraw();
}
paint.addEventListener('pointerup', endStroke);
paint.addEventListener('pointercancel', endStroke);

function dp(n){ return n*paint.width/Math.max(1,paint.clientWidth); }   // screen px -> buffer px
function ring(p, color){ ctx.beginPath(); ctx.arc(p[0],p[1],brush,0,7);
  ctx.lineWidth=dp(2.5); ctx.strokeStyle=color; ctx.stroke();
  ctx.beginPath(); ctx.arc(p[0],p[1],brush,0,7);
  ctx.lineWidth=dp(1); ctx.strokeStyle='rgba(0,0,0,0.7)'; ctx.stroke(); }
function redraw(){
  ctx.clearRect(0,0,paint.width,paint.height);
  if(!stem) return;
  if(mode==='erase' || mode==='paint'){
    const col = mode==='paint' ? 'rgba(40,220,90,0.45)' : 'rgba(255,80,80,0.5)';
    if(drawing){ ctx.fillStyle=col;
      for(const p of pts){ ctx.beginPath(); ctx.arc(p[0],p[1],brush,0,7); ctx.fill(); } }
    if(hover) ring(hover, mode==='paint'?'rgba(120,255,160,0.98)':'rgba(255,255,255,0.98)');
  } else if(mode==='smart'){
    ctx.lineWidth=dp(2.5); ctx.strokeStyle='rgba(6,182,212,0.98)';
    if(boxStart && boxNow){ const x=Math.min(boxStart[0],boxNow[0]), y=Math.min(boxStart[1],boxNow[1]);
      ctx.strokeRect(x,y,Math.abs(boxNow[0]-boxStart[0]),Math.abs(boxNow[1]-boxStart[1])); }
    else if(hover){ const rr=dp(9), arm=dp(15);
      ctx.beginPath(); ctx.arc(hover[0],hover[1],rr,0,7); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(hover[0]-arm,hover[1]); ctx.lineTo(hover[0]+arm,hover[1]);
      ctx.moveTo(hover[0],hover[1]-arm); ctx.lineTo(hover[0],hover[1]+arm); ctx.stroke(); }
  }
}

/* ---- middle-button drag pan ---- */
let panFrom=null;
function startPan(e){ panFrom=[e.clientX,e.clientY,stage.scrollLeft,stage.scrollTop];
  const mv=ev=>{ if(!panFrom) return; stage.scrollLeft=panFrom[2]-(ev.clientX-panFrom[0]);
    stage.scrollTop=panFrom[3]-(ev.clientY-panFrom[1]); };
  const up=()=>{ panFrom=null; window.removeEventListener('pointermove',mv);
    window.removeEventListener('pointerup',up); };
  window.addEventListener('pointermove',mv); window.addEventListener('pointerup',up);
}

/* ---- server edits ---- */
function applyMask(blob, blobs, area){
  setMaskSrc(URL.createObjectURL(blob));
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
  applyMask(blob, +r.headers.get('X-Blobs'), parseFloat(r.headers.get('X-Area')));
}
async function smartAdd(payload){
  if(!stem) return;
  payload.dw=paint.width; payload.dh=paint.height;
  busy(true, 'Segmenting…  (first time also loads the SAM model)');
  try{
    const r=await fetch('/api/smartadd/'+stem,{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    if(!r.ok){ const e=await r.json().catch(()=>({})); toast(e.error||'Smart-add failed', true); return; }
    const blob=await r.blob();
    applyMask(blob, +r.headers.get('X-Blobs'), parseFloat(r.headers.get('X-Area')));
  } catch(err){ toast('Smart-add failed', true); }
  finally{ busy(false); }
}
async function simple(url){
  if(!stem) return;
  const r=await fetch(url+stem,{method:'POST'});
  const blob=await r.blob();
  applyMask(blob, +r.headers.get('X-Blobs'), parseFloat(r.headers.get('X-Area')));
}

/* ---- toolbar ---- */
function setMode(m){ mode=m;
  $('#mClick').classList.toggle('on', m==='click');
  $('#mErase').classList.toggle('on', m==='erase');
  $('#mPaint').classList.toggle('on', m==='paint');
  $('#mSmart').classList.toggle('on', m==='smart');
  redraw(); if(stem) header(headerConf, headerArea, lastBlob, 'editing'); }
$('#mClick').onclick=()=>setMode('click');
$('#mErase').onclick=()=>setMode('erase');
$('#mPaint').onclick=()=>setMode('paint');
$('#mSmart').onclick=()=>setMode('smart');
$('#brush').oninput=e=>{ brush=+e.target.value; $('#bval').textContent=brush; redraw(); };
$('#op').oninput=e=>{ maskImg.style.opacity=e.target.value/100; };
maskImg.style.opacity=0.6;
$('#bKeep').onclick=()=>edit({action:'keepseed'});
$('#bSep').onclick=()=>edit({action:'separate', radius:brush});
$('#bFill').onclick=()=>edit({action:'fillholes'});
$('#bUndo').onclick=()=>simple('/api/undo/');
$('#bReset').onclick=()=>simple('/api/reset/');
$('#bNext').onclick=()=>nextStem();
$('#bSave').onclick=save;
function setPeek(on){ peeking=on; maskImg.style.visibility=on?'hidden':'visible'; }
$('#bPeek').addEventListener('pointerdown', ()=>setPeek(true));
window.addEventListener('pointerup', ()=>{ if(peeking) setPeek(false); });

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
function setBrush(v){ brush=Math.max(4,Math.min(80,v)); $('#brush').value=brush; $('#bval').textContent=brush; redraw(); }

document.addEventListener('keydown', e=>{
  if(e.target.tagName==='INPUT') return;
  if(e.code==='Space'){ if(!peeking) setPeek(true); e.preventDefault(); return; }
  const k=e.key.toLowerCase();
  if(k==='c') setMode('click');
  else if(k==='e'||k==='b') setMode('erase');
  else if(k==='a') setMode('paint');
  else if(k==='g') setMode('smart');
  else if(k==='k') edit({action:'keepseed'});
  else if(k==='o') edit({action:'separate', radius:brush});
  else if(k==='f') edit({action:'fillholes'});
  else if(k==='u') simple('/api/undo/');
  else if(k==='r') simple('/api/reset/');
  else if(k==='s'){ e.preventDefault(); save(); }
  else if(k==='n') nextStem();
  else if(k==='[') setBrush(Math.round(brush/1.3));
  else if(k===']') setBrush(Math.round(brush*1.3)+1);
  else if(k==='+'||k==='=') setZoom(zoom*1.25);
  else if(k==='-') setZoom(zoom/1.25);
  else if(k==='0') fitView();
});
document.addEventListener('keyup', e=>{ if(e.code==='Space' && peeking) setPeek(false); });

loadStems();
</script>
</body>
</html>"""


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="segmentation.sam3_annotation.review_app",
        description="Browser-based mask-review / editor for SAM3 center-plant masks.")
    p.add_argument("output_dir", help="A `segment` output dir containing masks/.")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1", help="Bind address (keep 127.0.0.1 — local only).")
    p.add_argument("--device", default="cpu",
                   help="Device for SAM smart-add (cpu/cuda/mps/auto). Default cpu — most stable on Mac.")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    if not (out / mr.MASKS_SUBDIR).is_dir():
        p.error(f"No '{mr.MASKS_SUBDIR}/' folder in {out} — pass a `segment` output dir.")
    n = len(mr.list_stems(out))

    app = create_app(out, device=args.device)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Mask Review — {n} image(s) from {out}")
    print(f"  Open:  {url}\n  (smart-add uses device={args.device}; Ctrl-C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
