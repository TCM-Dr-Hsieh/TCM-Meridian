"""The picture viewer of the image tab: the whole picture, zoomed, moved and turned, in a floating window.

A floating window like the voice window (no backdrop, dragged by its header, resized at its corner, put back by a double click on
the header), built once outside the tab panels. It opens on a picture of the image tab's preview and steps through the pictures
that are ticked there (◀ ▶). The zoom, the position and the turn live in the browser (`window.tcmViewer`): the wheel zooms around
the cursor, a drag moves the picture, the buttons zoom in and out, fit the picture to the window, show it at its own size (100 %) and
turn it by 90°. Turning only changes what is shown; the file is never touched. The zoom runs from 10 % to 800 %; a picture opens fitted.
The picture before is cleared as soon as another one is asked for; one that cannot be read (gone from the disk, or failing to load in
the browser) leaves the window empty with a message, never the picture before under its name.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Callable

APP_STATE_KEY = "_image_viewer"

CSS = """
.viewer-tools {display:flex; align-items:center; gap:2px; padding:4px 8px; border-bottom:1px solid #e0e8e3; flex-wrap:wrap;}
.viewer-pct {min-width:52px; text-align:center; font-family:Consolas,monospace; font-size:13px; color:#1b4332;}
.viewer-box {flex:1 1 0; min-height:0; position:relative; overflow:hidden; background:#f2f5f3; cursor:grab; touch-action:none;}
.viewer-box.moving {cursor:grabbing;}
.viewer-img {position:absolute; left:0; top:0; max-width:none; max-height:none; transform-origin:50% 50%; user-select:none;
             pointer-events:none; image-orientation:from-image;}
.viewer-msg {position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#888; font-size:14px;
             pointer-events:none;}
.viewer-msg[hidden] {display:none;}
"""

JS = """
window.tcmViewer = (function () {
  const MIN = 0.1, MAX = 8, STEP = 1.25, views = {};
  function parts(id) {
    const box = document.getElementById(id);
    return box ? {box: box, img: box.querySelector('img.viewer-img')} : null;
  }
  function state(id) { return views[id] || (views[id] = {s: 1, r: 0, cx: 0, cy: 0, w: 0, h: 0, pct: null, bound: false}); }
  function message(p, text) {
    const m = p.box.querySelector('.viewer-msg');
    if (m) { m.textContent = text; m.hidden = !text; }
  }
  function blank(id, text) {
    // nothing on show but the message: the previous picture never stays under another picture's name
    const p = parts(id), v = state(id);
    if (!p || !p.img) return;
    p.img.onload = p.img.onerror = null;
    p.img.removeAttribute('src');
    p.img.style.visibility = 'hidden';
    v.w = v.h = 0;
    message(p, text);
    const pct = v.pct && document.getElementById(v.pct);
    if (pct) pct.textContent = '—';
  }
  function apply(id) {
    const p = parts(id), v = state(id);
    if (!p || !p.img || !v.w) return;
    p.img.style.width = v.w + 'px';
    p.img.style.height = v.h + 'px';
    p.img.style.transform = 'translate(' + (v.cx - v.w / 2) + 'px,' + (v.cy - v.h / 2) + 'px) rotate(' + v.r + 'deg) scale(' + v.s + ')';
    const pct = v.pct && document.getElementById(v.pct);
    if (pct) pct.textContent = Math.round(v.s * 100) + '%';
  }
  function bind(id) {
    const p = parts(id), v = state(id);
    if (!p || v.bound) return;
    v.bound = true;
    p.box.addEventListener('wheel', function (event) {
      event.preventDefault();
      const r = p.box.getBoundingClientRect();
      zoomAt(id, event.deltaY < 0 ? STEP : 1 / STEP, event.clientX - r.left, event.clientY - r.top);
    }, {passive: false});
    let drag = null;
    p.box.addEventListener('pointerdown', function (event) {
      if (event.button !== undefined && event.button !== 0) return;
      drag = {x: event.clientX, y: event.clientY};
      p.box.classList.add('moving');
      try { p.box.setPointerCapture(event.pointerId); } catch (e) { /* the moves still reach the box */ }
      event.preventDefault();
    });
    p.box.addEventListener('pointermove', function (event) {
      if (!drag) return;
      v.cx += event.clientX - drag.x;
      v.cy += event.clientY - drag.y;
      drag = {x: event.clientX, y: event.clientY};
      apply(id);
    });
    function release() { drag = null; p.box.classList.remove('moving'); }
    p.box.addEventListener('pointerup', release);
    p.box.addEventListener('pointercancel', release);
    p.box.addEventListener('dblclick', function () { fit(id); });
  }
  function zoomAt(id, factor, x, y) {
    const v = state(id), s = Math.min(MAX, Math.max(MIN, v.s * factor)), k = s / v.s;
    v.cx = x + (v.cx - x) * k;
    v.cy = y + (v.cy - y) * k;
    v.s = s;
    apply(id);
  }
  function fit(id) {
    const p = parts(id), v = state(id);
    if (!p || !v.w || !v.h) return;
    const turned = v.r % 180 !== 0, bw = turned ? v.h : v.w, bh = turned ? v.w : v.h;
    const W = p.box.clientWidth, H = p.box.clientHeight;
    v.s = Math.min(MAX, Math.max(MIN, Math.min(W / bw, H / bh)));
    v.cx = W / 2;
    v.cy = H / 2;
    apply(id);
  }
  return {
    show: function (id, pctId, src) {
      const p = parts(id), v = state(id);
      if (!p || !p.img) return;
      v.pct = pctId;
      bind(id);
      blank(id, '載入中…');               // the previous picture goes at once, not when the next one has come
      p.img.onload = function () {
        v.w = p.img.naturalWidth; v.h = p.img.naturalHeight; v.r = 0;
        p.img.style.visibility = '';
        message(p, '');
        fit(id);
      };
      p.img.onerror = function () { blank(id, '讀取不到這張圖片'); };
      p.img.src = src;
    },
    blank: function (id, pctId, text) { state(id).pct = pctId; blank(id, text); },
    zoom: function (id, factor) { const p = parts(id); if (p) zoomAt(id, factor, p.box.clientWidth / 2, p.box.clientHeight / 2); },
    fit: fit,
    actual: function (id) { const v = state(id); v.s = 1; apply(id); },
    rotate: function (id, degrees) { const v = state(id); v.r = (v.r + degrees + 360) % 360; fit(id); },
    state: function (id) { return Object.assign({}, state(id)); },
  };
})();
"""

_SERVED: dict[tuple[str, int], str] = {}          # (file, modified time) -> its URL: one route per picture, not one per look


def served_url(add_static_file: Callable[..., str], path: str) -> str:
    """The URL the browser loads the picture from (NiceGUI's static-file route, as `ui.image` uses), made once per file version."""
    real = os.path.realpath(path)
    key = (real, os.stat(real).st_mtime_ns)
    if key not in _SERVED:
        _SERVED[key] = add_static_file(local_file=real, strict=False)
    return _SERVED[key]


class ImageViewerPanel:
    def __init__(self, *, ui: Any, app_state: dict[str, Any], url_for: Callable[[str], str]):
        self.ui = ui
        self.app_state = app_state
        self.url_for = url_for
        self.w = SimpleNamespace()
        self.opened = False
        self.images: list[dict] = []             # [{name, path}]: the pictures ticked in the image tab
        self.index = 0
        app_state[APP_STATE_KEY] = self

    # ------------------------------------------------------------------ building
    def build(self) -> None:
        ui, w = self.ui, self.w
        ui.add_head_html(f"<style>{CSS}</style>")
        ui.add_head_html(f"<script>{JS}</script>")
        with ui.element("div").classes("float-panel") as w.panel:
            w.panel.props("data-default-w=900 data-default-h=680 data-default-left=320")
            with ui.element("div").classes("float-head").tooltip("拖曳標題列移動；連按兩下回到預設位置；右下角可調整大小"):
                w.title = ui.label("圖片檢視").classes("float-title")
                ui.element("div").style("flex:1 1 auto")
                w.previous = ui.button(icon="chevron_left", on_click=lambda: self.step(-1)).props("dense flat round").tooltip("上一張")
                w.position = ui.label("").classes("hint")
                w.following = ui.button(icon="chevron_right", on_click=lambda: self.step(1)).props("dense flat round").tooltip("下一張")
                ui.button(icon="close", on_click=self.close).props("dense flat round").tooltip("關閉").mark("viewer-close")
            with ui.element("div").classes("viewer-tools"):
                self._tool("zoom_out", "縮小", "zoom", 0.8)
                w.pct = ui.label("100%").classes("viewer-pct")
                self._tool("zoom_in", "放大", "zoom", 1.25)
                self._tool("fit_screen", "符合視窗（整張圖）", "fit")
                ui.button("100%", on_click=lambda: self._js("actual")).props("dense flat no-caps").tooltip("原始大小")
                self._tool("rotate_left", "向左轉 90°（只影響顯示）", "rotate", -90)
                self._tool("rotate_right", "向右轉 90°（只影響顯示）", "rotate", 90)
                ui.label("滾輪縮放、拖曳移動、連按兩下符合視窗").classes("hint").style("margin-left:8px")
            with ui.element("div").classes("viewer-box") as w.box:
                ui.element("img").classes("viewer-img").props('alt=""')
                ui.element("div").classes("viewer-msg").props("hidden")
        w.panel.set_visibility(False)

    def _tool(self, icon: str, tip: str, action: str, value: float | int | None = None) -> None:
        self.ui.button(icon=icon, on_click=lambda: self._js(action, value)).props("dense flat round").tooltip(tip)

    def _box_id(self) -> str:
        return f"c{self.w.box.id}"

    def _js(self, action: str, value: float | int | None = None) -> None:
        argument = "" if value is None else f", {value}"
        self.ui.run_javascript(f"window.tcmViewer.{action}('{self._box_id()}'{argument})")

    # ------------------------------------------------------------------ opening, stepping, following the ticks
    def open(self, images: list[dict], index: int) -> None:
        """Show picture `index` of the ticked pictures."""
        if not images:
            return
        self.images = [{"name": item["name"], "path": item["path"]} for item in images]
        self.index = max(0, min(index, len(self.images) - 1))
        self.opened = True
        self.w.panel.set_visibility(True)
        self.ui.run_javascript(f"window.tcmFloat.show('c{self.w.panel.id}')")
        self._show()

    def step(self, delta: int) -> None:
        target = self.index + delta
        if self.opened and 0 <= target < len(self.images):
            self.index = target
            self._show()

    def sync(self, images: list[dict]) -> None:
        """The ticks of the image tab changed (or its list was made again: another patient, a new scan, a picture deleted). The
        picture on show stays if it is still ticked; otherwise the ticked one now at its place in the list (or the last one) is shown,
        and with nothing ticked the window closes."""
        if not self.opened:
            return
        paths = [item["path"] for item in images]
        current = self.images[self.index]["path"] if self.images else None
        if not images:
            self.close()
            return
        self.images = [{"name": item["name"], "path": item["path"]} for item in images]
        if current in paths:
            self.index = paths.index(current)
            self._update_header()
        else:
            self.index = min(self.index, len(self.images) - 1)
            self._show()

    def close(self) -> None:
        self.opened = False
        self.w.panel.set_visibility(False)

    def _update_header(self) -> None:
        item = self.images[self.index]
        self.w.title.set_text(item["name"])
        self.w.position.set_text(f"{self.index + 1} / {len(self.images)}")
        (self.w.previous.enable if self.index > 0 else self.w.previous.disable)()
        (self.w.following.enable if self.index < len(self.images) - 1 else self.w.following.disable)()

    def _show(self) -> None:
        self._update_header()
        path = self.images[self.index]["path"]
        try:
            url = self.url_for(path)
        except OSError:                                    # the file went away between the tick and the look
            self.w.title.set_text(f"{self.images[self.index]['name']}（讀取不到）")
            self.ui.run_javascript(f"window.tcmViewer.blank('{self._box_id()}', 'c{self.w.pct.id}', '讀取不到這張圖片')")
            return
        self.ui.run_javascript(f"window.tcmViewer.show('{self._box_id()}', 'c{self.w.pct.id}', {url!r})")
