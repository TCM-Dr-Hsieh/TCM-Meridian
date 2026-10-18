"""CSS and the small client-side helpers of the voice pop-up (draggable floating panels, transcript auto-scroll).

Ported from TCM-Meridian-mini (`mini/ui/style.py`). The scripts of the remote microphone and of the replay window are separate
files in `voice/web/` (served as static files, see `register_voice_web`).
"""
from __future__ import annotations

from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent.parent / 'voice' / 'web'
REMOTE_JS_URL = '/tcm-remote-capture.js'
REPLAY_JS_URL = '/tcm-voice-replay.js'

CSS = '''
.float-panel {position:fixed; right:24px; top:96px; width:520px; height:min(70vh, 640px); min-width:300px; min-height:160px;
              z-index:3000; display:flex; flex-direction:column; resize:both; overflow:hidden; box-sizing:border-box;
              background:#fff; border:1px solid #b9c9c1; border-radius:12px; box-shadow:0 8px 28px #1b433238;}
.float-panel.dragging {user-select:none;}
.float-head {display:flex; align-items:center; gap:6px; padding:5px 8px 5px 12px; background:#eaf1ed; cursor:move;
             touch-action:none; user-select:none; border-bottom:1px solid #d5dfda;}
.float-title {font-weight:700; color:#1b4332; font-size:15px;}
.float-body {flex:1 1 0; min-height:0; overflow:auto; padding:10px 14px;}
.voice-body {flex:1 1 0; min-height:0; display:flex; flex-direction:column; gap:6px; padding:8px 12px;}
.voice-scroll {flex:1 1 0; min-height:0; overflow:auto; line-height:1.75; font-size:14px; overflow-wrap:anywhere;
               border:1px solid #e0e8e3; border-radius:8px; padding:6px 8px; background:#fcfdfc;}
.voice-status {font-size:12px; color:#52665d; line-height:1.5; overflow-wrap:anywhere;}
.voice-status.warn {color:#9a4b00;}
.hint {color:#7a8a83; font-size:12px;}
.seg {margin:2px 0; padding:2px 6px; border-radius:6px;}
.seg-no {color:#7a8a83; font-size:11px; margin-right:6px; font-family:Consolas,monospace;}
.seg-no[data-start] {cursor:pointer; border-radius:4px; padding:0 2px; margin-left:-2px;}
.seg-no[data-start]:hover {color:#1b4332; background:#dcebe3; text-decoration:underline;}
.seg.replaying {background:#dcebe3; box-shadow:inset 3px 0 0 #2d6a4f;}
.who {font-weight:700; margin-right:2px;}
.who.doctor {color:#1b4332;} .who.other {color:#9a5a00;} .who.unknown {color:#7f8b86; font-weight:600;}
.who.background {color:#929e99; font-weight:600;} .bg-text {color:#929e99;}
.seg.unlocked {background:#fff6d6;}
.seg.gap {background:#fde8d4; color:#9a4b00;}
/* The edit box of the record shows no source tags; the reference window does, in light colour. The line under the box lists the tags the save would drop. */
.edit-preview {flex:none; font-size:12px; line-height:1.5; color:#7a8a83; padding:4px 2px 0; overflow-wrap:anywhere;}
.edit-preview.warn {color:#9a4b00;}
.reference-text {white-space:pre-wrap; overflow-wrap:anywhere; line-height:1.9; font-size:14px; user-select:text; cursor:text; margin-top:6px;}
.src-tag {color:#8fa398; font-size:12px; background:#f0f5f2; border-radius:4px; padding:0 2px; margin:0 1px;}
.src-tag.manual {color:#b08a3c; background:#fbf4e3;}
.lc-note {background:#fff1c9; border-radius:3px; padding:0 2px; font-weight:700;}
.lc-note .lc-why {color:#9a6a00; font-size:12px; font-weight:400;}
.ref-label {font-weight:700; color:#1b4332; font-size:13px; margin-top:8px;}
/* The 查閱病歷 window: the visits as a list, the record as the main page's browse box shows it. */
.lookup-dates {gap:2px; margin:6px 0 4px; max-height:30%; min-height:44px; overflow:auto; flex-wrap:nowrap;
               border:1px solid #e0e8e3; border-radius:8px; padding:4px;}
.lookup-date {width:100%; justify-content:flex-start; text-align:left; color:#1b4332; font-size:13px;}
.lookup-date.lookup-chosen {background:#dcebe3; font-weight:700;}
.lookup-date.lookup-open {color:#a7b3ad !important; opacity:1 !important;}
.lookup-heading {font-weight:700; color:#2d6a4f; font-size:14px; margin-top:6px;}
.lookup-text {padding:10px 12px; background:#fafafa; border:1px solid #e8e8e8; border-radius:8px; white-space:pre-wrap;
              overflow-wrap:anywhere; font-size:14px; line-height:1.7; user-select:text; cursor:text; margin-top:4px;}
/* The replay window (voice/web/replay.js) is a float-panel built in the browser. */
.replay-panel {resize:both;}
.replay-body {flex:1 1 0; min-height:0; overflow:auto; padding:10px 14px; display:flex; flex-direction:column; gap:8px; font-size:14px;}
.replay-now {font-family:Consolas,monospace; font-size:12px; color:#52665d; min-height:16px;}
.replay-text {flex:1 1 0; min-height:36px; overflow:auto; line-height:1.7; overflow-wrap:anywhere; user-select:text;
              background:#f5f8f6; border:1px solid #e0e8e3; border-radius:8px; padding:6px 10px;}
.replay-controls {display:flex; align-items:center; gap:8px;}
.replay-bar {flex:1 1 auto; min-width:60px; accent-color:#2d6a4f; cursor:pointer;}
.replay-time {font-family:Consolas,monospace; font-size:12px; color:#52665d; white-space:nowrap;}
.replay-btn {display:inline-flex; align-items:center; justify-content:center; width:32px; height:32px; padding:0; border:0;
             border-radius:50%; background:transparent; color:#1b4332; cursor:pointer;}
.replay-btn:hover {background:#dcebe3;}
.replay-btn .material-icons {font-size:22px;}
.replay-start {display:flex; align-items:center; gap:6px; font-size:13px; color:#52665d;}
.replay-start-input {width:84px; padding:2px 6px; border:1px solid #b9c9c1; border-radius:6px; font-family:Consolas,monospace;}
.replay-go {padding:3px 10px; border:1px solid #2d6a4f; border-radius:6px; background:#fff; color:#2d6a4f; cursor:pointer; font-size:13px;}
.replay-go:hover {background:#2d6a4f; color:#fff;}
.replay-msg {color:#b3261e; font-size:13px; display:none;}
.replay-note {color:#7a8a83; font-size:12px;}
'''

JS = '''
window.tcmFollow = true;
document.addEventListener('scroll', function (event) {
  const el = event.target;
  if (el && el.dataset && el.dataset.tcmScroll === '1') {
    window.tcmFollow = (el.scrollHeight - el.scrollTop - el.clientHeight) < 40;
  }
}, true);
window.tcmScrollToEnd = function (id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.dataset.tcmScroll = '1';
  if (window.tcmFollow) el.scrollTop = el.scrollHeight;
};
// Floating panels (class float-panel, header float-head): no backdrop, so the page behind stays usable. The header drags
// the panel (mouse or touch), the bottom-right corner resizes it (CSS), a double click on the header puts it back, and the
// position and size are remembered in this browser. A panel can never be dragged out of reach: its left and top edges stay
// inside the window, the whole header stays above the bottom edge (HEAD_H is only the fallback while the header is not
// laid out), and at least MIN_VISIBLE px of the title end stay inside on the right while dragging. When the panel opens
// and when the browser window shrinks, it is sized to the window and pulled fully into view, on both axes, where it fits.
window.tcmFloat = (function () {
  const KEY = 'tcmFloat:', MIN_VISIBLE = 80, HEAD_H = 40, DEFAULT_W = 520;
  let drag = null;
  function headOf(el) { const head = el.querySelector('.float-head'); return (head && head.offsetHeight) || HEAD_H; }
  function place(el, left, top) {
    // Left edge never past the window's left edge: the buttons sit at the right end of the header and are not draggable,
    // so a panel pushed left would leave nothing to grab. On the right, the title end (the draggable part) stays in.
    const x = Math.min(Math.max(left, 0), window.innerWidth - MIN_VISIBLE);
    const y = Math.min(Math.max(top, 0), window.innerHeight - headOf(el));
    el.style.left = x + 'px';
    el.style.top = y + 'px';
    el.style.right = 'auto';
  }
  function save(el) {
    try {
      const r = el.getBoundingClientRect();
      localStorage.setItem(KEY + el.id, JSON.stringify({left: r.left, top: r.top, width: r.width, height: r.height}));
    } catch (e) { /* blocked storage: show() reads this page's own place and size back from the element; a reload or a double click on the header resets */ }
  }
  function load(id) {
    try { return JSON.parse(localStorage.getItem(KEY + id) || 'null'); } catch (e) { return null; }
  }
  function visible(el) { return el.getClientRects().length > 0; }
  function show(id) {
    const el = document.getElementById(id);
    if (!el) return;
    let s = load(id);
    if (!(s && s.width > 0) && el.style.left) {    // storage blocked: keep where this page had it, but still fit it below
      s = {left: parseFloat(el.style.left), top: parseFloat(el.style.top),
           width: parseFloat(el.style.width) || DEFAULT_W, height: parseFloat(el.style.height) || 400};
    }
    if (!(s && s.width > 0)) {
      const defaultW = parseFloat(el.dataset.defaultW) || DEFAULT_W;
      s = {width: defaultW,
           height: parseFloat(el.dataset.defaultH) || Math.min(window.innerHeight * 0.7, 640),
           left: el.dataset.defaultLeft !== undefined ? parseFloat(el.dataset.defaultLeft)
                 : window.innerWidth - defaultW - (parseFloat(el.dataset.defaultRight) || 24),
           top: 96};
    }
    // sized to the window, then pulled fully into view where it fits (the saved place may come from a bigger window)
    const w = Math.min(s.width, window.innerWidth - 20), h = Math.min(s.height, window.innerHeight - 20);
    el.style.width = w + 'px';
    el.style.height = h + 'px';
    place(el, Math.min(s.left, window.innerWidth - w), Math.min(s.top, window.innerHeight - h));
  }
  function reset(el) {
    try { localStorage.removeItem(KEY + el.id); } catch (e) { /* nothing stored */ }
    el.style.left = el.style.top = el.style.width = el.style.height = '';
    el.style.right = '';
    show(el.id);
  }
  document.addEventListener('pointerdown', function (event) {
    const head = event.target.closest ? event.target.closest('.float-head') : null;
    if (!head || event.target.closest('button') || (event.button !== undefined && event.button !== 0)) return;
    const el = head.closest('.float-panel');
    const r = el.getBoundingClientRect();
    drag = {el: el, dx: event.clientX - r.left, dy: event.clientY - r.top};
    el.classList.add('dragging');
    try { head.setPointerCapture(event.pointerId); } catch (e) { /* the document-level listeners still follow the pointer */ }
    event.preventDefault();
  });
  document.addEventListener('pointermove', function (event) {
    if (drag) place(drag.el, event.clientX - drag.dx, event.clientY - drag.dy);
  });
  function release() {
    if (drag) drag.el.classList.remove('dragging');
    drag = null;
    // a drag or a resize (the CSS corner handle ends on a pointerup too): remember where every open panel is
    document.querySelectorAll('.float-panel').forEach(function (el) { if (visible(el)) save(el); });
  }
  document.addEventListener('pointerup', release);
  document.addEventListener('pointercancel', release);
  document.addEventListener('dblclick', function (event) {
    const head = event.target.closest ? event.target.closest('.float-head') : null;
    if (head && !event.target.closest('button')) reset(head.closest('.float-panel'));
  });
  window.addEventListener('resize', function () {
    document.querySelectorAll('.float-panel').forEach(function (el) {
      if (!visible(el)) return;
      // a smaller window must not leave the header's buttons or the content outside it
      if (el.offsetWidth > window.innerWidth - 20) el.style.width = (window.innerWidth - 20) + 'px';
      if (el.offsetHeight > window.innerHeight - 20) el.style.height = (window.innerHeight - 20) + 'px';
      const r = el.getBoundingClientRect();       // back fully into view, on both axes, where it fits
      place(el, Math.min(r.left, window.innerWidth - el.offsetWidth), Math.min(r.top, window.innerHeight - el.offsetHeight));
    });
  });
  return {show: show, reset: reset};
})();
'''


def register_voice_web(app, ui) -> None:
    """Serve the browser-side scripts of the voice features (once) and put them and the styles in this page. NiceGUI's script mode
    runs the main script again for every browser connection: the files are served once, the head HTML belongs to each page."""
    if not getattr(app, '_tcm_voice_web_served', False):
        app._tcm_voice_web_served = True
        app.add_static_file(local_file=WEB_DIR / 'remote_capture.js', url_path=REMOTE_JS_URL, max_cache_age=0)
        app.add_static_file(local_file=WEB_DIR / 'replay.js', url_path=REPLAY_JS_URL, max_cache_age=0)
    ui.add_head_html(f'<style>{CSS}</style>')
    ui.add_head_html(f'<script>{JS}</script>')
    ui.add_head_html(f'<script src="{REMOTE_JS_URL}"></script>')
    ui.add_head_html(f'<script src="{REPLAY_JS_URL}"></script>')
