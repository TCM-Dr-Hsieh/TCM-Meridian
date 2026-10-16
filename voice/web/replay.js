// Replay: click `#n mm:ss` in the live transcript to listen again to what was said (see voice/replay.py).
// A floating window (the same kind as 患者資訊: draggable by its header, resizable, no backdrop) plays the clip that
// starts at that audio time and is at most 30 s long, with a time bar, pause / resume and close. It lives outside the
// NiceGUI page tree and is driven entirely here, so it keeps working while the page re-renders the transcript.
// Rows are `<div class="seg" data-idx data-start>` and the click target is `<span class="seg-no" data-idx data-start>`.
window.tcmReplay = (function () {
  const URL_PATH = '/tcm-voice-replay.wav', ID = 'tcm-voice-replay', ROW_REFRESH_MS = 400;
  let panel = null, audio = null, ui = {};
  let state = {start: 0, loaded: false, blobUrl: null, seq: 0, scrubbing: false, timer: null, active: null, token: ''};

  function pad(n) { return n < 10 ? '0' + n : '' + n; }
  function clock(seconds) {                     // the same text as the transcript's own times (voice/textutil.py format_clock)
    const s = Math.max(0, Math.floor(seconds || 0)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return h ? h + ':' + pad(m) + ':' + pad(s % 60) : pad(m) + ':' + pad(s % 60);
  }
  function span(seconds) { const s = Math.max(0, Math.floor(seconds || 0)); return Math.floor(s / 60) + ':' + pad(s % 60); }
  function parseStart(text) {
    const t = String(text).trim();
    let m = t.match(/^(\d+):(\d{1,2})$/);
    if (m) return Number(m[1]) * 60 + Number(m[2]);
    m = t.match(/^(\d+):(\d{1,2}):(\d{1,2})$/);
    if (m) return Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]);
    if (/^\d+(\.\d+)?$/.test(t)) return Number(t);
    return null;
  }
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function iconButton(icon, tip) {
    const b = el('button', 'replay-btn');
    b.type = 'button';
    b.title = tip;
    b.appendChild(el('i', 'material-icons', icon));
    return b;
  }

  function build() {
    panel = el('div', 'float-panel replay-panel');
    panel.id = ID;
    panel.style.display = 'none';
    panel.dataset.defaultW = '400';
    panel.dataset.defaultH = '300';
    const head = el('div', 'float-head');
    head.title = '拖曳標題列移動；連按兩下回到預設位置；右下角可調整大小';
    ui.title = el('span', 'float-title', '回聽');
    const grow = el('div'); grow.style.flex = '1 1 auto';
    ui.close = iconButton('close', '關閉');
    ui.close.setAttribute('data-testid', 'replay-close');
    head.append(ui.title, grow, ui.close);

    const body = el('div', 'replay-body');
    ui.now = el('div', 'replay-now');
    ui.text = el('div', 'replay-text');
    const controls = el('div', 'replay-controls');
    ui.toggle = iconButton('pause', '暫停／繼續');
    ui.toggle.setAttribute('data-testid', 'replay-toggle');
    ui.bar = el('input', 'replay-bar');
    ui.bar.type = 'range'; ui.bar.min = '0'; ui.bar.max = '0'; ui.bar.step = '0.05'; ui.bar.value = '0';
    ui.bar.disabled = true;
    ui.bar.setAttribute('data-testid', 'replay-bar');
    ui.time = el('span', 'replay-time', '0:00 / 0:00');
    controls.append(ui.toggle, ui.bar, ui.time);
    const startRow = el('div', 'replay-start');
    startRow.append(el('span', null, '起點'));
    ui.start = el('input', 'replay-start-input');
    ui.start.type = 'text'; ui.start.size = 8; ui.start.title = '例如 01:20、1:02:03，或秒數；按 Enter 或「從此起點播放」';
    ui.start.setAttribute('data-testid', 'replay-start');
    ui.go = el('button', 'replay-go', '從此起點播放');
    ui.go.type = 'button';
    ui.go.setAttribute('data-testid', 'replay-go');
    startRow.append(ui.start, ui.go);
    ui.msg = el('div', 'replay-msg');
    ui.msg.setAttribute('data-testid', 'replay-msg');
    const note = el('div', 'replay-note', '請使用耳機收聽，以免聲音被麥克風錄進看診。');
    body.append(ui.now, ui.text, controls, startRow, ui.msg, note);
    panel.append(head, body);
    document.body.appendChild(panel);

    audio = new Audio();
    audio.preload = 'auto';
    audio.addEventListener('loadedmetadata', function () {
      ui.bar.max = String(audio.duration || 0);
      ui.bar.disabled = false;
      refresh();
    });
    audio.addEventListener('timeupdate', refresh);
    audio.addEventListener('play', refresh);
    audio.addEventListener('pause', refresh);
    audio.addEventListener('ended', refresh);
    ui.close.addEventListener('click', close);
    ui.toggle.addEventListener('click', function () {
      if (!state.loaded) return;
      if (audio.paused) play(); else audio.pause();
    });
    ui.bar.addEventListener('pointerdown', function () { state.scrubbing = true; });
    ['pointerup', 'pointercancel', 'blur'].forEach(function (name) {
      ui.bar.addEventListener(name, function () { state.scrubbing = false; });
    });
    ui.bar.addEventListener('input', function () {
      if (state.loaded) { audio.currentTime = Number(ui.bar.value); refresh(); }
    });
    function submit() {
      const typed = parseStart(ui.start.value);
      if (typed === null) { message('起點格式不對，請輸入 mm:ss（例如 01:20）或秒數。'); return; }
      // An unedited field still holds the clock text of the exact start: keep the exact (fractional) second then.
      load(ui.start.value.trim() === clock(state.start) ? state.start : typed);
    }
    ui.go.addEventListener('click', submit);
    ui.start.addEventListener('keydown', function (event) { if (event.key === 'Enter') submit(); });
  }

  function message(text) { ui.msg.textContent = text || ''; ui.msg.style.display = text ? '' : 'none'; }
  function play() {
    const p = audio.play();
    if (p && p.catch) p.catch(function () { refresh(); });     // the browser refused to start by itself: the button says play
  }

  function rows() { return Array.prototype.slice.call(document.querySelectorAll('.seg[data-start]')); }
  function rowText(row) {
    const copy = row.cloneNode(true);
    const no = copy.querySelector('.seg-no');
    if (no) no.remove();
    return copy.textContent.trim();
  }
  // The transcript row the playback is in now: the last one that starts at or before the playing time.
  function currentRow() {
    const at = state.start + (audio && state.loaded ? audio.currentTime : 0) + 0.05;
    let found = null;
    rows().forEach(function (row) { if (Number(row.dataset.start) <= at) found = row; });
    return found;
  }
  function highlight(row) {
    document.querySelectorAll('.seg.replaying').forEach(function (r) { if (r !== row) r.classList.remove('replaying'); });
    if (row) row.classList.add('replaying');
  }

  function refresh() {
    if (!panel || panel.style.display === 'none') return;
    // The player belongs to the visit whose row was clicked. If the transcript on screen is now another visit's (or none),
    // its clip must not go on playing, whatever else the page did meanwhile.
    const mark = document.querySelector('.seg-no[data-token]');
    if (!mark || mark.dataset.token !== state.token) { close(); return; }
    const dur = (audio && isFinite(audio.duration)) ? audio.duration : 0, cur = audio ? audio.currentTime : 0;
    if (!state.scrubbing) ui.bar.value = String(cur);
    ui.time.textContent = span(cur) + ' / ' + span(dur);
    const icon = ui.toggle.firstChild;
    icon.textContent = (state.loaded && !audio.paused && !audio.ended) ? 'pause' : 'play_arrow';
    const row = currentRow();
    highlight(row);
    if (row) {
      const no = row.querySelector('.seg-no');
      ui.now.textContent = no ? no.textContent : '';
      ui.text.textContent = rowText(row);
    }
  }

  function release() {
    if (state.blobUrl) { URL.revokeObjectURL(state.blobUrl); state.blobUrl = null; }
  }

  async function load(start) {
    const seq = ++state.seq;
    state.start = start;
    state.loaded = false;
    audio.pause();
    audio.removeAttribute('src');                           // the previous clip is gone: a failed load must not show its position
    audio.load();
    release();
    ui.bar.disabled = true; ui.bar.value = '0'; ui.bar.max = '0';
    ui.start.value = clock(start);
    ui.title.textContent = '回聽 · 從 ' + clock(start) + ' 起';
    message('載入中…');
    refresh();
    try {
      const response = await fetch(URL_PATH + '?start=' + encodeURIComponent(start.toFixed(2)) +
                                   '&token=' + encodeURIComponent(state.token), {cache: 'no-store'});
      if (seq !== state.seq) return;                       // another click came in meanwhile
      if (!response.ok) {
        const reason = await response.text();
        if (seq !== state.seq) return;                     // the error of an earlier click arrived after a newer one took over
        message(reason || '無法取得這段聲音。');
        return;
      }
      const blob = await response.blob();
      if (seq !== state.seq) return;
      release();
      state.blobUrl = URL.createObjectURL(blob);
      audio.src = state.blobUrl;
      state.loaded = true;
      message('');
      play();
    } catch (e) {
      if (seq === state.seq) message('無法取得這段聲音：' + (e && e.message ? e.message : e));
    }
  }

  function open(start, index, token) {
    if (!panel) build();
    const fresh = panel.style.display === 'none';
    panel.style.display = '';
    if (fresh && window.tcmFloat) window.tcmFloat.show(ID);
    if (!state.timer) state.timer = setInterval(refresh, ROW_REFRESH_MS);
    state.active = index;
    state.token = String(token || '');
    load(Math.max(0, Number(start) || 0));
  }

  function close() {
    if (!panel) return;
    state.seq++;                                           // a clip still arriving is dropped
    audio.pause();
    audio.removeAttribute('src');
    audio.load();
    release();
    state.loaded = false;
    if (state.timer) { clearInterval(state.timer); state.timer = null; }
    panel.style.display = 'none';
    highlight(null);
  }

  document.addEventListener('click', function (event) {
    const target = event.target.closest ? event.target.closest('.seg-no[data-start]') : null;
    if (!target) return;
    open(Number(target.dataset.start), Number(target.dataset.idx), target.dataset.token);
  });

  return {open: open, close: close, parseStart: parseStart, clock: clock};
})();
