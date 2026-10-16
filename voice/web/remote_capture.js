/* Remote microphone: capture this browser's microphone and stream it to the computer running TCM-Meridian.
 *
 * Adapted from voice_to_text (remote_capture.js). Flow, driven by ui_app/controllers/voice_controller.py:
 *   1. begin()   - called straight from the user's click (browsers only grant microphone / audio output to a gesture):
 *                  opens the microphone, creates the AudioContext and the PCM worklet, then tells the page 'prepared'.
 *   2. the page starts the visit on the server and calls connect(token) with the visit's one-time token.
 *   3. connect() - opens the same-origin WebSocket; on 'ready' the microphone is wired into the worklet and numbered
 *                  float32 packets stream out. Unacknowledged packets are kept, so a short outage (or a Wi-Fi hiccup)
 *                  is replayed after reconnecting instead of leaving a hole in the recording.
 *   4. the server sends 'stop' when the visit ends; the tail is flushed and the capture is released.
 * Events for the page go through NiceGUI's emitEvent('tcm_remote', {type, message}). */
const workletSource = `
class TcmPCM extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.length = 0;
    this.finishing = false;
    this.port.onmessage = event => {
      if (event.data === 'finish') {
        this.finishing = true;
        this.flush();
        this.port.postMessage('finished');
      }
    };
  }
  flush() {
    if (this.length) {
      const packet = this.buffer.slice(0, this.length);
      this.port.postMessage(packet, [packet.buffer]);
      this.length = 0;
    }
  }
  process(inputs) {
    if (this.finishing) return true;
    const channels = inputs[0];
    if (channels && channels.length) {
      for (let i = 0; i < channels[0].length; i++) {
        let sample = 0;
        for (const channel of channels) sample += channel[i] / channels.length;
        this.buffer[this.length++] = sample;
        if (this.length === this.buffer.length) this.flush();
      }
    }
    return true;
  }
}
registerProcessor('tcm-pcm', TcmPCM);
`;

const RECONNECT_MS = 30000;
const FIRST_CONNECT_MS = 10000;
const RETRY_DELAY_MS = 700;
const HEARTBEAT_MS = 5000;
const RESPONSE_TIMEOUT_MS = 15000;
const MAX_PENDING_BYTES = 32 * 1024 * 1024;
const MAX_SOCKET_BYTES = 512 * 1024;

function friendlyError(error) {
  const name = error && error.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return '瀏覽器沒有取得麥克風權限。請在網址列允許這個網站使用麥克風後再試一次。';
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') {
    return '找不到可用的麥克風（或所選裝置已拔除）。請確認麥克風已接上後重新整理裝置清單。';
  }
  if (name === 'NotReadableError') {
    return '麥克風正被其他程式使用，或無法開啟。';
  }
  return (error && error.message) || String(error);
}

class TcmRemote {
  constructor() {
    this.deviceId = '';
    this.reset();
  }

  reset() {
    this.stream = null;
    this.context = null;
    this.node = null;
    this.socket = null;
    this.token = null;
    this.sessionId = null;
    this.preparing = false;
    this.prepared = false;
    this.streaming = false;
    this.everReady = false;
    this.serverReady = false;
    this.formatSent = false;
    this.stopping = false;
    this.stopRequested = false;
    this.stopSent = false;
    this.captureError = '';
    this.frames = [];
    this.pendingBytes = 0;
    this.nextSequence = 0;
    this.lastAck = -1;
    this.lastSent = -1;
    this.retryTimer = null;
    this.deadlineTimer = null;
    this.heartbeatTimer = null;
    this.lastServerReply = 0;
    this.reconnectStarted = 0;
    this.flushResolve = null;
  }

  // -- reporting to the page --------------------------------------------------------
  notify(type, message = '', extra = {}) {
    try {
      if (typeof emitEvent === 'function') emitEvent('tcm_remote', {type, message, ...extra});
    } catch (_) { /* the page is gone */ }
    if (type === 'error') console.error('[remote microphone]', message);
  }

  static secure() {
    return !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  }

  // -- devices ------------------------------------------------------------------------
  /** List this computer's microphones. Labels only exist after the user has granted permission, so `prime` asks for
   * it once (the stream is released immediately). Returns [{id, label}]. */
  async listDevices(prime = false) {
    if (!TcmRemote.secure()) return [];
    if (prime) {
      const probe = await navigator.mediaDevices.getUserMedia({audio: true});
      probe.getTracks().forEach(track => track.stop());
    }
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter(d => d.kind === 'audioinput' && d.deviceId)
      .map((d, i) => ({id: d.deviceId, label: d.label || `麥克風 ${i + 1}`}));
  }

  // -- step 1: inside the click ----------------------------------------------------------
  async begin() {
    // `preparing` is set synchronously: a second click while the permission prompt is still open must not start a
    // second capture (two streams, two 'prepared' events, and the loser's abort() would tear down the winner).
    if (this.preparing || this.prepared || this.sessionId) return;
    this.preparing = true;
    try {
      if (!TcmRemote.secure()) {
        throw new Error('瀏覽器只在 HTTPS 網址（例如 Cloudflare）或 localhost 才允許使用麥克風。請改用 https:// 網址開啟，或在執行程式的電腦上直接開啟。');
      }
      const audio = {echoCancellation: false, noiseSuppression: false, autoGainControl: false};
      if (this.deviceId) audio.deviceId = {exact: this.deviceId};
      this.stream = await navigator.mediaDevices.getUserMedia({audio});
      if (!this.stream.getAudioTracks().length) throw new Error('瀏覽器沒有提供音軌。');
      for (const track of this.stream.getTracks()) {
        track.addEventListener('ended', () => {
          if (!this.stopping) this.fail('麥克風已中斷（裝置被拔除或權限被收回）。');
        });
      }
      const context = new AudioContext();            // created inside the gesture, so it may start
      this.context = context;
      await context.resume();
      const url = URL.createObjectURL(new Blob([workletSource], {type: 'text/javascript'}));
      try { await context.audioWorklet.addModule(url); }
      finally { URL.revokeObjectURL(url); }
      const node = new AudioWorkletNode(context, 'tcm-pcm');
      node.port.onmessage = event => {
        if (event.data instanceof Float32Array) this.queueFrame(event.data);
        else if (event.data === 'finished' && this.flushResolve) {
          this.flushResolve();
          this.flushResolve = null;
        }
      };
      this.node = node;
      this.prepared = true;
      const label = this.stream.getAudioTracks()[0].label || '';
      this.notify('prepared', '', {sample_rate: context.sampleRate, label});
    } catch (error) {
      const message = friendlyError(error);
      this.cleanup();
      this.notify('error', `無法使用麥克風：${message}`);
    } finally {
      this.preparing = false;
    }
  }

  // -- step 2: the server has started the visit ---------------------------------------------
  connect(token) {
    if (!this.prepared || this.sessionId) return;
    this.token = token;
    this.sessionId = crypto.randomUUID();
    this.openSocket(false);
  }

  /** The server could not start the visit: release the microphone. */
  abort() {
    this.cleanup();
  }

  openSocket(resume) {
    if (this.socket || !this.sessionId) return;
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/tcm-remote-audio/${this.token}`);
    this.socket = socket;
    this.lastServerReply = Date.now();
    this.startHeartbeat(socket);
    socket.onopen = () => {
      if (this.socket !== socket) return;
      socket.send(JSON.stringify({type: resume && this.everReady ? 'resume' : 'start', session_id: this.sessionId}));
    };
    socket.onmessage = async event => {
      if (this.socket !== socket) return;
      let message;
      try { message = JSON.parse(event.data); }
      catch (_) { this.fail('遠端音訊回應格式錯誤。'); return; }
      this.lastServerReply = Date.now();
      if (message.type === 'ready' || message.type === 'resumed') {
        const next = message.next_sequence;
        if (!Number.isInteger(next) || next < 0 || next > this.nextSequence || next < this.lastAck + 1) {
          this.fail('遠端音訊重連序號不一致。');
          return;
        }
        const first = !this.everReady;
        this.serverReady = this.everReady = true;
        this.formatSent = false;
        this.reconnectStarted = 0;
        clearTimeout(this.retryTimer);
        clearTimeout(this.deadlineTimer);
        this.retryTimer = this.deadlineTimer = null;
        this.ackThrough(next - 1);
        this.lastSent = next - 1;
        this.stopSent = false;
        this.startHeartbeat(socket);
        if (!this.streaming && !this.stopRequested) this.startStreaming();
        this.sendFormat();
        this.sendPending();
        this.maybeFinish();
        if (first) this.notify('recording');
        else this.notify('reconnected');
      } else if (message.type === 'ack') {
        if (!Number.isInteger(message.sequence) || message.sequence >= this.nextSequence) {
          this.fail('遠端音訊確認序號不合法。');
          return;
        }
        this.ackThrough(message.sequence);
        this.sendPending();
        this.maybeFinish();
      } else if (message.type === 'stop') {
        await this.stop();
      } else if (message.type === 'pong') {
        // The server is responsive even if there are no audio ACKs right now.
      } else if (message.type === 'finished' || message.type === 'cancel') {
        this.cleanup();
        this.notify('finished');
      } else if (message.type === 'error') {
        this.fail(message.message || '遠端音訊服務發生錯誤。');
      }
    };
    socket.onclose = () => this.abandonSocket(socket);
    socket.onerror = () => this.abandonSocket(socket);
  }

  startStreaming() {
    this.streaming = true;
    this.context.createMediaStreamSource(this.stream).connect(this.node);
    this.node.connect(this.context.destination);      // the worklet outputs silence, so there is no echo
  }

  // -- connection upkeep --------------------------------------------------------------------
  startHeartbeat(socket) {
    clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = setInterval(() => {
      if (this.socket !== socket) return;
      if (Date.now() - this.lastServerReply >= RESPONSE_TIMEOUT_MS) {
        this.abandonSocket(socket);
        return;
      }
      if (!this.serverReady) return;
      if (socket.readyState === WebSocket.OPEN) {
        try { socket.send(JSON.stringify({type: 'ping'})); }
        catch (_) { this.abandonSocket(socket); }
      } else {
        this.abandonSocket(socket);
      }
    }, HEARTBEAT_MS);
  }

  abandonSocket(socket) {
    if (this.socket !== socket) return;
    this.socket = null;
    this.serverReady = false;
    this.formatSent = false;
    this.stopSent = false;
    clearInterval(this.heartbeatTimer);
    this.heartbeatTimer = null;
    try { socket.close(); } catch (_) { /* replace the stalled socket anyway */ }
    this.scheduleReconnect();
  }

  /** Never connected at all: a proxy that rewrites the Host header, a blocked WebSocket, ... Say so quickly instead
   * of retrying for 30 seconds under a message that talks about a "reconnect" that never happened. */
  connectFailureMessage() {
    return '無法連上主機的遠端音訊通道（WebSocket）。常見原因：代理或 Cloudflare 設定改寫了 Host、網路擋掉 WebSocket。'
      + '看診已經開始但沒有錄到聲音，請按「結束並存檔」後改用本機麥克風，或排除問題後重新開始。';
  }

  scheduleReconnect() {
    if (!this.sessionId) return;
    const limit = this.everReady ? RECONNECT_MS : FIRST_CONNECT_MS;
    const message = this.everReady
      ? '遠端音訊超過 30 秒無法重連；已收到的聲音會保留，之後的聲音沒有錄到。'
      : this.connectFailureMessage();
    if (!this.reconnectStarted) {
      this.reconnectStarted = Date.now();
      if (this.everReady) this.notify('reconnecting');
      this.deadlineTimer = setTimeout(() => this.fail(message), limit);
    }
    if (Date.now() - this.reconnectStarted >= limit) {
      this.fail(message);
      return;
    }
    clearTimeout(this.retryTimer);
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this.openSocket(true);
    }, RETRY_DELAY_MS);
  }

  // -- audio packets --------------------------------------------------------------------------
  queueFrame(samples) {
    if (!this.sessionId) return;
    if (this.nextSequence >= 0xffffffff) {
      this.fail('遠端音訊錄音時間過長，無法繼續傳送。');
      return;
    }
    const packet = new ArrayBuffer(4 + samples.byteLength);
    new DataView(packet).setUint32(0, this.nextSequence, true);
    new Uint8Array(packet, 4).set(new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength));
    this.frames.push({sequence: this.nextSequence++, packet});
    this.pendingBytes += packet.byteLength;
    if (this.pendingBytes > MAX_PENDING_BYTES) {
      this.fail('遠端音訊暫存區已滿；已收到的聲音會保留，之後的聲音沒有錄到。');
      return;
    }
    this.sendPending();
  }

  ackThrough(sequence) {
    if (sequence < this.lastAck) return;
    this.lastAck = sequence;
    while (this.frames.length && this.frames[0].sequence <= sequence) {
      this.pendingBytes -= this.frames.shift().packet.byteLength;
    }
  }

  sendPending() {
    const socket = this.socket;
    if (!this.serverReady || !this.formatSent || socket?.readyState !== WebSocket.OPEN) return;
    for (const frame of this.frames) {
      if (frame.sequence <= this.lastSent) continue;
      if (socket.bufferedAmount >= MAX_SOCKET_BYTES) break;
      try { socket.send(frame.packet); }
      catch (_) { socket.close(); return; }
      this.lastSent = frame.sequence;
    }
  }

  sendFormat() {
    if (this.serverReady && this.socket?.readyState === WebSocket.OPEN && this.context) {
      try {
        this.socket.send(JSON.stringify({type: 'format', sample_rate: this.context.sampleRate}));
        this.formatSent = true;
      } catch (_) { this.socket.close(); }
    }
  }

  // -- stopping ---------------------------------------------------------------------------------
  async stop() {
    if (!this.sessionId || this.stopRequested) return;
    this.stopRequested = this.stopping = true;
    let finished = true;
    if (this.node && this.streaming) {
      finished = await new Promise(resolve => {
        const timer = setTimeout(() => {
          this.flushResolve = null;
          resolve(false);
        }, 5000);
        this.flushResolve = () => {
          clearTimeout(timer);
          resolve(true);
        };
        try { this.node.port.postMessage('finish'); }
        catch (_) {
          clearTimeout(timer);
          this.flushResolve = null;
          resolve(false);
        }
      });
    }
    if (!finished) this.captureError = '瀏覽器無法確認音訊尾段已送出。';
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    this.sendPending();
    this.maybeFinish();
  }

  maybeFinish() {
    if (!this.stopRequested || this.stopSent || this.frames.length ||
        !this.serverReady || this.socket?.readyState !== WebSocket.OPEN) return;
    try {
      this.socket.send(JSON.stringify(this.captureError
        ? {type: 'error', message: this.captureError}
        : {type: 'stop'}));
      this.stopSent = true;
    } catch (_) {
      this.socket.close();
    }
  }

  fail(message) {
    if (!this.sessionId && !this.prepared) return;
    if (this.socket?.readyState === WebSocket.OPEN) {
      try { this.socket.send(JSON.stringify({type: 'error', message})); }
      catch (_) { /* the server's reconnect deadline will close the partial recording */ }
    }
    this.cleanup();
    this.notify('error', message);
  }

  cleanup() {
    clearTimeout(this.retryTimer);
    clearTimeout(this.deadlineTimer);
    clearInterval(this.heartbeatTimer);
    if (this.node) { try { this.node.disconnect(); } catch (_) { /* already gone */ } }
    if (this.context) { try { this.context.close(); } catch (_) { /* already closed */ } }
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    const socket = this.socket;
    this.socket = null;
    if (socket) { try { socket.close(); } catch (_) { /* already closed */ } }
    this.reset();
  }
}

window.tcmRemote = new TcmRemote();
document.documentElement.dataset.tcmRemoteReady = 'true';
