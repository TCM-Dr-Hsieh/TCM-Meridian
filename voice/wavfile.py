"""A 16 kHz mono 16-bit WAV file that grows while it is recorded and can be continued later.

`wave.open` can only create a file from scratch. One session here has ONE recording that is stopped and started again (and
survives a restart of the program), so the file is opened for appending: the samples already in it stay, new ones follow, and
the recording clock (`samples / RATE`) simply goes on. The header is patched about once a second, so a crash leaves a playable file.

The sizes in a header that is still being written are not trusted anywhere: the sample count is always taken from the file size.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

RATE = 16000
SAMPLE_BYTES = 2
HEADER_BYTES = 44


class WavError(Exception):
    """The file is not a usable recording (the message is for the physician)."""


def data_offset(handle) -> tuple[int, int]:
    """(offset of the first sample, sample rate) found by walking the RIFF chunks of an open file."""
    handle.seek(0)
    head = handle.read(12)
    if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
        raise WavError('錄音檔還沒有可用的內容，請稍後再試。')
    rate = RATE
    position = 12
    while True:
        handle.seek(position)
        chunk = handle.read(8)
        if len(chunk) < 8:
            raise WavError('錄音檔還沒有可用的內容，請稍後再試。')
        kind, size = chunk[:4], struct.unpack('<I', chunk[4:])[0]
        if kind == b'fmt ':
            body = handle.read(min(size, 16))
            if len(body) < 16:
                raise WavError('錄音檔格式不完整。')
            tag, channels, rate, _, _, bits = struct.unpack('<HHIIHH', body)
            if tag != 1 or channels != 1 or bits != 16:
                raise WavError('錄音檔格式不是 16 位元單聲道，無法使用。')
        elif kind == b'data':
            return position + 8, rate
        position += 8 + size + (size & 1)


def header(rate: int = RATE, data_bytes: int = 0) -> bytes:
    data_bytes = min(max(0, data_bytes), 0xFFFFFFFF - 36)
    return (b'RIFF' + struct.pack('<I', 36 + data_bytes) + b'WAVEfmt ' +
            struct.pack('<IHHIIHH', 16, 1, 1, rate, rate * SAMPLE_BYTES, SAMPLE_BYTES, 16) +
            b'data' + struct.pack('<I', data_bytes))


def sample_count(path: Path) -> int:
    """How many samples the recording at `path` holds (0 when there is no usable file), by the file size."""
    try:
        with open(path, 'rb') as handle:
            offset, _ = data_offset(handle)
            handle.seek(0, 2)
            return max(0, handle.tell() - offset) // SAMPLE_BYTES
    except (OSError, WavError):
        return 0


class WavRecorder:
    """Writes samples to the end of a WAV file; `append=True` continues an existing recording."""

    def __init__(self, path: Path, *, append: bool = False, rate: int = RATE):
        self.path = Path(path)
        self.rate = rate
        self._offset = HEADER_BYTES
        self._bytes = 0
        existing = append and self.path.is_file() and self.path.stat().st_size >= HEADER_BYTES
        if existing:
            try:
                self._file = open(self.path, 'r+b')
                self._offset, found_rate = data_offset(self._file)
            except WavError:
                self._file.close()
                raise
            if found_rate != rate:
                self._file.close()
                raise WavError(f'既有錄音檔的取樣率是 {found_rate} Hz，不是 {rate} Hz，無法續錄。')
            size = self._file.seek(0, 2)
            self._bytes = max(0, size - self._offset)
            if self._bytes % SAMPLE_BYTES:                        # a half sample left by a crash: drop it
                self._bytes -= self._bytes % SAMPLE_BYTES
                self._file.truncate(self._offset + self._bytes)
            self._file.seek(self._offset + self._bytes)
        else:
            self._file = open(self.path, 'wb')
            self._file.write(header(rate))

    @property
    def samples(self) -> int:
        return self._bytes // SAMPLE_BYTES

    def write(self, samples) -> None:
        """Append float samples in [-1, 1] (clipped)."""
        pcm = (np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1, 1) * 32767).astype('<i2').tobytes()
        self._file.write(pcm)
        self._bytes += len(pcm)

    def patch_header(self) -> None:
        """Rewrite the length fields (best effort) so a crash leaves a playable file."""
        try:
            self._file.flush()
            end = self._file.tell()
            data = min(self._bytes, 0xFFFFFFFF - 36)
            self._file.seek(4)
            self._file.write(struct.pack('<I', self._offset - 8 + data))
            self._file.seek(self._offset - 4)
            self._file.write(struct.pack('<I', data))
            self._file.seek(end)
            self._file.flush()
        except Exception:
            pass

    def close(self) -> None:
        if self._file is not None and not self._file.closed:
            self.patch_header()
            self._file.close()

    def __enter__(self) -> 'WavRecorder':
        return self

    def __exit__(self, *exc) -> None:
        self.close()
