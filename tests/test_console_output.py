"""Output sent to a file is written in the Windows code page (cp950): a character it has no place for used to raise inside an agent's
print and fail the agent's call. It is now written as an escape and the program goes on."""
import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ui_app.console_output import tolerate_unencodable_output

ROOT = Path(__file__).resolve().parent.parent
TEXT = "鼻衂 ❌ 頭痛"                       # 「衂」 and the emoji are not in cp950


def test_a_character_the_stream_cannot_encode_is_written_as_an_escape():
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp950")
    with pytest.raises(UnicodeEncodeError):
        print(TEXT, file=stream)
    tolerate_unencodable_output(stream)
    print(TEXT, file=stream)
    stream.flush()
    written = stream.buffer.getvalue().decode("cp950")
    assert "\\u8842" in written and "\\u274c" in written and "鼻" in written and "頭痛" in written
    tolerate_unencodable_output(None, object())                     # no console, or a stream that cannot be reconfigured: left alone


@pytest.mark.parametrize("tolerant", [False, True])
def test_a_program_whose_output_goes_to_a_file_keeps_running(tmp_path, tolerant):
    log = tmp_path / "out.txt"
    script = ("import sys; sys.path.insert(0, %r)\n" % str(ROOT)
              + ("from ui_app.console_output import tolerate_unencodable_output; tolerate_unencodable_output()\n" if tolerant else "")
              + "print(%r)\nprint('still running')\n" % TEXT)
    env = {**os.environ, "PYTHONIOENCODING": "cp950"}              # what a redirect gives on a Traditional Chinese Windows
    with open(log, "wb") as out:
        done = subprocess.run([sys.executable, "-c", script], stdout=out, stderr=subprocess.PIPE, env=env, timeout=60)
    written = log.read_bytes().decode("cp950")
    if tolerant:
        assert done.returncode == 0 and "still running" in written and "\\u8842" in written
    else:                                                           # the old behaviour: the print raises and nothing after it runs
        assert done.returncode != 0 and b"UnicodeEncodeError" in done.stderr and "still running" not in written


def test_the_main_program_makes_its_output_tolerant():
    source = (ROOT / "TCM_Meridian_main.py").read_text(encoding="utf-8")
    assert "from ui_app.console_output import tolerate_unencodable_output" in source
    assert "\ntolerate_unencodable_output()" in source
