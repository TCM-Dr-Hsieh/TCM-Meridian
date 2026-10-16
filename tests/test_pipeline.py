import asyncio
import json
import wave

from voice.llm import LLMCaller, LLMClient
from voice.asr import ASRError
from voice.corrector import TranscriptCorrector
from voice.pipeline import TranscriptPipeline
from tests.helpers import FakeASR, FakeLLM, make_settings, silent_source


def make_pipeline(tmp_path, seconds, asr, corrector=None, wav=False):
    settings = make_settings(tmp_path)
    path = tmp_path / 'audio.wav' if wav else None
    source = silent_source(settings, seconds, recording_path=path)
    events = []
    pipeline = TranscriptPipeline(asr=asr, asr_settings=settings.asr, corrector=corrector, source=source,
                                  emit=lambda type, **f: events.append((type, f)))
    return pipeline, source, events


async def run(pipeline, source):
    source.start()
    await asyncio.wait_for(pipeline.run(), 20)
    await source.close()


async def test_windows_become_numbered_segments_with_time_axis(tmp_path):
    asr = FakeASR(default='我頭痛三天了')
    pipeline, source, events = make_pipeline(tmp_path, 12, asr, wav=True)
    await run(pipeline, source)
    assert len(asr.calls) == 3 and all(abs(c - 6.0) < 0.01 for c in asr.calls)      # windows 0-6, 3-9, 6-12
    assert [s.index for s in pipeline.segments] == [1, 2, 3]
    assert pipeline.segments[0].start == 0 and pipeline.segments[1].start == 3
    assert pipeline.segments[1].new_start >= 6.0 - 0.01                                 # overlap belongs to the earlier window
    assert pipeline.finished and not pipeline.unlocked_indexes()
    assert [t for t, _ in events if t == 'transcript_segment'] == ['transcript_segment'] * 3
    with wave.open(str(tmp_path / 'audio.wav')) as wav:                                # the WAV holds every sample once
        assert wav.getnframes() == 12 * 16000 and wav.getframerate() == 16000


async def test_stop_flushes_the_unfinished_tail(tmp_path):
    asr = FakeASR(default='頭痛')
    pipeline, source, _ = make_pipeline(tmp_path, 7, asr)
    await run(pipeline, source)
    assert len(asr.calls) == 2 and abs(asr.calls[1] - 4.0) < 0.01                       # 6 s window + 4 s tail


async def test_simplified_asr_text_is_converted_but_the_original_is_kept(tmp_path):
    asr = FakeASR(default='我觉得头痛三天了')
    pipeline, source, _ = make_pipeline(tmp_path, 6, asr)
    await run(pipeline, source)
    seg = pipeline.segments[0]
    assert seg.raw_asr == '我觉得头痛三天了' and seg.raw == '我覺得頭痛三天了'
    assert seg.added_asr == '我觉得头痛三天了'                                          # kept for the corrector's disambiguation
    assert '覺' in seg.corrected and '觉' not in seg.corrected


async def test_asr_failure_retries_once_then_marks_a_numbered_gap_and_continues(tmp_path):
    asr = FakeASR({0: ASRError('壞掉了'), 1: ASRError('壞掉了')}, default='後續內容')
    pipeline, source, events = make_pipeline(tmp_path, 9, asr)
    await run(pipeline, source)
    gap = pipeline.segments[0]
    assert gap.kind == 'gap' and '音訊可能缺漏' in gap.corrected and gap.index == 1
    assert pipeline.segments[1].kind == 'speech' and pipeline.segments[1].index == 2
    assert pipeline.rolling_start >= 1                                                  # never revise across a gap
    assert ('transcript_gap' in [t for t, _ in events]) and [t for t, _ in events].count('asr_error') == 2
    assert '缺漏' in pipeline.snapshot().text


class CutWordASR(FakeASR):
    """The aligner stretches the last word of every window past the end of the audio: it starts 0.12 s past it, ends `over` s past."""

    def __init__(self, over, **kwargs):
        super().__init__(**kwargs)
        self.over = over

    async def recognize(self, settings, samples):
        result = await super().recognize(settings, samples)
        duration = len(samples) / 16000
        result['items'][-1].update(start=duration + 0.12, end=duration + self.over)
        return result


async def test_a_word_cut_off_by_the_end_of_the_window_does_not_turn_the_window_into_a_gap(tmp_path):
    asr = CutWordASR(0.52, default='我頭痛三天了')
    pipeline, source, events = make_pipeline(tmp_path, 12, asr)
    await run(pipeline, source)
    assert [s.kind for s in pipeline.segments] == ['speech'] * 3 and 'asr_error' not in [t for t, _ in events]
    for segment in pipeline.segments:
        assert segment.alignment[-1]['end'] == segment.end                              # held at the end of the window
        assert all(u['start'] <= u['end'] for u in segment.alignment)
    assert '頭痛' in pipeline.snapshot().text


async def test_a_word_stretched_further_than_the_edge_allowance_still_makes_a_gap(tmp_path):
    asr = CutWordASR(0.9, default='我頭痛三天了')
    pipeline, source, events = make_pipeline(tmp_path, 9, asr)
    await run(pipeline, source)
    assert pipeline.segments[0].kind == 'gap' and [t for t, _ in events].count('asr_error') >= 2     # tried twice, then a gap


async def test_silence_creates_no_segment_but_advances(tmp_path):
    asr = FakeASR(default='')
    pipeline, source, _ = make_pipeline(tmp_path, 6, asr)
    await run(pipeline, source)
    assert pipeline.segments == [] and pipeline.committed_end == 6.0


async def test_llm_correction_revises_recent_segments_and_records_it(tmp_path):
    fake = FakeLLM()
    fake.queue('corrector', lambda m: {'segments': [{'index': 1, 'text': '我頭痛三天了（已校稿）'}]})
    caller = LLMCaller(LLMClient(fake.transport()), timeout=lambda: 5, retries=lambda: 3, backoff=0)
    corrector = TranscriptCorrector(caller, lambda: make_settings(tmp_path).corrector,
                                    lambda: '黃耆')
    asr = FakeASR(default='我頭痛三天了')
    pipeline, source, events = make_pipeline(tmp_path, 6, asr, corrector=corrector)
    await run(pipeline, source)
    seg = pipeline.segments[0]
    assert seg.corrected == '我頭痛三天了（已校稿）' and seg.added == '我頭痛三天了' and seg.corrected_by == 'llm'
    assert pipeline.revisions and 'transcript_revised' in [t for t, _ in events]
    payload = fake.calls_for('corrector')[0][1]['content']
    assert '黃耆' in fake.calls_for('corrector')[0][0]['content'] and 'segments' in payload


async def test_failed_correction_keeps_text_and_warns(tmp_path):
    fake = FakeLLM()
    fake.queue('corrector', 'not json', 'not json', 'not json')
    caller = LLMCaller(LLMClient(fake.transport()), timeout=lambda: 5, retries=lambda: 3, backoff=0)
    corrector = TranscriptCorrector(caller, lambda: make_settings(tmp_path).corrector, lambda: '')
    pipeline, source, _ = make_pipeline(tmp_path, 6, FakeASR(default='頭痛'), corrector=corrector)
    await run(pipeline, source)
    seg = pipeline.segments[0]
    assert seg.corrected == '頭痛' and '保留目前文字' in seg.warning
    assert len(fake.calls_for('corrector')) == 2                                        # corrector uses 2 attempts, not 4


async def test_snapshot_lists_unlocked_segments_and_pending_work(tmp_path):
    pipeline, source, _ = make_pipeline(tmp_path, 12, FakeASR(default='頭痛三天'))
    await run(pipeline, source)
    pipeline.finished = False                      # as if recording were still going
    snap = pipeline.snapshot()
    assert snap.max_index == 3 and snap.unlocked == [2, 3]
    assert snap.text.startswith('【診間對話逐字稿 · 截至 00:12 · 最新的 #2、#3 仍在校稿中（文字日後可能微調，仍是有效來源，可以引用）】')
    assert '尚未鎖定' not in snap.text                  # the old wording made the agents invent "do not cite" rules
    assert '語音#1 00:00–00:06 ' in snap.lines


async def test_corrector_receives_the_asr_original_so_it_can_fix_opencc_ambiguities(tmp_path):
    fake = FakeLLM()
    caller = LLMCaller(LLMClient(fake.transport()), timeout=lambda: 5, retries=lambda: 3, backoff=0)
    corrector = TranscriptCorrector(caller, lambda: make_settings(tmp_path).corrector, lambda: '')
    pipeline, source, _ = make_pipeline(tmp_path, 6, FakeASR(default='大便偏干'), corrector=corrector)
    await run(pipeline, source)
    row = json.loads(fake.calls_for('corrector')[0][1]['content'])['segments'][0]
    assert row['added_asr'] == '大便偏干' and row['added'] == '大便偏幹'                  # the known 干 -> 幹 mis-conversion
    assert '干→乾／幹' in fake.calls_for('corrector')[0][0]['content']                  # and the prompt tells the model to fix it
