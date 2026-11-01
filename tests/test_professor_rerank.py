"""Rerank asks the model about every candidate parent: at the configured concurrency and never beyond
it, ranking them as one-at-a-time would, stopping on request and counting what failed."""
import threading
from types import SimpleNamespace

import pytest

from Professor import ProfessorInstance


class FakeRerankClient:
    """The slice of the OpenAI client `_llm_rerank` uses; records how many calls overlapped."""

    def __init__(self, answer=lambda text: "0.5"):
        self.answer = answer
        self.calls = 0
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages):
        with self._lock:
            self.calls += 1
            self._active += 1
            self.peak = max(self.peak, self._active)
        try:
            content = self.answer(messages[-1]["content"])
        finally:
            with self._lock:
                self._active -= 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def parents(n, dense=None):
    return [(SimpleNamespace(page_content=f"段落{i}"), dense if dense is not None else i / 100) for i in range(n)]


def rerank(client, docs, *, stop=None, **rerank_cfg):
    prof = ProfessorInstance("professor_test", {"rerank": {"model_name": "fake", **rerank_cfg}})
    prof.prompt_rerank = "評分"
    prof._get_rerank_client = lambda: client
    logs = []
    return prof._llm_rerank("問題", docs, logs.append, manual_stop_event=stop), logs


def test_parents_are_scored_side_by_side_up_to_the_configured_concurrency():
    together = threading.Barrier(4, timeout=5)        # breaks, and fails the test, unless 4 calls overlap

    def wait_for_the_others(_text):
        together.wait()
        return "0.8"

    client = FakeRerankClient(wait_for_the_others)
    out, logs = rerank(client, parents(8), max_concurrency=4)
    assert client.calls == 8 and client.peak == 4
    assert all(score == 0.8 for score, _, _ in out)
    assert "並行 4，失敗 0/8 段" in logs[-1]


def test_no_more_workers_than_parents():
    client = FakeRerankClient()
    _, logs = rerank(client, parents(3), max_concurrency=50)
    assert "並行 3，" in logs[-1]


@pytest.mark.parametrize("cfg", [{}, {"max_concurrency": None}, {"max_concurrency": ""}, {"max_concurrency": 0},
                                 {"max_concurrency": -3}, {"max_concurrency": "很多"}])
def test_an_unset_or_unusable_concurrency_stays_one_at_a_time(cfg):
    client = FakeRerankClient()
    out, _ = rerank(client, parents(5), **cfg)
    assert len(out) == 5 and client.peak == 1


def test_the_ranking_is_the_same_as_one_at_a_time():
    scores = {f"段落{i}": s for i, s in enumerate(["0.9", "0.2", "0.9", "0.5", "0.2", "0.5", "0.9"])}

    def score_of(text):
        return scores[text.split("段落：")[1].split("\n")[0]]

    docs = parents(7, dense=0.3)                       # rerank and dense both tie, so only the order decides
    serial, _ = rerank(FakeRerankClient(score_of), docs, max_concurrency=1)
    parallel, _ = rerank(FakeRerankClient(score_of), docs, max_concurrency=5)
    assert [d.page_content for _, _, d in parallel] == [d.page_content for _, _, d in serial]


def test_a_failed_or_unreadable_answer_scores_zero_and_is_counted_unlike_an_honest_zero():
    def answer(text):
        if "段落0" in text:
            raise ConnectionError("server down")
        return "無法判斷" if "段落1" in text else "0.0"

    out, logs = rerank(FakeRerankClient(answer), parents(3), max_concurrency=2)
    assert [score for score, _, _ in out] == [0.0, 0.0, 0.0]
    assert "失敗 2/3 段" in logs[-1]


def test_a_stop_skips_the_parents_not_yet_sent():
    stop = threading.Event()

    def stop_after_the_first(_text):
        stop.set()
        return "0.6"

    client = FakeRerankClient(stop_after_the_first)
    out, logs = rerank(client, parents(6), stop=stop, max_concurrency=1)
    assert client.calls == 1 and len(out) == 1
    assert any("收到手動中斷" in line for line in logs)


def test_a_stop_before_anything_was_sent_sends_nothing():
    stop = threading.Event()
    stop.set()
    client = FakeRerankClient()
    out, _ = rerank(client, parents(4), stop=stop, max_concurrency=4)
    assert client.calls == 0 and out == []


def test_no_candidates_means_no_calls():
    client = FakeRerankClient()
    out, _ = rerank(client, [], max_concurrency=4)
    assert out == [] and client.calls == 0


def test_without_a_rerank_model_the_dense_order_is_kept():
    prof = ProfessorInstance("professor_test", {"rerank": {"max_concurrency": 4}})
    out = prof._llm_rerank("問題", parents(3), lambda _msg: None)
    assert [dense for _, dense, _ in out] == [0.02, 0.01, 0.0]
