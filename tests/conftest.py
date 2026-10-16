import pytest

from tests.helpers import FakeLLM


@pytest.fixture
def fake_llm():
    return FakeLLM()
