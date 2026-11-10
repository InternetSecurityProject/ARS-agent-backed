"""som 방식 조합(combo.interp=som) 최소 시험. 실통화·실제 JEV 호출 없음.

python -m pytest -q test_som_mode.py
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from pathlib import Path

import pytest

from app import build_deps
from combos import ComboError, ComboFactory
from hybrid_interpreter import HybridJevInterpreter
from jev_interpreter import JevInterpreter
from local_llm import LocalLlmInterpreter
from navigator import Navigator
from session import Deps, SessionManager
from som_jev.jev_menu_interpreter import UnifiedJevInterpreter
from som_mode import SomNavigator
from test_combo import end_all
from test_flows import ev, make_settings, run, wait_for
from test_hybrid_interpreter import FakeJev, FakeLocal

HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("RESULT_JUDGE", raising=False)
    monkeypatch.delenv("COMBO_DEFAULT_INTERP", raising=False)


class ScriptedJev:
    """som 원본 JevClient 자리. 질문 이름마다 정해진 답(선택형은 기준 하나에 0.9, 예/아니오는 0.9 또는 0.1)을 주고 요청을 모음."""

    def __init__(self) -> None:
        self.requests: list[tuple[dict, dict]] = []

    async def decide(self, state, questions):
        self.requests.append((state, questions))
        out = {}
        for name, q in questions.items():
            h = int(hashlib.md5(name.encode()).hexdigest(), 16)
            if q["type"] == "noul":
                out[name] = {"type": "noul", "noul": 0.9 if h % 3 else 0.1}
            else:
                crit = list(q["criteria"])
                want = crit[h % len(crit)]
                out[name] = {"type": "choice", "choice": want, "confidence": 0.9,
                             "probabilities": {c: (0.9 if c == want else 0.1 / max(1, len(crit) - 1)) for c in crit}}
        return out


def factory(som_client):
    base = HybridJevInterpreter(LocalLlmInterpreter(FakeLocal()), JevInterpreter(FakeJev("3")))
    return base, ComboFactory(base.local.client, base.jev.client, "hybrid", "0", som_client=som_client)


def test_vendored_som_modules_differ_only_in_import_paths():
    """som_jev 의 각 파일은 원본 커밋과 가져오기 경로만 다름 (원본 커밋이 이 저장소에 없으면 건너뜀)."""
    rev = "9f628ec8"
    if subprocess.run(["git", "-C", str(HERE), "cat-file", "-e", rev], capture_output=True).returncode != 0:
        pytest.skip("원본 커밋이 없음")
    for f in sorted((HERE / "som_jev").glob("*.py")):
        if f.name == "__init__.py":
            continue
        orig = subprocess.run(["git", "-C", str(HERE), "show", f"{rev}:{f.name}"], capture_output=True, check=True).stdout
        o, n = orig.decode("utf-8").splitlines(), f.read_bytes().decode("utf-8").splitlines()
        assert len(o) == len(n), f.name
        for a, b in zip(o, n):
            if a != b:
                assert a in (re.sub(r"from \.(\w+)", r"from \1", b), re.sub(r"from \. import (\w+)", r"import \1", b)), (f.name, b)


def test_som_combo_values_and_default(monkeypatch):
    base, f = factory(ScriptedJev())
    assert "som" in [x["id"] for x in f.options()["interp"]]
    assert f.normalize("som", "separate") == ("som", "0") and f.normalize_result("som", "local") == "jev"
    a, b = f.build("som", "0"), f.build("som", "0")
    assert isinstance(a, UnifiedJevInterpreter) and a is not b and a.client is b.client is f.som_client
    _, none = factory(None)
    assert "som" not in [x["id"] for x in none.options()["interp"]]
    with pytest.raises(ComboError):
        none.normalize("som", "0")
    monkeypatch.setenv("COMBO_DEFAULT_INTERP", "som")
    assert factory(ScriptedJev())[1].default_interp == "som"
    assert factory(None)[1].default_interp == "hybrid"          # 고를 수 없으면 기본을 바꾸지 않음


def test_som_call_runs_on_sim_backend(caplog):
    """가상 백엔드로 som 방식 통화 하나: 원본 탐색기·해석기가 쓰이고, 이력과 함께 분류 요청이 JEV 로 가며, 통화 처리 오류가 없음."""
    caplog.set_level(logging.INFO)

    async def go():
        jev = ScriptedJev()
        base, f = factory(jev)
        d = build_deps(make_settings(sim_lines=1))
        d.backend.tree["timeout_s"] = 300
        await d.backend.start()
        m = SessionManager(Deps(d.settings, d.backend, d.recognizer, base, Navigator(base), f))
        s = m.sessions[(await m.create("15889999", "통장 잔액 알고 싶어", interp="som"))["session_id"]]
        try:
            await wait_for(s, lambda s: ev(s, "path_updated") or s.pending is not None or "보류" in str(ev(s, "status_changed")),
                           timeout=10)
            return s, jev
        finally:
            await end_all(m)

    s, jev = run(go())
    assert isinstance(s.d.navigator, SomNavigator) and isinstance(s.d.interp, UnifiedJevInterpreter)
    assert s.combo == {"interp": "som", "fix": "0", "result": "jev"}
    assert jev.requests and any("current_ars_transcript" in str(st) or "transcript" in str(st) for st, _ in jev.requests)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "안내 처리 오류" not in text and "세션 작업 오류" not in text


def test_som_option_names_are_trimmed_only_for_display():
    """som 해석기가 원문 구절 그대로 뽑은 이름('…을 원하시면')은 화면용 고침으로만 다듬음 (선택지 이름 자체는 그대로)."""
    from interpreter import Option, Prompt
    from session import fix_words
    from som_mode import display_fixes_for, display_label

    assert display_label("상담원 연결을 원하시면") == "상담원 연결"
    assert display_label("혜택을 한눈에 보며 카드 발급 신청을 할 수 있는 모바일 홈페이지 문자 수신을 원하시면") == \
        "혜택을 한눈에 보며 카드 발급 신청을 할 수 있는 모바일 홈페이지 문자 수신"
    assert display_label("잔액 조회는") == "잔액 조회" and display_label("상담원 연결") == "상담원 연결" and display_label("면") == "면"
    p = Prompt(kind="menu", text="…", options=[Option("1", "잔액 조회를 원하시면"), Option("0", "상담원 연결")])
    fixes = display_fixes_for(p)
    assert fixes == [("잔액 조회를 원하시면", "잔액 조회")] and p.options[0].label == "잔액 조회를 원하시면"
    assert fix_words("잔액 조회를 원하시면 / 상담원 연결 중 어떤 쪽인가요?", fixes) == "잔액 조회 / 상담원 연결 중 어떤 쪽인가요?"
