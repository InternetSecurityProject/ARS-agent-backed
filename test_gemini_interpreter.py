"""
gemini_interpreter 시험 (실제 Gemini 호출 없음, 가짜 응답 사용)

python -m pytest -q test_gemini_interpreter.py
"""

from __future__ import annotations

import asyncio

from gemini_interpreter import GeminiInterpreter, confirm_key, normalize_keys, rule_classify
from interpreter import Option, Prompt


class FakeClient:
    """Gemini 대신 미리 정한 응답을 돌려준다. fail=True 면 항상 실패."""

    def __init__(self, replies: list[dict] | None = None, fail: bool = False) -> None:
        self.replies, self.fail, self.calls = list(replies or []), fail, []

    async def generate(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append(user)
        if self.fail or not self.replies:
            raise RuntimeError("503")
        return self.replies.pop(0)


def run(coro):
    return asyncio.run(coro)


ROOT = ("KB국민은행입니다. 잔액조회는 3번, 거래내역조회는 4번, 카드 업무는 5번, 상담직원 연결은 0번을 눌러 주십시오. "
        "다시 들으시려면 우물정자를 눌러 주십시오.")


# 한국어 번호 -------------------------------------------------------------
def test_normalize_spoken_numbers():
    assert normalize_keys("잔액조회는 삼 번을 눌러 주세요") == "잔액조회는 3번을 눌러 주세요"
    assert normalize_keys("이전 메뉴는 별표, 다시 듣기는 우물정자") == "이전 메뉴는 *, 다시 듣기는 #"
    assert normalize_keys("상담원 연결은 공번") == "상담원 연결은 0번"
    assert normalize_keys("일이삼 번") == "123번"


def test_normalize_keeps_non_key_words():
    assert normalize_keys("이번 달 카드 대금") == "이번 달 카드 대금"
    assert normalize_keys("이번에 결제할 금액") == "이번에 결제할 금액"
    assert normalize_keys("전화번호를 입력") == "전화번호를 입력"
    assert normalize_keys("일반 상담") == "일반 상담"


# classify ----------------------------------------------------------------
def test_classify_drops_guessed_keys():
    reply = {"kind": "menu", "options": [
        {"key": "3", "label": "잔액조회", "category": "잔액 조회"},
        {"key": "7", "label": "환전", "category": "환전"},          # 안내에 없는 번호 → 제거
        {"key": "0", "label": "상담직원 연결", "category": "상담원 연결"}],
        "back_key": None, "home_key": None, "repeat_key": "#", "agent_key": "0",
        "label": None, "terminator": None, "critical": False}
    p = run(GeminiInterpreter(FakeClient([reply])).classify(ROOT))
    assert p.kind == "menu"
    assert [o.key for o in p.options] == ["3", "0"]
    assert p.repeat_key == "#" and p.agent_key == "0"


def test_classify_spoken_numbers_reach_model_as_digits():
    fake = FakeClient([{"kind": "menu", "options": [{"key": "3", "label": "잔액조회", "category": "잔액 조회"}],
                        "back_key": "*", "home_key": None, "repeat_key": None, "agent_key": None,
                        "label": None, "terminator": None, "critical": False}])
    p = run(GeminiInterpreter(fake).classify("잔액조회는 삼 번, 이전 메뉴는 별표를 눌러 주십시오."))
    assert "3번" in fake.calls[0] and "*" in fake.calls[0]
    assert p.options[0].key == "3" and p.back_key == "*"


def test_classify_failure_falls_back_to_rules():
    interp = GeminiInterpreter(FakeClient(fail=True))
    p = run(interp.classify(ROOT))
    assert p.kind == "menu"
    assert {o.key for o in p.options} == {"3", "4", "5", "0"}
    assert p.repeat_key == "#"


def test_sensitive_is_caught_even_if_model_misses():
    reply = {"kind": "info", "options": [], "back_key": None, "home_key": None, "repeat_key": None,
             "agent_key": None, "label": None, "terminator": None, "critical": False}
    p = run(GeminiInterpreter(FakeClient([reply])).classify("주민등록번호 앞 여섯 자리와 우물정자를 눌러 주십시오."))
    assert p.kind == "sensitive" and p.terminator == "#"
    p = run(GeminiInterpreter(FakeClient(fail=True)).classify("계좌 비밀번호 네 자리를 눌러 주십시오."))
    assert p.kind == "sensitive"


def test_rule_classify_readback_is_not_sensitive():
    p = rule_classify("입력하신 주민등록번호는 ******입니다. 맞으면 1번, 틀리면 2번을 눌러 주십시오.")
    assert p.kind == "confirm"


# score -------------------------------------------------------------------
OPTS = [Option("3", "잔액조회", "잔액 조회"), Option("5", "카드 업무", "카드 업무")]


def test_score_clamps_and_fills_missing():
    reply = {"scores": [{"key": "3", "confidence": 1.4}]}
    s = run(GeminiInterpreter(FakeClient([reply])).score("통장 잔액", OPTS))
    assert s == {"3": 1.0, "5": 0.0}


def test_score_failure_asks_user_instead_of_pressing():
    s = run(GeminiInterpreter(FakeClient(fail=True)).score("통장 잔액", OPTS))
    assert all(0.30 <= v < 0.50 for v in s.values())  # 서버 기준 '불확실' → low_confidence 재질문


# 확인 질문 '네/아니요' --------------------------------------------------------
CONFIRM = Prompt(kind="confirm", text="입력하신 번호가 맞으면 1번, 틀리면 2번을 눌러 주십시오.",
                 options=[Option("1", "맞음"), Option("2", "틀림")])


def test_confirm_rule_without_llm():
    assert confirm_key(CONFIRM, "네") == "1"
    assert confirm_key(CONFIRM, "맞아") == "1"
    assert confirm_key(CONFIRM, "아니요") == "2"
    assert confirm_key(CONFIRM, "음 글쎄") is None


def test_answer_confirm_uses_user_yes():
    fake = FakeClient(fail=True)
    interp = GeminiInterpreter(fake)
    goal = run(interp.merge_answer("건강보험증 발급", "ARS 가 이렇게 묻습니다", "맞아"))
    assert goal == "건강보험증 발급"
    assert run(interp.answer_confirm(goal, CONFIRM)) == "1"
    assert fake.calls == []  # LLM 호출 없음


def test_answer_confirm_never_for_critical():
    interp = GeminiInterpreter(FakeClient([{"key": "1"}]))
    interp.last_answer = "네"
    crit = Prompt(kind="confirm", text="해지하시려면 1번", options=[Option("1", "해지 진행")], critical=True)
    assert run(interp.answer_confirm("해지", crit)) is None


# merge_answer ------------------------------------------------------------
def test_merge_answer_spoken_number_uses_option():
    interp = GeminiInterpreter(FakeClient(fail=True))
    interp.last_menu = Prompt(kind="menu", text=ROOT, options=OPTS)
    assert run(interp.merge_answer("결제 관련", "어느 쪽인가요?", "삼 번이요")) == "잔액 조회"


def test_merge_answer_failure_keeps_both():
    interp = GeminiInterpreter(FakeClient(fail=True))
    assert run(interp.merge_answer("카드 관련 문의", "자세히 말씀해 주세요", "결제일 바꾸고 싶어")) == \
        "카드 관련 문의 (결제일 바꾸고 싶어)"


# 결과·요약 ----------------------------------------------------------------
def test_summaries_fall_back_without_model():
    interp = GeminiInterpreter(FakeClient(fail=True))
    assert run(interp.summarize_goal("통장 잔액")) == "'통장 잔액' 업무를 찾아 드릴게요."
    res = run(interp.summarize_result("통장 잔액", ["출금 가능 금액은 1,234,000원입니다."]))
    assert res == {"summary": "출금 가능 금액은 1,234,000원입니다.", "key_info": []}
    assert "카드 업무 > 카드 분실 신고" in run(interp.handoff_script("카드 분실", ["카드 업무", "카드 분실 신고"]))


def test_summarize_result_key_info():
    reply = {"summary": "수원 오늘은 오후에 비가 옵니다.",
             "key_info": [{"label": "강수확률", "value": "60%"}, {"label": "", "value": "x"}]}
    res = run(GeminiInterpreter(FakeClient([reply])).summarize_result("수원 오늘 날씨", ["..."]))
    assert res["key_info"] == [{"label": "강수확률", "value": "60%"}]


# 가상 ARS 전체 흐름 (서버 + GeminiInterpreter, Gemini 대신 가짜 응답) -----------------
class KeywordClient:
    """classify 는 실패시켜 규칙 기반 분류를 쓰고, score 는 선택지 이름별로 정한 값을 준다 (표에 없으면 0.05)."""

    def __init__(self, table: dict[str, float] | None = None, down: bool = False) -> None:
        self.table, self.down = table or {}, down

    async def generate(self, system: str, user: str, schema: dict) -> dict:
        import json
        import re

        if self.down or "kind" in schema["properties"]:
            raise RuntimeError("503")
        if "scores" in schema["properties"]:
            opts = json.loads(re.search(r"<options>\n(.*?)\n</options>", user, re.S).group(1))
            return {"scores": [{"key": o["key"], "confidence": self.table.get(o["label"], 0.05)} for o in opts]}
        raise RuntimeError("503")


def _flow(client, goal):
    from test_flows import new_manager, start, status_seq, wait_for

    async def go():
        m = await new_manager()
        m.deps.interp = GeminiInterpreter(client)
        m.deps.navigator.interp = m.deps.interp
        _, s = await start(m, goal)
        return s, wait_for, status_seq
    return go


def test_flow_with_gemini_interpreter_reaches_result():
    async def go():
        s, wait_for, status_seq = await _flow(KeywordClient({"카드 업무": 0.8, "결제일 변경": 0.95}), "카드 결제일 바꾸고 싶어")()
        await wait_for(s, lambda s: "agent_done" in status_seq(s) or s.ended)
        assert [p["selected"]["key"] for p in s.path] == ["5", "3"]
        assert s.summary["kind"] == "result"
        if not s.ended:
            await s.action({"type": "cancel"})
    run(go())


def test_flow_when_gemini_is_down_asks_user():
    async def go():
        s, wait_for, _ = await _flow(KeywordClient(down=True), "카드 결제일 바꾸고 싶어")()
        req = await wait_for(s, lambda s: s.pending)
        assert req["reason"] == "low_confidence" and s.path == []  # 번호를 누르지 않고 사용자에게 묻는다
        await s.action({"type": "cancel"})
    run(go())


# 건강보험공단 평가 사례 (docs/evals, 소민님 데이터) ------------------------------
def test_truncated_menu_number_is_not_replaced_by_similar_option():
    """nhis_card_003: '건강보험증 발급은,' 번호가 잘리면 비슷한 '증명서 발급'(1번)을 누르지 않는다."""
    text = "증명서 발급은 1번, 건강보험증 발급은, 고지서 발급은 3번을 눌러주십시오."
    cls = {"kind": "menu", "options": [{"key": "1", "label": "증명서 발급", "category": "증명서 발급"},
                                       {"key": "3", "label": "고지서 발급", "category": "고지서 발급"}],
           "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
           "label": None, "terminator": None, "critical": False}
    interp = GeminiInterpreter(FakeClient([cls, {"scores": [{"key": "1", "confidence": 0.7}, {"key": "3", "confidence": 0.1}]}]))
    p = run(interp.classify(text))
    s = run(interp.score("건강보험증을 발급받고 싶다", p.options))
    assert max(s.values()) < 0.5


def test_result_with_repeat_keys_is_result():
    """nhis_card_010: '발송됩니다' 뒤에 다시 듣기·전 단계 키가 있어도 결과 안내다."""
    reply = {"kind": "info", "options": [], "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None,
             "agent_key": None, "label": None, "terminator": None, "critical": False}
    p = run(GeminiInterpreter(FakeClient([reply])).classify(
        "고객님의 주민등록상 주소지로 발송됩니다. 다시 듣기는 별표, 전 단계는 우물정자를 눌러주십시오."))
    assert p.kind == "result"


def test_spoken_menu_numbers_nhis():
    """nhis_card_002: '이번'이 2번 메뉴 번호로 쓰인 경우."""
    t = normalize_keys("디지털 알스는 구번, 증명서 발급 일번, 건강 보헙증 발급 이번, 고지서 발급 삼번을 눌러주십시오.")
    assert "9번" in t and "1번" in t and "발급 2번" in t and "3번" in t
