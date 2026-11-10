"""
gemini_interpreter 시험 (실제 Gemini 호출 없음, 가짜 응답 사용)

python -m pytest -q test_gemini_interpreter.py
"""

from __future__ import annotations

import asyncio

import pytest

from gemini_interpreter import GeminiInterpreter, confirm_key, menu_overrides_kind, normalize_keys, rule_classify
from interpreter import Option, Prompt


class FakeClient:
    """Gemini 대신 미리 정한 응답 반환. fail=True 면 항상 실패."""

    def __init__(self, replies: list[dict] | None = None, fail: bool = False) -> None:
        self.replies, self.fail, self.calls = list(replies or []), fail, []

    async def generate(self, system: str, user: str, schema: dict, **kw) -> dict:
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


NH = ("이체 및 납부는 1번, 조회는 2번, 외환·외국어 상담은 7번, 상담원 연결은 0번, 다시 듣기는 별표 버튼, "
      "이전 단계로 돌아가기는 샵 버튼을 눌러주십시오.")


def reply(options: dict, agent: str | None, scores: dict | None = None, kind: str = "menu", back=None, rep=None) -> dict:
    d = {"kind": kind, "options": [{"key": k, "label": v, "category": v} for k, v in options.items()],
         "unnumbered": [], "back_key": back, "home_key": None, "repeat_key": rep, "agent_key": agent,
         "label": None, "terminator": None, "critical": False}
    if scores is not None:
        d["scores"] = [{"key": k, "confidence": v} for k, v in scores.items()]
    return d


def classify_with(text: str, d: dict, goal: str = "", more: tuple = ()):
    fake = FakeClient([d, *more])
    interp = GeminiInterpreter(fake)
    interp.goal = goal
    return interp, fake, run(interp.classify(text))


def scores(**kv) -> dict:
    return {"scores": [{"key": k.lstrip("k"), "confidence": v} for k, v in kv.items()]}


def test_agent_key_missing_from_options_is_restored_and_rescored():
    """모델이 상담원 번호를 agent_key 에만 적고 options·scores 에서 빠뜨린 경우(목적에 맞는 선택지 없음) 확인:
    규칙으로 찾은 같은 번호 선택지로 복원, 점수는 모델에 재질의 (목적이 상담원 연결인지는 모델 판단)."""
    d = reply({"1": "이체 및 납부", "2": "조회", "7": "외환·외국어 상담"}, "0", {"1": 0.0, "2": 0.0, "7": 0.0}, back="#", rep="*")
    interp, fake, p = classify_with(NH, d, "상담원 연결", (scores(k1=0.0, k2=0.0, k7=0.1, k0=0.95),))
    assert (p.options[-1].key, p.options[-1].label, p.options[-1].category) == ("0", "상담원 연결", "상담원 연결")
    assert p.agent_key == "0"
    s = run(interp.score("상담원 연결", p.options))
    assert s["0"] == 0.95 and len(fake.calls) == 2
    # 재질의 실패 시: 분류 때 받은 점수는 그대로 사용, 복원한 선택지만 재질문 값 (분명한 선택은 그대로 누름)
    from gemini_interpreter import FAIL_SCORE
    d = reply({"1": "이체 및 납부", "2": "조회", "7": "외환·외국어 상담"}, "0", {"1": 0.0, "2": 0.95, "7": 0.0}, back="#", rep="*")
    interp, fake, p = classify_with(NH, d, "조회하고 싶어")
    s = run(interp.score("조회하고 싶어", p.options))
    assert s["2"] == 0.95 and s["0"] == FAIL_SCORE


def test_agent_key_restored_for_symbol_key_and_other_phrasings():
    text = "잔액 조회는 1번, 카드 업무는 2번, 상담원 연결은 샵 버튼을 눌러 주세요."
    interp, fake, p = classify_with(text, reply({"1": "잔액 조회", "2": "카드 업무"}, "#", {"1": 0.1, "2": 0.1}), "상담원 연결",
                                    (scores(k1=0.1, k2=0.1, **{"#": 0.95}),))
    assert p.options[-1].key == "#" and run(interp.score("상담원 연결", p.options))["#"] == 0.95 and len(fake.calls) == 2
    for text, key, label in [
            ("잔액 조회는 1번 상담원과 통화를 원하시면 0번", "0", "상담원과 통화"),       # 쉼표 없음, 끝의 '원하시(면)'
            ("잔액 조회 1번 상담원 연결은 0번", "0", "상담원 연결"),                    # 앞 선택지가 이름에 붙음
            ("잔액 조회는 1번, 카드 업무는 2번, 상담원하고 통화하시려면 0번을 눌러 주세요.", "0", "상담원하고 통화"),
            ("잔액 조회는 1번, 0번은 상담직원 연결입니다.", "0", "상담직원 연결"),          # 번호가 앞에 옴
            ("상담 시간은 평일 9시부터 6시까지입니다. 잔액 조회는 1번, 상담원 연결은 9번", "9", "상담원 연결"),  # 시각의 9 는 번호가 아님
            ("잔액 조회는 1번, 카드 분실 신고는 2번, 상담원 연결은 0번입니다. 감사합니다.", "0", "상담원 연결")]:
        _, _, p = classify_with(text, reply({"1": "잔액 조회"}, key))
        assert (p.options[-1].key, p.options[-1].label) == (key, label) and p.agent_key == key, text


def test_agent_key_is_not_restored_when_unsafe():
    # 상담원 연결이 아닌 번호를 모델이 agent_key 로 잘못 적음: '외국어 상담은 7번', '상담 원하시면 7번', 쉼표 없는 '9번 상담원 연결은'
    for text, agent in [(NH, "7"),
                        ("이체는 1번, 외환 상담 원하시면 7번, 상담원 연결은 0번을 눌러 주십시오.", "7"),
                        ("잔액 조회는 1번 카드 분실 신고는 9번 상담원 연결은 0번", "9"),
                        ("잔액 조회는 1번, 대출 상담 사전 예약은 3번", "3"),
                        ("잔액 조회는 1번, 상담 사이트 안내는 3번", "3")]:
        _, _, p = classify_with(text, reply({"1": "이체"}, agent))
        assert agent not in [o.key for o in p.options] and p.agent_key is None, text
    # 같은 번호가 두 번 들림 (STT 오류 의심): 복원 안 함
    dup = "카드 분실 신고는 0번, 잔액 조회는 1번, 상담원 연결은 0번을 눌러 주세요."
    for d in (reply({"1": "잔액 조회"}, "0"),
              {**reply({"1": "잔액 조회"}, "0"), "options": [{"key": "0", "label": "카드 분실 신고"}, {"key": "1", "label": "잔액 조회"},
                                                          {"key": "0", "label": "상담원 연결"}]}):
        _, _, p = classify_with(dup, d)
        assert [o.key for o in p.options] == ["1"] and p.agent_key is None
    # 결과 안내에는 선택지 미추가 (자막·결과 요약에 원문 유지 필요)
    result = "고객님의 출금 가능 금액은 12,000원입니다. 다시 들으시려면 별표, 상담원 연결은 0번을 눌러 주십시오."
    _, _, p = classify_with(result, reply({}, "0", kind="result", rep="*"))
    assert p.kind == "result" and p.options == []
    # 다른 선택지가 없는 상담원 안내 한 줄은 메뉴로 만들지 않음
    _, _, p = classify_with("상담원 연결을 원하시면 0번을 눌러 주십시오.", reply({}, "0"))
    assert p.kind != "menu" and p.options == []


def test_greeting_ending_with_thanks_is_not_forced_to_result():
    """'○○카드입니다. 감사합니다.' 같은 인사는 모델이 info 로 판정하면 결과로 바꾸지 않음 확인 (첫 메뉴 전 탐색 종료 방지).
    강한 결과 표현('~원입니다', '접수되었습니다')은 결과로 변경."""
    _, _, p = classify_with("안녕하십니까, 신한카드입니다. 감사합니다.", reply({}, None, kind="info"))
    assert p.kind == "info"
    _, _, p = classify_with("이번 달 결제 금액은 12,000원입니다. 감사합니다.", reply({}, None, kind="info"))
    assert p.kind == "result"


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
    """분류는 규칙 기반 결과, 점수는 선택지 이름별로 정한 값(표에 없으면 0.05)을 Gemini 응답 형식으로 반환."""

    def __init__(self, table: dict[str, float] | None = None, down: bool = False) -> None:
        self.table, self.down = table or {}, down

    async def generate(self, system: str, user: str, schema: dict, **kw) -> dict:
        import json
        import re

        if self.down:
            raise RuntimeError("503")
        if "kind" in schema["properties"]:  # 분류: 규칙 기반 결과를 Gemini 응답 형식으로 반환
            ars = re.search(r"<ars>\n(.*?)\n</ars>", user, re.S).group(1)
            p = rule_classify(ars)
            return {"kind": p.kind, "options": [{"key": o.key, "label": o.label, "category": o.label} for o in p.options],
                    "unnumbered": [], "back_key": p.back_key, "home_key": p.home_key, "repeat_key": p.repeat_key,
                    "agent_key": p.agent_key, "label": p.label, "terminator": p.terminator, "critical": p.critical,
                    "scores": [{"key": o.key, "confidence": self.table.get(o.label, 0.05)} for o in p.options]}
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
        assert req["reason"] == "low_confidence" and s.path == []  # 번호를 누르지 않고 사용자에게 재질문
        await s.action({"type": "cancel"})
    run(go())


# 번호 잘림·결과 안내·말로 한 번호 사례 ------------------------------
def test_truncated_menu_number_is_not_replaced_by_similar_option():
    """번호가 잘린 선택지를 목적으로 할 때 이름이 비슷한 다른 선택지(1번)를 누르지 않음 확인."""
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
    """'발송됩니다' 뒤에 다시 듣기·전 단계 키가 있어도 결과 안내로 판정 확인."""
    reply = {"kind": "info", "options": [], "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None,
             "agent_key": None, "label": None, "terminator": None, "critical": False}
    p = run(GeminiInterpreter(FakeClient([reply])).classify(
        "고객님의 주민등록상 주소지로 발송됩니다. 다시 듣기는 별표, 전 단계는 우물정자를 눌러주십시오."))
    assert p.kind == "result"


def test_spoken_menu_numbers_nhis():
    """'이번'이 2번 메뉴 번호로 쓰인 경우의 변환 확인."""
    t = normalize_keys("디지털 알스는 구번, 증명서 발급 일번, 건강 보헙증 발급 이번, 고지서 발급 삼번을 눌러주십시오.")
    assert "9번" in t and "1번" in t and "발급 2번" in t and "3번" in t


# '#' 키 이름 ------------------------------------------------------------------
def test_hash_key_common_names_only():
    """'#' 변환 대상이 모든 ARS 공통 이름(샵 버튼·우물정자·해시)뿐인지 확인. 특정 통화의 STT 오인식 낱말(취업·시합 버튼,
    우물 강자)은 범용성을 위해 미사용."""
    assert normalize_keys("다른 서비스는 샵 버튼을 눌러주십시오") == "다른 서비스는 #을 눌러주십시오"
    assert normalize_keys("전단계는 우물정자를") == "전단계는 #를"
    assert normalize_keys("전단계는 우물 정자를") == "전단계는 #를"
    assert normalize_keys("다른 서비스는 취업 버튼을 눌러주십시오") == "다른 서비스는 취업 버튼을 눌러주십시오"
    assert normalize_keys("취업 준비 상담은 1번") == "취업 준비 상담은 1번"
    # 다른 낱말 안의 샵·샾은 키 아님
    assert normalize_keys("워크샾 신청은 1번") == "워크샾 신청은 1번"
    assert normalize_keys("KT샵 주문 조회는 3번") == "KT샵 주문 조회는 3번"
    assert normalize_keys("포토샵 문의는 1번") == "포토샵 문의는 1번"
    assert normalize_keys("다른 서비스는 샾") == "다른 서비스는 #"
    assert normalize_keys("샵버튼을 누르세요") == "#을 누르세요"
    assert normalize_keys("샾 키") == "#"
    assert normalize_keys("이전 메뉴는샵 버튼을 눌러 주세요") == "이전 메뉴는#을 눌러 주세요"  # 조사에 붙은 키 이름


def test_real_kb_spoken_menu_numbers():
    t = normalize_keys("다시 듣기는 영 번 국민은행으로 송금은 일 번 다른 은행으로 송금은 이번 예금 펀드")
    assert t == "다시 듣기는 0번 국민은행으로 송금은 1번 다른 은행으로 송금은 2번 예금 펀드"


def test_real_nhis_rule_fallback():
    """Gemini 실패 시에도 실제 안내 문장의 이동 키·민감정보를 규칙 분류로 판정하는지 확인."""
    p = rule_classify("건강보험증 발급은 1번, 상담사 연결은 0번, 다시 듣기는 별표 전 단계는 우물 정자를 눌러주십시오.")
    assert p.kind == "menu" and p.repeat_key == "*" and p.back_key == "#"
    assert rule_classify("고객님의 주민등록번호 앞 여섯자리와 샵 버튼을 눌러주십시오.").kind == "sensitive"


# 분류·점수 단일 호출과 선택지 처리 ------------------------------------------------
def _menu(options, **kw):
    d = {"kind": "menu", "options": options, "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None,
         "agent_key": None, "label": None, "terminator": None, "critical": False}
    d.update(kw)
    return d


def test_classify_and_score_in_one_call_and_cache():
    fake = FakeClient([_menu([{"key": "3", "label": "잔액조회", "category": "잔액 조회"},
                              {"key": "5", "label": "카드 업무", "category": "카드 업무"}],
                             scores=[{"key": "3", "confidence": 0.9}, {"key": "5", "confidence": 0.1}])])
    interp = GeminiInterpreter(fake)
    interp.goal = "통장 잔액"
    text = "잔액조회는 3번, 카드 업무는 5번을 눌러 주십시오."
    p = run(interp.classify(text))
    assert run(interp.score("통장 잔액", p.options)) == {"3": 0.9, "5": 0.1}
    run(interp.classify(text))                      # 같은 안내 다시 듣기
    assert len(fake.calls) == 1                     # Gemini 호출은 한 번뿐


def test_decide_calls_use_short_timeout_without_retry():
    seen = []

    class Spy(FakeClient):
        async def generate(self, system, user, schema, **kw):
            seen.append(kw)
            raise RuntimeError("timeout")

    interp = GeminiInterpreter(Spy())
    run(interp.classify("잔액조회는 3번을 눌러 주십시오."))
    assert seen[0] == {"timeout_s": interp.decide_timeout_s, "retries": 0}
    assert interp.decide_timeout_s == 4.0


def test_symbol_options_kept_when_model_drops_them():
    """모델이 빠뜨린 기호 키 선택지('빠른 말 또는 쉬운 말 서비스는 샵 버튼')의 규칙 보완 확인."""
    text = "잔액 조회는 3번, 빠른 말 또는 쉬운 말 서비스는 샵 버튼 거래 내역 조회는 4번을 눌러주십시오."
    reply = _menu([{"key": "3", "label": "잔액 조회", "category": "잔액 조회"},
                   {"key": "4", "label": "거래 내역 조회", "category": "거래내역 조회"}])
    p = run(GeminiInterpreter(FakeClient([reply])).classify(text))
    assert "#" in [o.key for o in p.options]


def test_native_korean_numbers_after_topic_marker():
    """주제 조사 뒤 고유어 번호 변환 확인: '문의는 한 번' → 1번, '다시 한 번 눌러' 는 그대로."""
    assert normalize_keys("카드 문의는 한 번, 상담원은 두 번") == "카드 문의는 1번, 상담원은 2번"
    assert normalize_keys("다시 한 번 눌러 주십시오") == "다시 한 번 눌러 주십시오"


def test_duplicate_key_is_suspicious_not_silently_dropped():
    """같은 번호가 두 선택지에 붙으면 둘 다 제외, 목적이 그쪽이면 누르지 않음 확인."""
    text = "카드 분실 신고는 1번, 결제일 변경은 1번, 상담원 연결은 0번을 눌러 주십시오."
    reply = _menu([{"key": "1", "label": "카드 분실 신고", "category": "카드 분실 신고"},
                   {"key": "1", "label": "결제일 변경", "category": "카드 결제일 변경"},
                   {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}])
    interp = GeminiInterpreter(FakeClient([reply, {"scores": [{"key": "0", "confidence": 0.6}]}]))
    p = run(interp.classify(text))
    assert [o.key for o in p.options] == ["0"]
    assert max(run(interp.score("카드 분실 신고하고 싶어", p.options)).values()) < 0.5


def test_score_failure_uses_exact_name_match():
    opts = [Option("1", "증명서 발급"), Option("2", "건강보험증 발급"), Option("3", "고지서 발급")]
    s = run(GeminiInterpreter(FakeClient(fail=True)).score("건강보험증을 발급받고 싶다", opts))
    assert s["2"] >= 0.5 and s["1"] < 0.3


def test_option_label_drops_trailing_key_phrase():
    """확인 안내의 label 이 '맞으면 1번' 처럼 오면 끝의 번호를 떼어 경로에 '맞으면 (1번)' 으로 표시."""
    from gemini_interpreter import clean_label
    assert clean_label("맞으면 1번") == "맞으면"
    assert clean_label("발급을 원하시면 1번") == "발급을 원하시면"
    assert clean_label("분실신고는 별표") == "분실신고"
    assert clean_label("빠른 말 서비스는 샵 버튼") == "빠른 말 서비스"
    assert clean_label("상담직원 연결") == "상담직원 연결"
    assert clean_label("1번") == "1번"  # 전부 지워지면 원문 유지
    text = "입력하신 번호가 맞으면 1번, 아니면 2번을 눌러 주십시오."
    reply = _menu([{"key": "1", "label": "맞으면 1번", "category": "확인"}, {"key": "2", "label": "아니면 2번", "category": "재입력"}])
    reply["kind"] = "confirm"
    p = run(GeminiInterpreter(FakeClient([reply])).classify(text))
    assert [o.label for o in p.options] == ["확인", "다시 입력"]  # 번호 제거 뒤 확인 질문의 동작 이름으로 변환


def test_confirm_labels_become_action_names():
    """확인 질문의 선택지 이름이 조건문 조각이면 동작 이름으로 변환 (경로에 '맞으면 → 발급을 원하시면' 이 보이는 문제 방지)."""
    from gemini_interpreter import confirm_label, confirm_key
    assert confirm_label("맞으면 1번") == "맞으면 1번"       # 끝의 번호는 clean_label 이 먼저 제거
    assert confirm_label("맞으면") == "확인"
    assert confirm_label("틀리면") == "다시 입력"
    assert confirm_label("아니면") == "다시 입력"
    assert confirm_label("발급을 원하시면") == "발급 진행"
    assert confirm_label("변경 확정") == "변경 확정"
    assert confirm_label("맞음") == "확인" and confirm_label("예") == "확인"      # 모델이 조건문을 줄여 보낸 경우
    assert confirm_label("틀림") == "다시 입력" and confirm_label("아니요") == "다시 입력"
    assert confirm_label("취소") == "취소"
    reply = _menu([{"key": "1", "label": "발급을 원하시면 1번"}, {"key": "2", "label": "취소하시려면 2번"}])
    reply["kind"] = "confirm"
    p = run(GeminiInterpreter(FakeClient([reply])).classify("건강보험증 발급을 원하시면 1번, 취소하시려면 2번을 눌러 주십시오."))
    assert [o.label for o in p.options] == ["발급 진행", "취소"]
    assert confirm_key(p, "네") == "1" and confirm_key(p, "아니요") == "2"


def test_transfer_to_another_ars_is_not_agent_wait():
    """'카드 ARS로 연결해 드리겠습니다' 는 사람 상담원이 아닌 다른 자동응답으로 넘기는 안내. 계속 듣기(info) 확인."""
    from gemini_interpreter import rule_classify
    assert rule_classify("카드 ARS로 연결해 드리겠습니다.").kind == "info"
    assert rule_classify("상담원 연결 중입니다. 잠시만 기다려 주십시오.").kind == "agent_wait"
    reply = {"kind": "agent_wait", "options": [], "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None,
             "agent_key": None, "label": None, "terminator": None, "critical": False}
    assert run(GeminiInterpreter(FakeClient([reply])).classify("카드 ARS로 연결해 드리겠습니다.")).kind == "info"


def test_rule_added_hash_option_does_not_trigger_second_score_call():
    """분류 때 받은 점수에 없는 '#' 선택지(규칙으로 채움)가 있어도 score 의 Gemini 재호출 없음 확인."""
    reply = _menu([{"key": "1", "label": "신용 체크카드", "category": "신용·체크카드"}, {"key": "2", "label": "현금카드", "category": "현금카드"}],
                  scores=[{"key": "1", "confidence": 0.9}, {"key": "2", "confidence": 0.1}])
    fake = FakeClient([reply])
    interp = GeminiInterpreter(fake)
    interp.goal = "카드 분실 신고"
    p = run(interp.classify("신용 체크카드는 1번, 현금카드는 2번, 다른 서비스는 샵 버튼을 눌러 주십시오."))
    assert [o.key for o in p.options] == ["1", "2", "#"]           # '#' 은 규칙으로 채움
    s = run(interp.score("카드 분실 신고", p.options))
    assert s == {"1": 0.9, "2": 0.1, "#": 0.0} and len(fake.calls) == 1


# 받아쓰기 고치기 (LLM_FIX_TRANSCRIPT=1) ------------------------------------
def test_apply_fixes_keeps_only_safe_fixes():
    from gemini_interpreter import apply_fixes
    text = "잔액 조회는 삼 번, 거대 내역 조회는 4번, 상담직원 연결은 0번"
    out, applied = apply_fixes(text, [{"from": "거대 내역", "to": "거래 내역"}])
    assert out == "잔액 조회는 삼 번, 거래 내역 조회는 4번, 상담직원 연결은 0번" and applied == [("거대 내역", "거래 내역")]
    rejected = [
        {"from": "4번", "to": "5번"},              # 숫자가 든 고침
        {"from": "삼 번", "to": "사 번"},           # 한국어로 말한 번호가 바뀜
        {"from": "잔액 조회는 삼", "to": "잔액 조회는"},  # 번호가 사라짐
        {"from": "없는 말", "to": "다른 말"},       # 원문에 없음
        {"from": "거대 내역", "to": "거대내역"},     # 띄어쓰기만 다름
        {"from": "거", "to": "거래"},              # 너무 짧음
        {"from": text[:25], "to": "다른 문장"},     # 너무 긺
        {"from": "상담직원", "to": "*"},            # 기호
        "거대→거래", {"from": "거대", "to": ""},    # 형식 오류·빈 값
    ]
    assert apply_fixes(text, rejected) == (text, [])
    assert apply_fixes(text, None) == (text, [])


def test_fix_transcript_corrects_caption_text_and_labels(monkeypatch):
    class SchemaClient(FakeClient):
        async def generate(self, system, user, schema, **kw):
            self.schemas = getattr(self, "schemas", []) + [(system, schema)]
            return await super().generate(system, user, schema, **kw)

    heard = "KB국민은행입니다. 잔액조회는 3번, 거대 내역 조회는 4번, 상담직원 연결은 0번을 눌러 주십시오."
    reply = {"fixes": [{"from": "거대 내역", "to": "거래 내역"}, {"from": "3번", "to": "4번"}], "kind": "menu", "options": [
        {"key": "3", "label": "잔액조회", "category": "잔액 조회"},
        {"key": "4", "label": "거대 내역 조회", "category": "거대 내역 조회"},   # 모델이 label 을 고치지 않아도 코드가 고침
        {"key": "0", "label": "상담직원 연결", "category": "상담원 연결"}],
        "back_key": None, "home_key": None, "repeat_key": None, "agent_key": "0",
        "label": None, "terminator": None, "critical": False, "unnumbered": []}
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "1")
    fake = SchemaClient([dict(reply)])
    p = run(GeminiInterpreter(fake).classify(heard))
    assert [o.label for o in p.options] == ["잔액조회", "거래 내역 조회", "상담직원 연결"]
    assert p.options[1].category == "거래 내역 조회"
    assert p.shown == heard.replace("거대 내역", "거래 내역") and p.fixes == [("거대 내역", "거래 내역")]  # 번호 고침은 버림
    assert p.text == heard                                    # 원문은 그대로 (판단·기록용)
    system, schema = fake.schemas[0]
    assert "fixes" in schema["required"] and "받아쓰기 고치기" in system and "거래" not in system  # 낱말 힌트 없음
    from gemini_interpreter import CLASSIFY
    assert system.startswith(CLASSIFY)  # 분류 규칙은 그대로 (label 규칙 변경으로 확인 질문 options 가 비는 문제 방지)

    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "0")
    fake = SchemaClient([dict(reply)])
    p = run(GeminiInterpreter(fake).classify(heard))
    assert p.shown is None and p.fixes == [] and p.options[1].label == "거대 내역 조회"
    system, schema = fake.schemas[0]
    assert "fixes" not in schema["properties"] and "받아쓰기 고치기" not in system


def test_fix_conflicting_with_text_or_labels_is_dropped(monkeypatch):
    """고친 말이 같은 안내에 이미 있거나, 같은 응답의 선택지 이름과 어긋나는 고침은 제외 확인 (다른 고침은 그대로 적용)."""
    from gemini_interpreter import apply_fixes
    from interpreter import Option
    text = "업무 시간 안내는 5번, 포위는 ARS 이용 안내 7번"
    opts = [Option("5", "업무 시간 안내"), Option("7", "보이는 ARS 이용 안내")]
    # '포위는→문의는' 은 소리 유사도 하한(0.53 < 0.75)에서 먼저 제외. 이름 규칙은 소리가 비슷한 고침으로 시험
    rej: list = []
    assert apply_fixes(text, [{"from": "포위는", "to": "문의는"}], None, rej) == (text, [])
    assert rej == [("포위는", "문의는", "소리")]
    assert apply_fixes(text, [{"from": "포위는", "to": "보위는"}], opts) == (text, [])     # 이름과 어긋남
    assert apply_fixes(text, [{"from": "포위는", "to": "보위는"}])[1] == [("포위는", "보위는")]  # 이름을 안 주면 적용
    opts2 = [Option("5", "업무 시간 안내"), Option("7", "ARS 이용 안내")]                   # 이름이 그 자리를 빼고 시작
    assert apply_fixes(text, [{"from": "포위는", "to": "보위는"}], opts2)[1] == [("포위는", "보위는")]
    a, b = Option("1", "가나 접수"), Option("2", "가다 접수")
    assert apply_fixes("가나 접수 1번, 가다 접수 2번", [{"from": "가다", "to": "가라"}], [a, b])[1] == [("가다", "가라")]
    assert apply_fixes("가나 접수 1번, 가다 접수 2번", [{"from": "가다 접수", "to": "가나 접수"}], [a, b])[1] == []


def test_fix_that_makes_two_options_same_is_dropped(monkeypatch):
    """고친 말이 이미 안내에 있으면(국내 … / 국대 → 국내) 그 고침은 제외, 다른 고침은 적용 확인."""
    heard = "국내 등기 우편 조회는 2번, EMS 국대 등기 우편 조회는 3번, 반품 수추인 접수는 1번을 눌러 주십시오."
    reply = {"fixes": [{"from": "국대", "to": "국내"}, {"from": "수추인", "to": "수취인"}], "kind": "menu", "options": [
        {"key": "2", "label": "국내 등기 우편 조회", "category": "국내 등기 조회"},
        {"key": "3", "label": "EMS 국대 등기 우편 조회", "category": "국제 등기 조회"},
        {"key": "1", "label": "반품 수추인 접수", "category": "반품 접수"}],
        "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
        "label": None, "terminator": None, "critical": False, "unnumbered": []}
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "1")
    p = run(GeminiInterpreter(FakeClient([reply])).classify(heard))
    assert p.fixes == [("수추인", "수취인")]
    assert [o.label for o in p.options] == ["국내 등기 우편 조회", "EMS 국대 등기 우편 조회", "반품 수취인 접수"]
    assert p.shown == heard.replace("수추인", "수취인")


def test_fix_in_key_slot_is_dropped():
    """누르라는 번호 자리의 말은 고치지 않음 확인 (숫자로 인식되지 않은 번호를 낱말로 덮는 문제 방지)."""
    from gemini_interpreter import apply_fixes
    text = "생활정보 홍보 우편 서비스 6번, 인터넷 우체국 최번, 상담사 연결 연거를 눌러주십시오. 수추인 정보 변경"
    fixes = [{"from": "최번", "to": "최선"}, {"from": "연거를", "to": "연결을"}, {"from": "연거", "to": "연결"},
             {"from": "수추인", "to": "수취인"}]
    out, applied = apply_fixes(text, fixes)
    assert applied == [("수추인", "수취인")] and "최번" in out and "연거를" in out


def test_fix_that_repeats_next_word_is_dropped():
    from gemini_interpreter import apply_fixes
    text = "배달 조회, 재배달의 및 배달 장소 지정 2번"
    assert apply_fixes(text, [{"from": "재배달의", "to": "재배달 및"}]) == (text, [])
    assert apply_fixes(text, [{"from": "재배달의 및", "to": "재배달 및"}])[1] == [("재배달의 및", "재배달 및")]


def test_fix_that_adds_words_is_dropped():
    """고친 말이 공백 제외 두 글자 넘게 길어지면 ARS 가 하지 않은 말을 덧붙인 것으로 보고 제외 확인."""
    from gemini_interpreter import apply_fixes
    text = "상담직원 연결은 0번을 눌러주십시오. 다시 듣게는 별표, 이전 단계로 가시려면 샵 버튼을 눌러주십시오."
    assert apply_fixes(text, [{"from": "다시 듣게는", "to": "다시 듣기를 원하시면"}]) == (text, [])
    assert apply_fixes(text, [{"from": "다시 듣게는", "to": "다시 듣기는"}])[1] == [("다시 듣게는", "다시 듣기는")]
    assert apply_fixes("회원가인 안내", [{"from": "회원가인", "to": "회원 가입"}])[1] == [("회원가인", "회원 가입")]  # 공백은 길이에서 제외


def test_grounded_label_reverts_names_not_in_transcript():
    """고치기 사용 시 모델이 fixes 없이 직접 바꾼 이름의 원문 구간 복원 확인 (번역·요약 이름은 그대로)."""
    from gemini_interpreter import _grounded_label
    text = ("말로 하는 AI 상담은 샵 버튼, 증명서 발급은 1번, 가상계좌 등 보험료 문의는 3번, 건강검진, 증생 문의는 4번, "
            "For foreign language service, press 6. 우체국 소포, 국내 등기 우편 조회는 2번, EMS, 국대 등기 우편 조회는 3번")
    assert _grounded_label("건강검진, 증명서 문의", text) == "건강검진, 증생 문의"
    assert _grounded_label("EMS, 국내 등기 우편 조회", text) == "EMS, 국대 등기 우편 조회"
    assert _grounded_label("증명서 발급", text) == "증명서 발급"            # 원문에 있음
    assert _grounded_label("외국어 서비스", text) == "외국어 서비스"        # 비슷한 구간 없음


def test_fix_mode_grounds_labels_then_applies_fixes(monkeypatch):
    heard = "잔액 조회는 3번, 거대 내역 조회는 4번, 건강검진, 증생 문의는 5번을 눌러 주십시오."
    reply = {"fixes": [{"from": "거대 내역", "to": "거래 내역"}], "kind": "menu", "options": [
        {"key": "3", "label": "잔액 조회", "category": "잔액 조회"},
        {"key": "4", "label": "거래 내역 조회", "category": "거래 내역 조회"},     # 모델이 직접 고침 + fixes 있음
        {"key": "5", "label": "건강검진, 증명서 문의", "category": "건강검진"}],   # 모델이 fixes 없이 바꿈
        "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
        "label": None, "terminator": None, "critical": False, "unnumbered": []}
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "1")
    p = run(GeminiInterpreter(FakeClient([dict(reply)])).classify(heard))
    assert [o.label for o in p.options] == ["잔액 조회", "거래 내역 조회", "건강검진, 증생 문의"]
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "0")
    p = run(GeminiInterpreter(FakeClient([dict(reply)])).classify(heard))
    assert p.options[2].label == "건강검진, 증명서 문의"                  # 끈 상태: 모델이 준 이름 그대로


# 받아쓰기 고침만 따로 묻는 호출 (LLM_FIX_TRANSCRIPT=separate) ------------------------------------------
class FixClient:
    """generate 인자 기록, 정한 응답 반환. delay 초 대기 뒤 응답."""

    def __init__(self, replies, delay: float = 0.0, fail: bool = False):
        self.replies, self.delay, self.fail, self.calls = list(replies), delay, fail, []

    async def generate(self, system, user, schema, timeout_s=None, retries=None, rewarm=True, max_tokens=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "timeout_s": timeout_s,
                           "retries": retries, "rewarm": rewarm, "max_tokens": max_tokens})
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("503")
        return self.replies.pop(0)


def test_fix_mode_env_values(monkeypatch):
    import pytest

    from gemini_interpreter import fix_mode_from_env, fix_timeout_from_env
    for raw, want in (("", "0"), ("0", "0"), ("1", "1"), ("separate", "separate"), (" Separate ", "separate")):
        monkeypatch.setenv("LLM_FIX_TRANSCRIPT", raw)
        assert fix_mode_from_env() == want
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "2")
    with pytest.raises(ValueError):
        fix_mode_from_env()
    monkeypatch.setenv("LLM_FIX_TIMEOUT_S", "")
    assert fix_timeout_from_env() == 1.5
    monkeypatch.setenv("LLM_FIX_TIMEOUT_S", "0")
    with pytest.raises(ValueError):
        fix_timeout_from_env()


def test_sound_floor_keeps_similar_and_drops_far_fixes():
    """잘못 들은 말은 소리가 비슷하다는 전제: 자모 유사도 0.75 미만의 고침 제외 확인 (낱말 목록 없는 일반 규칙)."""
    from gemini_interpreter import _sound_ratio, apply_fixes
    text = "이용정지 및 분실 보이스티싱은 1번, 단순무늬 비회원 및 카드 발급 문의는 4번"
    rej: list = []
    out, applied = apply_fixes(text, [{"from": "보이스티싱", "to": "보이스피싱"},
                                      {"from": "단순무늬", "to": "상담원 연결"}], None, rej)
    assert applied == [("보이스티싱", "보이스피싱")] and "보이스피싱은 1번" in out
    assert rej == [("단순무늬", "상담원 연결", "소리")]
    assert _sound_ratio("단순무늬", "단순문의") >= 0.75 and _sound_ratio("양반", "안 됩니다") < 0.75


def test_fix_prompt_separate_call_and_result(monkeypatch):
    """고침만 따로 묻는 호출 확인: 고침 지시문(낱말 힌트 없음)·재시도 없음·다시 올리기 없음·짧은 출력 상한으로 호출,
    입력 선택지는 바꾸지 않고 결과만 반환."""
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    monkeypatch.setenv("LLM_FIX_TIMEOUT_S", "1.5")
    heard = "이용정지 및 분실 보이스티싱은 1번, 카드 문의는 3번을 눌러 주십시오."
    fake = FixClient([{"fixes": [{"from": "보이스티싱", "to": "보이스피싱"}, {"from": "1번", "to": "2번"}]}])
    it = GeminiInterpreter(fake)
    opts = [Option("1", "이용정지 및 분실 보이스티싱"), Option("3", "카드 문의")]
    r = run(it.fix_prompt(heard, opts, "menu"))
    assert r["source"] == "model" and r["applied"] == [("보이스티싱", "보이스피싱")]
    assert r["shown"] == heard.replace("보이스티싱", "보이스피싱")
    assert [x[2] for x in r["rejected"]] == ["길이·숫자"]  # 번호를 바꾸는 고침은 제외
    assert opts[0].label == "이용정지 및 분실 보이스티싱"  # 판단에 쓰는 이름은 그대로
    c = fake.calls[0]
    assert c["retries"] == 0 and c["rewarm"] is False and c["max_tokens"] == 128 and c["timeout_s"] == 2.0
    assert "음성 인식으로 받아 적은 글을 검토" in c["system"] and "보이스" not in c["system"]  # 낱말 힌트 없음
    assert list(c["schema"]["properties"]) == ["fixes"]


def test_fix_prompt_cache_timeout_and_skip(monkeypatch):
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    monkeypatch.setenv("LLM_FIX_TIMEOUT_S", "0.05")
    heard = "거대 내역 조회는 4번, 잔액 조회는 3번"
    fake = FixClient([{"fixes": [{"from": "거대 내역", "to": "거래 내역"}]}, {"fixes": []}])
    it = GeminiInterpreter(fake)
    assert run(it.fix_prompt(heard, [], "menu"))["source"] == "model"
    r = run(it.fix_prompt(heard, [], "menu"))  # 같은 메뉴는 저장한 제안 사용
    assert r["source"] == "cache" and r["applied"] == [("거대 내역", "거래 내역")] and len(fake.calls) == 1
    run(it.fix_prompt("업무 시간 안내입니다", [], "info"))
    r = run(it.fix_prompt("업무 시간 안내입니다", [], "info"))
    assert len(fake.calls) == 3 and r["source"] == "error"  # 일반 안내는 저장하지 않아 재질의 (응답이 없어 실패)
    slow = GeminiInterpreter(FixClient([{"fixes": []}], delay=0.5))
    r = run(slow.fix_prompt("다른 안내 문장입니다", [], "info"))
    assert r["source"] == "timeout" and slow._slow_until == 0.0
    import time as _t
    slow._slow_until = _t.monotonic() + 20
    r = run(slow.fix_prompt("또 다른 안내 문장입니다", [], "info"))
    assert r["source"] == "skipped" and len(slow.client.calls) == 1


def test_separate_mode_classify_is_unchanged(monkeypatch):
    """separate 의 분류는 고침 규칙 없는 기본 분류 지시문·스키마 사용, 선택지 이름 고침 없음 확인."""
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    heard = "잔액조회는 3번, 거대 내역 조회는 4번을 눌러 주십시오."
    reply = {"kind": "menu", "options": [{"key": "3", "label": "잔액조회", "category": "잔액 조회"},
                                         {"key": "4", "label": "거래 내역 조회", "category": "거래내역 조회"}],
             "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
             "label": None, "terminator": None, "critical": False}
    class SchemaClient(FakeClient):
        async def generate(self, system, user, schema, **kw):
            self.schemas = getattr(self, "schemas", []) + [(system, schema)]
            return await super().generate(system, user, schema, **kw)

    fake = SchemaClient([reply])
    p = run(GeminiInterpreter(fake).classify(heard))
    system, schema = fake.schemas[0]
    assert "fixes" not in schema["properties"] and "받아쓰기 고치기" not in system
    assert p.fixes == [] and p.shown is None and p.display_fixes == []
    assert p.options[1].label == "거래 내역 조회"  # 원문 근거 되돌림(_grounded_label)은 LLM_FIX_TRANSCRIPT=1 전용


def test_rule_classify_needs_an_input_request_not_just_the_word():
    """'입력되지 않아'·'입력하신' 만으로는 입력 요청 아님 확인 (종료 안내를 휴대폰 번호 입력 요청으로 오분류하는 문제 방지)."""
    assert rule_classify("휴대폰 번호가 입력되지 않아 자세한 내용을 보내지 못했습니다. 고객님 다른 문의사항이 있으신가요?").kind == "info"
    assert rule_classify("숫자 키패드로 안내받으실 휴대폰 번호를 눌러주세요.").kind == "sensitive"
    assert rule_classify("주민등록번호 13자리를 입력해 주십시오.").kind == "sensitive"
    assert rule_classify("계좌번호를 입력하시고 우물정자를 눌러 주십시오.").kind == "sensitive"
    assert rule_classify("요양 기관 코드 8자리를 입력 후 샵 버튼을 누르십시오.").kind == "sensitive"
    # '누르신 후'·'입력하신 후'·'입력한 다음' 도 입력 요청
    assert rule_classify("주민등록번호를 누르신 후 잠시 기다려 주십시오.").kind == "sensitive"
    assert rule_classify("카드번호를 입력하신 후 잠시 기다려 주십시오.").kind == "sensitive"
    assert rule_classify("휴대폰 번호를 입력한 다음 잠시 기다려 주세요.").kind == "sensitive"
    # 이동 키·상담원 연결 안내의 '눌러' 는 입력 요청 아님. '입력 시간이 초과' 는 입력 오류
    assert rule_classify("휴대폰 번호가 입력되지 않아 안내를 마칩니다. 전 단계로 돌아가시려면 우물정자를 눌러 주세요.").kind == "info"
    assert rule_classify("입력하신 휴대폰 번호는 확인되지 않습니다. 상담원과 통화를 원하시면 0번을 눌러 주세요.").kind != "sensitive"
    assert rule_classify("주민번호 입력 시간이 초과되었습니다.").kind == "input_error"


def test_rule_classify_value_then_terminator_key_is_an_input_request():
    """값을 누른 뒤 끝맺음 키를 누르라는 말은 항목 이름이 목록에 없어도 입력 요청으로 판정 확인
    ('안내 받으실 핸드폰 번호를 누르신 후 우물정자를 눌러주십시오' 를 # 하나짜리 메뉴로 오분류하는 문제 방지)."""
    from jev_interpreter import extract_prompt
    h1 = ("국민건강보험공단 홈페이지에서 나의 보험료 계산하기 서비스 안내입니다. "
          "안내 받으실 핸드폰 번호를 누르신 후 우물정자를 눌러주십시오.")
    for f in (rule_classify, extract_prompt):
        p = f(h1)
        assert p.kind == "sensitive" and p.label == "핸드폰번호" and p.terminator == "#" and not p.options
    p = rule_classify("수신하실 팩스 번호와 샵 버튼을 눌러 주십시오.")
    assert p.kind == "sensitive" and p.label == "팩스번호" and p.terminator == "#"
    p = rule_classify("조회된 정보가 없습니다. 고객님의 팩스 번호를 지역번호와 함께 누르신 후 우물정자를 눌러주십시오.")
    assert p.kind == "sensitive" and p.label == "팩스번호"
    p = rule_classify("조회하실 번호를 입력하시고 별표를 눌러 주세요.")
    assert p.kind == "sensitive" and p.terminator == "*"
    # 끝맺음 키 없이 '누르신 후' 만 있거나, 이동 키·메뉴의 '눌러' 는 입력 요청 아님
    assert rule_classify("확인하신 후 다시 걸어 주시기 바랍니다. 이전 단계로 가시려면 우물정자를 눌러 주세요.").kind == "info"
    assert rule_classify("팩스 발송 결과 조회는 6번, 내 보험료 계산하기는 샵 버튼을 눌러주십시오.").kind == "menu"
    assert rule_classify("입력하신 번호가 맞으면 1번, 틀리면 2번을 누르신 후 우물정자를 눌러 주세요.").kind != "sensitive"


HANA_CARD = ("하나카드사로 연결해 드리겠습니다. 모두의 기쁨 하나카드입니다. 최근 카드사를 사칭한 문자메시지가 발송되고 있습니다. "
             "카드 분실 신고, 해제, 카드 이용 정지, 등록 해제 및 보이스피싱은 1번, 개인 회원은 2번, 가맹점은 3번, "
             "기업 회원은 4번, 신용 대출 및 자동차 금융은 5번을 눌러주세요. 반복 안내는 별표를 눌러주세요.")


def test_model_info_on_menu_ending_with_press_becomes_menu():
    """모델이 일반 안내로 보아도 업무 선택지 둘 이상과 누르라는 말로 끝나는 단락은 메뉴로 판정 확인 (연결 안내 뒤에
    메뉴가 이어진 긴 단락의 일반 안내 오분류 방지). 선택지 하나이거나 누르라는 말이 없는 안내는 그대로."""
    info = {"kind": "info", "options": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
            "label": None, "terminator": None, "critical": False}
    p = run(GeminiInterpreter(FakeClient([dict(info)])).classify(HANA_CARD))
    assert p.kind == "menu" and [o.key for o in p.options] == ["1", "2", "3", "4", "5"] and p.repeat_key == "*"
    one = "자세한 내용은 홈페이지를 참고하시고, 상담원 연결은 0번을 눌러주세요."
    assert run(GeminiInterpreter(FakeClient([dict(info)])).classify(one)).kind == "info"
    talk = "잔액 조회는 3번, 거래내역 조회는 4번 서비스가 새로 바뀌었습니다."
    assert run(GeminiInterpreter(FakeClient([dict(info)])).classify(talk)).kind == "info"


def test_model_out_of_hours_with_self_service_menu_becomes_menu():
    """업무시간 외 안내 뒤에 셀프 메뉴가 이어지면 모델이 업무시간 외로 보아도 메뉴로 판정 확인 (셀프 메뉴를 두고
    통화를 끝내는 문제 방지). 선택지 없는 업무시간 외 안내는 그대로 업무시간 외."""
    ooh = {"kind": "out_of_hours", "options": [], "back_key": None, "home_key": None, "repeat_key": None,
           "agent_key": None, "label": None, "terminator": None, "critical": False}
    menu = ("지금은 셀프 발급 및 조회 업무만 이용 가능합니다. 상담사 연결이 필요하시면 평일 오전 9시부터 오후 6시 사이에 "
            "다시 걸어주시기 바랍니다. 증명서 발급은 1번, 건강보험증 발급은 2번, 고지서 발급은 3번, 다시 듣기는 별표를 눌러주십시오.")
    p = run(GeminiInterpreter(FakeClient([dict(ooh)])).classify(menu))
    assert p.kind == "menu" and [o.key for o in p.options] == ["1", "2", "3"]
    only = "지금은 상담 시간이 아닙니다. 평일 오전 9시부터 오후 6시 사이에 다시 걸어주시기 바랍니다."
    assert run(GeminiInterpreter(FakeClient([dict(ooh)])).classify(only)).kind == "out_of_hours"


def test_model_agent_wait_is_checked_by_model():
    """모델이 상담원 대기라고 답하면 같은 모델에 따로 질의(agent_wait_menu): 선택지 이름 자체가 상담원 연결인 키가 있고, 그 키가
    본문에 있고, 대기 안내가 아니라는 답이면 메뉴로 유지. 선택지는 규칙 후보 추출로 만들고 그 키에 상담원 연결 범주를 붙임(없으면 추가).
    그 밖·실패는 상담원 대기 그대로 (선택지 이름의 '상담원' 만으로는 대기열의 콜백 예약과 구분 불가)."""
    from interpreter import AGENT_CATEGORY
    base = {"options": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
            "label": None, "terminator": None, "critical": False}
    wait = dict(base, kind="agent_wait")

    def check(has, key, waiting):
        return {"has_agent_option": has, "agent_key": key, "waiting": waiting}

    def classify(text, *replies):
        client = FakeClient([dict(wait)] + list(replies))
        return run(GeminiInterpreter(client).classify(text)), client

    p, client = classify("우편 접수는 1번, 배달 조회는 2번, 상담사 연결은 0번을 눌러주십시오.", check(True, "0", False))
    assert p.kind == "menu" and [o.key for o in p.options][:2] == ["1", "2"] and p.agent_key == "0"
    assert next(o for o in p.options if o.key == "0").category == AGENT_CATEGORY and len(client.calls) == 2
    p, _ = classify('소포 접수 1번, 배달 조회, 재배달 신청 2번, 요금 안내 3번, 상담사 연결 0번을 눌러주십시오. 처음부터 다시 들으시려면 별표를 눌러주십시오.', check(True, "0번", False))  # 조사 없는 나열형, 모델이 '0번' 으로 적음
    assert p.kind == "menu" and [o.key for o in p.options] == ["1", "2", "3", "0"] and p.repeat_key == "*"
    p, _ = classify("카드 분실 신고 1번, 결제 대금 조회 2번, 0번을 누르시면 상담원과 연결되며, 다시 듣기는 별표를 눌러주세요.",
                    check(True, "0", False))  # 규칙 후보에 없는 키는 상담원 연결로 추가
    assert p.kind == "menu" and {"1", "2", "0"} <= {o.key for o in p.options}
    assert next(o for o in p.options if o.key == "0").label == "상담원 연결"
    for reply in (check(True, "1", True), check(False, "", True), check(False, "", False), check(True, "7", False)):
        p, _ = classify('상담원과 연결 중입니다. 잠시만 기다려 주십시오. 기다리지 않고 콜백 예약을 원하시면 1번, 문자 상담을 원하시면 2번을 눌러주세요.', reply)  # 대기라고 함·선택지 없음·본문에 없는 키
        assert p.kind == "agent_wait", reply
    p, client = classify('상담원과 연결 중입니다. 잠시만 기다려 주십시오. 기다리지 않고 콜백 예약을 원하시면 1번, 문자 상담을 원하시면 2번을 눌러주세요.')  # 확인 실패(응답 없음)
    assert p.kind == "agent_wait" and len(client.calls) == 2
    # 같은 단락은 한 번만 질의
    client = FakeClient([dict(wait), check(True, "0", False), dict(wait)])
    interp = GeminiInterpreter(client)
    text = "우편 접수는 1번, 배달 조회는 2번, 상담사 연결은 0번을 눌러주십시오."
    assert run(interp.classify(text)).kind == "menu"
    interp._cache.clear()
    assert run(interp.classify(text)).kind == "menu" and len(client.calls) == 3
    # 입력 요청은 질의하지 않음 (상담원 선택지가 붙어 있어도)
    p, client = classify("주민등록번호 13자리를 입력하신 후 우물정자를 눌러 주십시오. 상담원 연결은 0번을 눌러 주십시오.",
                         check(True, "0", False))
    assert p.kind != "menu" and len(client.calls) == 1


def test_menu_overrides_does_not_take_agent_wait():
    """규칙 메뉴로 되돌리는 종류는 일반 안내·업무시간 외만. 상담원 대기는 모델 확인(agent_wait_menu)으로만 메뉴가 됨."""
    menu = "보험료 조회는 1번, 상담사 연결은 0번을 눌러주세요."
    r, n = rule_classify(menu), normalize_keys(menu)
    assert not menu_overrides_kind("agent_wait", r, n)
    assert menu_overrides_kind("info", r, n) and menu_overrides_kind("out_of_hours", r, n)


def test_repeat_announcement_is_a_repeat_key():
    """'반복 안내는 별표' 는 다시 듣기 키로 판정 확인 (업무 선택지 아님)."""
    p = rule_classify("개인 회원은 2번, 가맹점은 3번을 눌러주세요. 반복 안내는 별표를 눌러주세요.")
    assert p.repeat_key == "*" and [o.key for o in p.options] == ["2", "3"]


def test_english_press_key_is_a_key():
    """'for foreign language service, press 9' 의 9 를 키로 판정 확인 (영어 키 표현, 모든 ARS 공통)."""
    from gemini_interpreter import rule_keys
    menu = "폰뱅킹 전용 메뉴는 7번, for foreign language service, press 9. 다시 듣기는 별표를 눌러주십시오."
    t = rule_keys(menu)
    assert "service는 9번" in t and "7번" in t
    assert rule_keys("Press 1 for English") == "Press 1 for English"  # 번호가 앞에 오는 말투는 추정하지 않음
    assert "press 9" in normalize_keys(menu)  # 모델에 보내는 문장은 그대로 유지 (모델 입력 문장 변경은 판단 결과에 영향)


def test_nav_keys_and_info_prompts_carry_navigation_keys():
    from gemini_interpreter import nav_keys
    dead = ("증명서 발급 시 휴대전화 본인인증이 필요합니다. 홈페이지 또는 모바일 앱을 통해 발급받을 수 있습니다. "
            "전 단계로 돌아가시려면 우물정자를 눌러주세요.")
    assert nav_keys(dead) == ("#", None, None)
    assert nav_keys("처음으로 가시려면 9번, 다시 들으시려면 별표를 눌러 주세요.") == (None, "9", "*")
    # 선택지 이름이 앞 문장의 본문 말('처음'·'다시 이용')을 끌어와도 이름 끝의 말로 키 종류 판정
    assert nav_keys("앱을 처음 이용하시는 분은 홈페이지를 참고하세요. 이전 단계로 돌아가시려면 별표를 눌러주세요.") == ("*", None, None)
    assert nav_keys("휴대전화로 다시 이용하시거나 앱을 이용하세요. 다시 들으시려면 9번, 처음으로 가시려면 별표를 눌러주세요.") \
        == (None, "*", "9")
    assert nav_keys("이전 메뉴 다시 듣기는 8번") == (None, None, "8")
    p = run(GeminiInterpreter(FakeClient([{"kind": "info", "options": [], "back_key": None, "home_key": None,
                                           "repeat_key": None, "agent_key": None, "label": None, "terminator": None,
                                           "critical": False, "unnumbered": []}])).classify(dead))
    assert p.kind == "info" and p.back_key == "#"
    p = run(GeminiInterpreter(FakeClient(fail=True)).classify(dead))   # 규칙 분류 대체 경로도 이동 키 포함
    assert p.kind == "info" and p.back_key == "#"


def test_answers_goal_verdicts():
    it = GeminiInterpreter(FakeClient([{"verdict": "answer"}, {"verdict": "other"}, {"verdict": "unsure"}]))
    assert run(it.answers_goal("증명서 발급", "… 앱을 이용하세요. 전 단계는 우물정자", "증명서 발급")) is True
    assert run(it.answers_goal("건강보험증 발급", "…", "증명서 발급")) is False
    assert run(it.answers_goal("건강보험증 발급", "…", "증명서 발급")) is None
    assert run(GeminiInterpreter(FakeClient(fail=True)).answers_goal("목적", "안내")) is None
    assert run(it.answers_goal("", "안내")) is None
    # 답이라면서 가장 맞는 선택지(best)로 다른 선택지를 적으면 판단 불일치로 재질문
    it = GeminiInterpreter(FakeClient([{"best": "건강보험 자격 문의", "verdict": "answer"},
                                       {"best": "증명서 발급", "verdict": "answer"}]))
    sib = ["건강보험 자격 문의", "가상계좌 등 보험료 문의"]
    assert run(it.answers_goal("건강보험증 발급", "…", "증명서 발급", sib)) is None
    assert run(it.answers_goal("증명서 발급받고 싶어", "…", "증명서 발급", sib)) is True

@pytest.mark.parametrize("text", [
    "비밀번호 재설정을 원하시면 1번을 눌러 주십시오.",
    "입력 시간이 초과되었습니다. 비밀번호 변경을 원하시면 1번을 눌러 주십시오.",
    "잘못 누르셨습니다. 휴대폰 번호로 문자를 받으시려면 1번을 눌러 주세요.",
    "비밀번호재설정을원하시면1번을눌러주십시오.",
])
def test_rule_single_option_menu_with_sensitive_word_is_not_input_request(text):
    """선택지 하나짜리 메뉴('…원하시면 1번을 눌러')의 '눌러'는 선택지 고르기라 민감정보 입력 요청 아님."""
    assert rule_classify(text).kind != "sensitive"


@pytest.mark.parametrize("text,label", [
    ("입력 시간이 지났습니다. 지역번호를 포함한 팩스 번호를 누르신 후 우물정자를 눌러주십시오.", "지역번호를 포함한 팩스번호"),
    ("안내 받으실 핸드폰 번호를 누르신 후 우물정자를 눌러 주십시오.", "핸드폰번호"),
    ("고객번호를 누르시고 별표를 눌러 주세요.", "고객번호"),
    ("주민등록번호 앞 6자리를 입력하신 후 1번을 눌러 주세요.", "주민등록번호 앞 6자리"),
    ("가입자 번호를 확인하신 후 번호를 누르시고 우물정자를 눌러 주십시오.", "가입자번호"),
    ("본인확인을위해1번을누르시고주민등록번호를눌러주십시오.", "주민등록번호"),
])
def test_rule_secure_label_near_input_verb(text, label):
    """끝맺음 키 입력 안내의 항목명은 입력 동사 바로 앞의 '…번호' ('X번호를 포함한 Y번호' 는 그대로 담음). 입력 신호가 있으면
    선택지 말이 있어도 민감정보 입력 요청."""
    p = rule_classify(text)
    assert p.kind == "sensitive" and p.label == label

def test_rule_path_restores_direction_particle_and_keeps_dead_end_info():
    """규칙 경로: '처음으로 9번'(은·는 없이)의 '으로' 를 되붙여 처음으로 키로 봄 (입력 요청의 끝맺음 키에서도 뺌).
    목적어가 붙은 다시 듣기만 남은 막다른 안내는 메뉴가 아니라 일반 안내 + 이동 키."""
    from gemini_interpreter import nav_keys, rule_classify
    from jev_interpreter import extract_prompt
    t = "잔액 조회는 1번, 거래내역 조회는 2번, 처음으로 9번, 이전 메뉴는 별표를 눌러주세요."
    r = rule_classify(t)
    assert [o.key for o in r.options] == ["1", "2"] and (r.back_key, r.home_key) == ("*", "9") and nav_keys(t)[:2] == ("*", "9")
    assert rule_classify("주민등록번호 앞 6자리를 입력해 주세요. 처음으로 우물정자를 눌러주세요.").terminator is None
    dead = "해당 업무는 홈페이지에서 신청하실 수 있습니다. 전체 메뉴를 다시 들으시려면 별표를, 처음으로 가시려면 샵을 눌러주세요."
    assert rule_classify(dead).kind == "info" and nav_keys(dead) == (None, "#", "*")
    e = extract_prompt(dead)
    assert e.kind == "info" and (e.home_key, e.repeat_key) == ("#", "*")


@pytest.mark.parametrize("label,kind", [
    ("주거 이전 서비스", None), ("명의 이전 신청", None), ("처음 이용하시는 고객", None), ("번호 이전", None),
    ("이전 메뉴", "back"), ("전단계로 가시려면", "back"), ("이전 단계로 이동하시려면", "back"), ("상위 메뉴", "back"),
    ("처음으로", "home"), ("초기 메뉴로 가시려면", "home"),
    ("다시 들으시려면", "repeat"), ("다시 듣기", "repeat"), ("반복 안내", "repeat"), ("처음부터 다시 들으시려면", "repeat"),
    ("앱을 처음 이용하시는 분은 3번, 전 단계로 돌아가시려", "back"), ("상담원 연결", None),
    # 흔한 이동 말투 (규칙 추출에서 끝의 '면' 이 잘린 꼴 포함)
    ("이전 메뉴로 되돌아가시려", "back"), ("이전 메뉴를 원하시", "back"), ("바로 전 단계", "back"), ("이전 메뉴로 가실 때", "back"),
    ("전 메뉴로 가시려", "back"), ("전단계로 가", "back"), ("이전으로", "back"), ("앞 단계", "back"), ("이전 화면", "back"),
    ("맨 처음으로 돌아가시려", "home"), ("메인 메뉴로 가시려", "home"), ("첫 화면으로", "home"),
    ("다시 듣고 싶으시", "repeat"), ("다시 듣기를 원하시", "repeat"), ("이 안내를 다시 들으시려", "repeat"),
    ("현재 메뉴를 다시 들으시려", "repeat"), ("다시 들어보시려", "repeat"), ("다시 들으", "repeat"),
    # 업무 이름 (이동 말이 들어 있어도)
    ("처음 가입하시려", None), ("처음 가입비", None), ("이전 가입", None), ("이전 가격", None), ("이전으로 가입", None),
    ("이전", None), ("처음", None), ("설치, 이전", None), ("명의, 이전", None), ("재발급 안내", None), ("메뉴얼 문의", None),
    ("이동통신", None), ("전화 상담", None), ("처음 선택", None),
    # 목적어가 붙은 다시 듣기, '반복해서', 격식 어미, '들어가시려면'(다시 듣기 아님), 이어 받아쓴 문장
    ("전체 메뉴를 다시 들으시려", "repeat"), ("안내 멘트를 다시 들으시려", "repeat"), ("방금 안내를 다시 들으시려", "repeat"),
    ("상기 내용을 다시 들으시려", "repeat"), ("반복해서 들으시려", "repeat"), ("반복하여 들으시려", "repeat"),
    ("이전 메뉴로 돌아가고자 하시", "back"), ("다시 들으실 분", "repeat"), ("처음으로 가실 분", "home"),
    ("이전 단계로 돌아갈 때", "back"), ("이전 메뉴로 이동하실 경우에", "back"), ("전 단계로 들어가시려", "back"),
    ("2번을 눌러주세요 이전 메뉴로 가시려", "back"), ("주시기 바랍니다 다시 들으", "repeat"),
    ("이전 안내", None), ("처음 안내", None), ("이전 내용", None), ("이전하시려", None),
])
def test_nav_kind_by_shape(label, kind):
    """이동 키는 이름 전체가 이동 말 단위(대상 말·꾸밈 말·조사·이동·듣기 동사와 어미)일 때만. 업무 이름이 섞이면 업무 선택지.
    대상 말 하나뿐인 이름('이전')은 이동 키로 보지 않음 ('설치는 1번, 이전은 2번' 의 이전)."""
    from gemini_interpreter import _nav_kind
    assert _nav_kind(label) == kind



# 로컬 분류 경로의 이동 키 모양 확인 (merge_nav 와 같은 이름 모양 판정 사용) -------------------------------------------------------
POST5 = ("손해배상제도 6번, 주거 이전 서비스 7번, 국내 우편 반환 및 수추인 정보 변경 신청 8번, 부가 우편 서비스 13번을 눌러주십시오. "
         "다시 들으시려면 별표를 초기 메뉴로 가시려면 9번을 눌러주십시오.")
POST5_OPTS = {"6": "손해배상제도", "8": "국내 우편 반환 및 수취인 정보 변경 신청", "13": "부가 우편 서비스"}


@pytest.mark.parametrize("with_label", [False, True])
def test_model_nav_key_with_business_name_stays_business_option(with_label):
    """모델(로컬 분류)이 '주거 이전 서비스 7번' 을 이전 키로 적어도 업무 선택지로 유지.
    모델이 7 을 선택지에서 뺐든(이름은 안내 원문에서 찾음) 이름까지 적었든 같음. 이름 모양으로 정해진 처음으로(9)·다시 듣기(*)는 그대로."""
    opts = dict(POST5_OPTS, **({"7": "주거 이전 서비스"} if with_label else {}))
    p = run(GeminiInterpreter(FakeClient([reply(opts, None, back="7", rep="*") | {"home_key": "9"}])).classify(POST5))
    assert p.kind == "menu"
    assert {o.key: o.label for o in p.options}["7"] == "주거 이전 서비스"
    assert p.back_key is None and p.home_key == "9" and p.repeat_key == "*"


def test_model_nav_key_cleared_route_takes_shaped_rule_key():
    """잘못 고른 이전 키를 비운 칸은 이름 모양으로 찾은 같은 종류 키('전단계로 가시려면 우물정자')로 채움."""
    text = POST5 + " 전단계로 가시려면 우물정자를 눌러주십시오."
    p = run(GeminiInterpreter(FakeClient([reply(POST5_OPTS, None, back="7", rep="*") | {"home_key": "9"}])).classify(text))
    assert "7" in [o.key for o in p.options] and "#" not in [o.key for o in p.options]
    assert p.back_key == "#" and p.home_key == "9"


@pytest.mark.parametrize("text,back", [
    ("잔액 조회는 1번, 이전 메뉴는 별표, 처음으로 가시려면 9번을 눌러주세요.", "*"),
    ("잔액 조회는 1번, 이전은 별표를 눌러주세요.", "*"),                 # 대상 말 하나뿐인 이름: 모양으로 정할 수 없어 모델 판단
    ("잔액 조회는 1번을 눌러주세요. 별표를 누르시면 이전 메뉴로 갑니다.", "*"),  # 이름을 찾지 못한 키: 모델 판단 그대로
])
def test_model_nav_key_with_nav_name_is_kept(text, back):
    p = run(GeminiInterpreter(FakeClient([reply({"1": "잔액 조회"}, None, back=back)])).classify(text))
    assert p.back_key == back and [o.key for o in p.options] == ["1"]


def test_model_nav_key_business_names_for_other_words():
    """업무 이름에 이동 말('이전'·'처음')이 든 선택지는 모델이 이동 키로 적어도 업무 선택지로 남음."""
    text = ("사업장 이전 신고는 1번, 명의 이전은 2번, 번호 이전은 3번, 처음 이용하시는 고객은 4번, 처음 가입은 5번을 눌러주세요. "
            "이전 메뉴로 가시려면 별표를 눌러주세요.")
    for wrong in ("1", "2", "3", "4", "5"):
        for field in ("back", "home"):
            d = reply({"1": "사업장 이전 신고", "2": "명의 이전", "3": "번호 이전", "4": "처음 이용하시는 고객", "5": "처음 가입"}, None)
            d[f"{field}_key"] = wrong
            p = run(GeminiInterpreter(FakeClient([d])).classify(text))
            assert sorted(o.key for o in p.options) == ["1", "2", "3", "4", "5"], (wrong, field)
            assert p.back_key in (None, "*") and p.home_key is None, (wrong, field, p.back_key, p.home_key)


# 모양으로 정할 수 없는 이름은 모델 판단 유지, 되살린 선택지 점수 재질의, 후보 이름 앞 조사 ----------------------
@pytest.mark.parametrize("text,field,key", [
    ("잔액 조회는 1번, 카드 분실 신고는 2번을 눌러주세요. 안내 말씀을 다시 들으시려면 별표를 눌러주세요.", "repeat_key", "*"),
    ("잔액 조회는 1번, 카드 분실 신고는 2번을 눌러주세요. 음성 안내를 다시 들으시려면 별표를 눌러주세요.", "repeat_key", "*"),
    ("잔액 조회는 1번, 카드 분실 신고는 2번을 눌러주세요. 홈 메뉴로 가시려면 9번을 눌러주세요.", "home_key", "9"),
    ("잔액 조회는 1번, 카드 분실 신고는 2번을 눌러주세요. 이전 메누로 가시려면 별표를 눌러주세요.", "back_key", "*"),
    ("잔액 조회는 1번, 카드 분실 신고는 2번을 눌러주세요. 다시 듯기는 별표를 눌러주세요.", "repeat_key", "*"),
    ("1번 잔액 조회 2번 거래내역 조회 3번 카드 분실 신고 9번 처음으로 이동을 눌러주세요.", "home_key", "9"),
    ("잔액 조회는 1번, 카드 분실 신고는 2번을 눌러주세요 처음부터 다시 들으시려면 별표", "repeat_key", "*"),
])
def test_model_nav_key_kept_when_name_shape_cannot_decide(text, field, key):
    """규칙 어휘 밖의 말투·오타, 번호가 앞에 오는 말투(이름이 한 칸 밀림), 구두점 없이 앞 문장이 붙은 이름은 모양으로 정할 수 없어
    모델의 이동 키 판단을 그대로 사용 (업무 선택지로 돌리지 않음)."""
    d = reply({"1": "잔액 조회", "2": "카드 분실 신고" if "3번" not in text else "거래내역 조회"}, None)
    if "3번" in text:
        d["options"].append({"key": "3", "label": "카드 분실 신고", "category": "카드 분실 신고"})
    d[field] = key
    p = run(GeminiInterpreter(FakeClient([d])).classify(text))
    assert getattr(p, field) == key and key not in [o.key for o in p.options]


def test_model_nav_key_on_info_paragraph_is_kept():
    """일반 안내(막다른 안내)에서는 모델이 적은 이동 키를 그대로 유지: 구두점 없이 앞 안내가 이름에 붙어도 이동 키를 지우거나
    선택지를 만들지 않음 (지우면 막다른 안내 판단을 잃음)."""
    text = ("건강보험 자격득실 확인서는 홈페이지에서 간편하게 발급받을 수 있습니다 전 단계로 돌아가시려면 우물정자를 눌러주세요")
    d = reply({}, None, kind="info", back="#")
    p = run(GeminiInterpreter(FakeClient([d])).classify(text))
    assert p.kind == "info" and p.back_key == "#" and p.options == []


def test_revived_business_option_gets_model_score_again():
    """이동 키에서 업무 선택지로 돌린 키는 분류 때 점수가 없으므로 캐시하지 않고, 점수를 모델에 재질의."""
    d = reply(POST5_OPTS, None, scores={"6": 0.05, "8": 0.3, "13": 0.2}, back="7", rep="*") | {"home_key": "9"}
    fake = FakeClient([d, {"scores": [{"key": "6", "confidence": 0.0}, {"key": "8", "confidence": 0.3},
                                      {"key": "13", "confidence": 0.1}, {"key": "7", "confidence": 0.95}]}])
    interp = GeminiInterpreter(fake)
    goal = "이사가는데 우편물을 새 주소로 받는 방법 알려줘"
    interp.goal = goal
    p = run(interp.classify(POST5))
    assert "7" in [o.key for o in p.options]
    s = run(interp.score(goal, p.options))
    assert len(fake.calls) == 2 and s["7"] == 0.95


def test_candidate_label_drops_leading_object_particle():
    """구두점 없이 앞 키 표현의 조사만 남은 후보 이름('를 주거 이전 서비스')은 조사 제거."""
    from jev_interpreter import candidate_options
    text = "손해배상제도 6번, 국내 우편 반환 신청 8번을 눌러주십시오. 다시 들으시려면 별표를 주거 이전 서비스는 7번을 눌러주십시오."
    assert candidate_options(text)["7"] == "주거 이전 서비스"
    d = reply({"6": "손해배상제도", "8": "국내 우편 반환 신청"}, None, back="7", rep="*")
    p = run(GeminiInterpreter(FakeClient([d])).classify(text))
    assert {o.key: o.label for o in p.options}["7"] == "주거 이전 서비스" and p.back_key is None


@pytest.mark.parametrize("name", ["주거이전 서비스", "주거이전서비스"])
def test_model_nav_key_with_joined_business_name_stays_business_option(name):
    """붙여 쓴 업무 이름('주거이전 서비스'·'주거이전서비스')도 모델이 이전 키로 적으면 업무 선택지로 유지 (JEV 경로와 같은 결론)."""
    from jev_interpreter import extract_prompt
    text = POST5.replace("주거 이전 서비스", name)
    p = run(GeminiInterpreter(FakeClient([reply(POST5_OPTS, None, back="7", rep="*") | {"home_key": "9"}])).classify(text))
    assert "7" in [o.key for o in p.options] and p.back_key is None
    j = extract_prompt(text, {"back": "7", "home": "9", "repeat": "*"})
    assert "7" in [o.key for o in j.options] and j.back_key is None


@pytest.mark.parametrize("key", ["3", "#", "*"])
def test_revived_option_is_rescored_once_and_cached(key):
    """되살린 업무 선택지는 '*'·'#' 키여도 점수를 모델에 다시 묻고, 받은 점수는 캐시에 넣어 같은 메뉴의 다음 판단에서 모델을 또
    부르지 않음."""
    word = {"3": "3번", "#": "샵", "*": "별표"}[key]
    text = (f"카드 분실 신고는 1번, 결제 대금 조회는 2번, 처음 방문하신 고객은 {word}를 눌러주세요. "
            "이전 메뉴는 9번을 눌러주세요.")
    d = reply({"1": "카드 분실 신고", "2": "결제 대금 조회"}, None, scores={"1": 0.05, "2": 0.05})
    d["home_key"] = key
    d["back_key"] = "9"
    fake = FakeClient([d, {"scores": [{"key": "1", "confidence": 0.0}, {"key": "2", "confidence": 0.0},
                                      {"key": key, "confidence": 0.95}]}])
    interp = GeminiInterpreter(fake)
    interp.goal = "처음 방문했는데 회원 가입하고 싶어요"
    p = run(interp.classify(text))
    assert key in [o.key for o in p.options] and p.home_key is None
    s1 = run(interp.score(interp.goal, p.options))
    s2 = run(interp.score(interp.goal, p.options))
    assert s1[key] == 0.95 and s2[key] == 0.95 and len(fake.calls) == 2


@pytest.mark.parametrize("text,field,key", [
    ("잔액 조회는 1번, 거래 내역 조회는 2번을 눌러주세요. 이전매뉴는 별표를 눌러주세요.", "back_key", "*"),
    ("잔액 조회는 1번, 거래 내역 조회는 2번을 눌러주세요. 처음매뉴는 9번을 눌러주세요.", "home_key", "9"),
    ("잔액 조회는 1번, 거래 내역 조회는 2번을 눌러주세요. 상위매뉴는 별표를 눌러주세요.", "back_key", "*"),
    ("잔액 조회는 1번, 거래 내역 조회는 2번을 눌러주세요. 제일처음으로는 9번을 눌러주세요.", "home_key", "9"),
])
def test_model_nav_key_kept_for_typo_joined_nav_names(text, field, key):
    """이동 말로 시작하는 붙은 이름(이동 키 이름의 오타 '이전매뉴'·'처음매뉴'·'상위매뉴', '제일처음으로')은 규칙이 같은 종류의
    다른 이동 키를 찾지 못하면 모델 판단을 그대로 사용."""
    d = reply({"1": "잔액 조회", "2": "거래 내역 조회"}, None)
    d[field] = key
    p = run(GeminiInterpreter(FakeClient([d])).classify(text))
    assert getattr(p, field) == key and key not in [o.key for o in p.options]


def test_model_nav_key_joined_business_name_starting_with_nav_word_needs_rule_evidence():
    """'인터넷 이전설치' 처럼 이동 말로 시작하는 붙은 업무 이름은, 규칙이 같은 종류의 다른 이동 키('이전 메뉴로 가시려면 별표')를
    찾았으면 업무 선택지로 유지."""
    text = "인터넷 신규 가입은 1번, 인터넷 이전설치는 2번을 눌러주세요. 이전 메뉴로 가시려면 별표를 눌러주세요."
    d = reply({"1": "인터넷 신규 가입"}, None, back="2")
    p = run(GeminiInterpreter(FakeClient([d])).classify(text))
    assert "2" in [o.key for o in p.options] and p.back_key == "*"


def test_agent_wait_menu_keeps_critical_confirm_and_reads_key_words():
    """상담원 대기 확인으로 만든 메뉴: 해석기의 중요 선택 표시를 남기고, 확인 질문 꼴이면 확인 질문으로 두며, 모델이 키를 '영번'·
    '영'·'별표' 처럼 적어도 본문 키로 풀고, 이름을 적으면 규칙 후보 추출이 상담원 연결로 본 키로 대체."""
    from interpreter import Prompt

    def ask(text, reply, prompt=None):
        interp = GeminiInterpreter(FakeClient([reply]))
        return run(interp.agent_wait_menu(text, prompt))

    yes = lambda k: {"has_agent_option": True, "agent_key": k, "waiting": False}
    text = "보험 청약을 확정하시려면 1번, 상담사 연결은 0번을 눌러주세요."
    m = ask(text, yes("0"), Prompt(kind="agent_wait", text=text, critical=True))
    assert m.kind == "menu" and m.critical and m.agent_key == "0"
    conf = "카드 재발급을 신청하시겠습니까? 맞으면 1번, 틀리면 2번, 상담원 연결은 0번을 눌러주세요."
    assert ask(conf, yes("0")).kind == "confirm"
    menu = "우편 접수는 1번, 배달 조회는 2번, 상담사 연결은 영번을 눌러주십시오."
    for k in ("영번", "영", "0번", "상담원 연결"):
        m = ask(menu, yes(k))
        assert m is not None and m.agent_key == "0" and "0" in {o.key for o in m.options}, k
    star = "잔액 조회는 1번, 이체는 2번, 상담원 연결은 별표를 눌러주세요."
    assert ask(star, yes("별표")).agent_key == "*"
