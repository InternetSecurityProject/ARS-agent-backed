"""hybrid_interpreter 시험 (실제 LLM·JEV 호출 없음, 가짜 클라이언트).

판단 정확도·속도는 bench/menu_eval.py 로 측정. 이 시험은 로컬·JEV 분배 규칙과 상태 이어 받기만 확인.
python -m pytest -q test_hybrid_interpreter.py
"""
from __future__ import annotations

import asyncio

import pytest

from hybrid_interpreter import HybridJevInterpreter, mask_goal, private_reason
from jev_interpreter import JevInterpreter
from local_llm import LocalLlmInterpreter

MENU = "KB국민은행입니다. 잔액조회는 3번, 거래내역조회는 4번, 상담직원 연결은 0번을 눌러 주십시오."
LOCAL_MENU = {"kind": "menu", "options": [{"key": "3", "label": "잔액조회", "category": "잔액 조회"},
                                          {"key": "4", "label": "거래내역조회", "category": "거래내역 조회"}],
              "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": None,
              "label": None, "terminator": None, "critical": False,
              "scores": [{"key": "3", "confidence": 0.9}, {"key": "4", "confidence": 0.1}]}


def run(coro):
    return asyncio.run(coro)


class FakeLocal:
    """LocalLlmInterpreter 의 client 대역. 받은 요청을 seen 에 수집."""

    def __init__(self, reply=None):
        self.seen: list[str] = []
        self.reply = reply or LOCAL_MENU
        self.fixes: list[dict] = []      # 받아쓰기 고침 호출(따로)에 돌려줄 제안
        self.fix_seen: list[str] = []    # 받아쓰기 고침 호출로 받은 요청
        self.goal_done = {"best": "증명서 발급", "verdict": "answer"}  # 막다른 안내 판단(answers_goal)에 돌려줄 답
        self.goal_done_seen: list[str] = []

    async def generate(self, system, user, schema, timeout_s=None, retries=None):
        self.seen.append(user)
        if set(schema.get("properties", {})) == {"best", "verdict"}:
            self.goal_done_seen.append(user)
            return dict(self.goal_done)
        if set(schema.get("properties", {})) == {"fixes"}:
            self.fix_seen.append(user)
            return {"fixes": list(self.fixes)}
        if "text" in schema.get("properties", {}) and "kind" not in schema.get("properties", {}):
            return {"text": "목적 요약"}
        if set(schema.get("properties", {})) == {"key"}:
            return {"key": "1"}
        return dict(self.reply)


class FakeJev:
    """JevClient 대역. 질문 이름에 맞춘 분명한 답 반환."""

    def __init__(self, pick="3", fail=False, kind="menu"):
        self.states: list[dict] = []
        self.pick, self.fail, self.kind = pick, fail, kind

    async def decide(self, state, questions):
        self.states.append(state)
        if self.fail:
            raise TimeoutError()
        out = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                out[name] = {"type": "noul", "noul": 0.0 if name == "critical" else 0.95}
                continue
            crit = list(q["criteria"])
            want = {"kind": self.kind, "route_status": "proceed", "matching_key": "dtmf_" + self.pick,
                    "confirm_key": "key_1"}.get(name, crit[0])
            want = want if want in crit else crit[0]
            out[name] = {"type": "choice", "choice": want, "confidence": 0.95,
                         "probabilities": {c: (0.95 if c == want else 0.05 / max(1, len(crit) - 1)) for c in crit}}
        return out


def make(pick="3", fail=False, reply=None, kind="menu"):
    local, jev = FakeLocal(reply), FakeJev(pick, fail, kind)  # 받아쓰기 고침 방식은 LocalLlmInterpreter 가 환경 변수에서 읽음
    h = HybridJevInterpreter(LocalLlmInterpreter(local), JevInterpreter(jev))
    h.bench_reset("잔액 조회")
    return h, local, jev


# 민감정보 판별 -------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    MENU,
    "원하시는 서비스 번호를 눌러 주세요. 상담원 연결은 0번입니다.",
    "고객님, 반갑습니다. 카드 분실신고는 1번, 이전 메뉴는 별표를 눌러 주세요.",
    "업무 시간은 평일 9시부터 6시까지입니다.",
    "등록된 주소가 맞으면 1번, 틀리면별표를 눌러 주세요.",  # '별표' → '*' 변환 뒤에도 가린 이름으로 보지 않음
])
def test_plain_menus_are_not_private(text):
    assert private_reason(text) is None


@pytest.mark.parametrize("text,reason", [
    ("입력하신 번호는 **** 맞으면 1번, 틀리면 2번", "가린 입력값·이름"),
    ("홍*동 고객님의 요청이 접수되었습니다", "가린 입력값·이름"),
    ("입력하신 번호가 맞으시면 1번을 눌러 주세요", "입력값 되읽기"),
    ("공일공 일이삼사 오육칠팔 번호로 보내 드립니다", "한국어 숫자열"),
    ("대표번호 1588-9999 로 문의해 주세요", "네 자리 이상 숫자"),
    ("출금 가능액은 36만 원입니다", "금액"),
    ("미납 금액은 삼만 원이며", "금액"),
    ("김철수 고객님 안녕하십니까", "이름 호칭"),
    ("신청이 접수되었습니다", "결과 안내"),
])
def test_private_texts_are_detected(text, reason):
    assert private_reason(text) == reason


def test_goal_numbers_are_masked_for_jev():
    assert mask_goal("010-1234-5678 로 문자 받기") == "*********** 로 문자 받기"
    assert mask_goal("공일공 일이삼사 오육칠팔 번호 변경") == "*********** 번호 변경"
    assert mask_goal("3번 잔액 조회") == "3번 잔액 조회"


# 나누기 ---------------------------------------------------------------------------
def test_plain_menu_goes_to_jev_only():
    h, local, jev = make()
    p = run(h.classify(MENU))
    assert h.last_route == ("jev", None) and len(jev.states) == 1 and not local.seen
    assert [o.key for o in p.options] == ["3", "4", "0"]
    assert jev.states[0]["current_ars_transcript"].startswith("KB국민은행")


def test_private_text_goes_to_local_only():
    h, local, jev = make()
    run(h.classify("김철수 고객님, 잔액조회는 3번, 거래내역조회는 4번을 눌러 주십시오."))
    assert h.last_route == ("local", "이름 호칭") and local.seen and not jev.states


def test_after_input_request_everything_is_local_until_next_call():
    h, local, jev = make()
    p = run(h.classify("주민등록번호 13자리를 입력해 주세요."))
    assert p.kind == "sensitive" and not jev.states  # 입력 요청은 규칙으로 잡혀 JEV 호출 없음
    run(h.classify(MENU))
    assert h.last_route == ("local", "민감정보 입력 요청 뒤") and not jev.states
    run(h.summarize_goal("거래내역 조회"))  # 같은 통화의 목적 다시 말하기: 해제되지 않음
    run(h.classify(MENU))
    assert h.last_route == ("local", "민감정보 입력 요청 뒤") and not jev.states
    h.start_call()  # 새 통화 (Session 이 목적 요약보다 먼저 호출)
    run(h.summarize_goal("잔액 조회"))
    run(h.classify(MENU))
    assert h.last_route[0] == "jev" and len(jev.states) == 1


def test_goal_sent_to_jev_is_masked():
    h, local, jev = make()
    run(h.summarize_goal("010-1234-5678 번호로 잔액 문자 받기"))
    run(h.classify(MENU))
    assert "1234" not in jev.states[0]["user_goal"] and "*" in jev.states[0]["user_goal"]


# 점수·확인 질문은 분류한 쪽에서 ----------------------------------------------------------
def test_jev_scores_from_classify_are_reused_even_if_path_changed():
    h, local, jev = make(pick="4")
    p = run(h.classify(MENU))
    s = run(h.score("잔액 조회", p.options, ["카드"]))
    assert len(jev.states) == 1 and s["4"] > 0.8 and s["3"] < 0.1


def test_local_menu_is_scored_locally():
    h, local, jev = make()
    p = run(h.classify("김철수 고객님, 잔액조회는 3번, 거래내역조회는 4번을 눌러 주십시오."))
    s = run(h.score("잔액 조회", p.options, []))
    assert s["3"] == pytest.approx(0.9) and not jev.states


def test_unknown_options_are_scored_locally():
    h, local, jev = make()
    from interpreter import Option
    run(h.score("잔액 조회", [Option("1", "처음 보는 선택지")], []))
    assert not jev.states


def test_confirm_of_local_prompt_never_goes_to_jev():
    reply = dict(LOCAL_MENU, kind="confirm", options=[{"key": "1", "label": "맞으면"}, {"key": "2", "label": "틀리면"}],
                 scores=None)
    h, local, jev = make(reply=reply)
    p = run(h.classify("입력하신 번호가 맞으면 1번, 틀리면 2번"))
    assert p.kind == "confirm"
    run(h.answer_confirm("잔액 조회", p))
    assert not jev.states


def test_jev_confirm_uses_one_choice_question():
    h, local, jev = make(kind="confirm")
    p = run(h.classify("등록된 주소로 발송합니다. 맞으시면 1번, 틀리시면 2번을 눌러 주세요."))
    assert p.kind == "confirm" and h.last_route[0] == "jev"
    assert run(h.answer_confirm("건강보험증 발급", p)) == "1"
    assert set(jev.states[-1]) == {"current_ars_transcript", "user_goal", "options"} and not local.seen


def test_critical_jev_confirm_is_never_answered():
    h, local, jev = make(kind="confirm")
    p = run(h.classify("등록된 주소로 발송합니다. 맞으시면 1번, 틀리시면 2번을 눌러 주세요."))
    p.critical = True
    n = len(jev.states)
    assert run(h.answer_confirm("건강보험증 발급", p)) is None and len(jev.states) == n


def test_reask_answer_reaches_jev_confirm_rule():
    h, local, jev = make(kind="confirm")
    p = run(h.classify("등록된 주소로 발송합니다. 맞으시면 1번, 틀리시면 2번을 눌러 주세요."))
    run(h.merge_answer("건강보험증 발급", "주소가 맞나요?", "아니요"))
    n = len(jev.states)
    assert run(h.answer_confirm("건강보험증 발급", p)) == "2" and len(jev.states) == n  # 규칙(네/아니요)으로 답함


def test_jev_failure_falls_back_to_rule_options_and_ask_scores():
    h, local, jev = make(fail=True)
    p = run(h.classify(MENU))
    assert [o.key for o in p.options] == ["3", "4", "0"]  # 규칙으로 뽑은 선택지
    s = run(h.score("잔액 조회", p.options, []))
    assert set(s.values()) == {0.4}  # 재질문 구간 (0.3~0.5)
    assert not local.seen


# jev_interpreter 규칙 -------------------------------------------------
def test_word_containing_ijeon_is_not_back_key():
    from jev_interpreter import extract_prompt
    p = extract_prompt("손해배상제도는 6번, 주거이전 서비스는 7번, 이전 단계는 샵, 처음으로는 9번을 눌러주십시오")
    assert [o.key for o in p.options] == ["6", "7"] and p.back_key == "#" and p.home_key == "9"


def test_jev_rule_extraction_on_live_menu_shapes():
    """실제 메뉴 형태의 규칙 추출 확인: 쉼표로 묶은 선택지 이름 전체 유지, 전화번호 끝자리('…-1111번')는 키 아님,
    '반복 안내' 는 업무 선택지가 아닌 이동 키, 영어 'press 9' 를 키로 읽음. 번호가 앞에 오는 말투·앞 문장 조각은 이름에서 제외."""
    from jev_interpreter import extract_prompt
    p = extract_prompt("국민은행으로 송금은 1번, 다른 은행으로 송금은 2번, 자동이체, 지로, 대학 등록금, 국세납부, 예약이체, "
                       "지연이체는 4번, 다시 듣기는 별표, 이전 단계는 샵 버튼을 눌러주십시오.")
    assert [(o.key, o.label) for o in p.options] == [("1", "국민은행으로 송금"), ("2", "다른 은행으로 송금"),
                                                     ("4", "자동이체, 지로, 대학 등록금, 국세납부, 예약이체, 지연이체")]
    p = extract_prompt("하나카드사 전원연결이 안될 경우 1800-1111번을 이용해 주시기 바랍니다. 개인 회원은 2번, 가맹점은 3번을 "
                       "눌러주세요. 반복 안내는 별표를 눌러주세요.")
    assert [o.key for o in p.options] == ["2", "3"] and p.repeat_key == "*"
    p = extract_prompt("잔액 조회는 3번, 폰뱅킹 전용 메뉴는 7번, for foreign language service, press 9. 다시 듣기는 별표를 눌러주십시오.")
    assert [(o.key, o.label) for o in p.options] == [("3", "잔액 조회"), ("7", "폰뱅킹 전용 메뉴"),
                                                     ("9", "for foreign language service")]
    p = extract_prompt("1번 잔액조회, 2번 거래내역 조회를 눌러 주세요.")       # 번호가 앞에 오는 말투: 이름 짐작 없음
    assert all(o.label != "잔액조회" or o.key != "2" for o in p.options)
    p = extract_prompt("평일 9시부터 6시까지, 상담원 연결은 0번, 고객님, 잔액조회는 1번을 눌러주세요.")
    assert {o.key: o.label for o in p.options} == {"0": "상담원 연결", "1": "잔액조회"}


def test_live_label_and_route_fixes():
    """선택지 이름의 서술격 조건 끝말 제거, 입력을 시키는 '누르신 후' 를 되읽기로 보지 않음,
    입력 요청 이름에 '앞 6자리 또는 사업자등록번호' 같은 부분 유지 확인."""
    from gemini_interpreter import rule_classify
    from jev_interpreter import extract_prompt
    p = extract_prompt("가지고 계신 신분증이 주민등록증이면 1번, 운전면허증이면 2번을 눌러주세요.")
    assert [o.label for o in p.options] == ["주민등록증", "운전면허증"]
    p = extract_prompt("개인 고객이시라면 1번, 법인 사업자면 2번을 눌러주세요.")
    assert [o.label for o in p.options] == ["개인 고객", "법인 사업자"]
    # 한계: '어린이면' 처럼 '…이' 로 끝나는 말 뒤의 '면' 은 서술격과 구별 불가, '어린' 으로 남음
    assert private_reason("본인 확인을 위해 팩스 번호를 지역번호와 함께 누르신 후 우물정자를 눌러주십시오.") is None
    assert private_reason("입력하신 번호가 맞으시면 1번, 틀리시면 2번을 눌러주세요.") == "입력값 되읽기"
    p = rule_classify("손님의 주민등록번호 앞 6자리 또는 사업자등록번호 10자리와 샵 버튼을 눌러주십시오.")
    assert p.kind == "sensitive" and p.label == "주민등록번호 앞 6자리 또는 사업자등록번호 10자리"
    p = rule_classify("고객님의 주민등록번호 앞 6자리와 샵 버튼 또는 사업자등록번호 10자리와 샵 버튼을 눌러주십시오. "
                      "다른 서비스는 샵 버튼을 눌러주십시오.")                       # 입력 방법 둘과 다른 서비스 안내가 이어진 문장 형태
    assert p.label == "주민등록번호 앞 6자리 또는 사업자등록번호 10자리" and p.terminator == "#"
    p = rule_classify("주민등록번호 앞 6자리와 샵 버튼 또는 사용자 번호와 샵 버튼을 눌러주십시오.")
    assert p.label == "주민등록번호 앞 6자리 또는 사용자번호"
    assert rule_classify("고객님의 주민등록번호 앞 여섯자리와 샵 버튼을 눌러주십시오.").label == "주민등록번호 앞 여섯자리"
    assert rule_classify("숫자 키패드로 안내받으실 휴대폰 번호를 눌러주세요.").label == "휴대폰번호"


def test_jev_info_on_menu_ending_with_press_stays_menu():
    """JEV 가 일반 안내라고 해도 업무 선택지 둘 이상과 누르라는 말로 끝나는 단락의 메뉴 유지 확인 (로컬과 같은 기준)."""
    h, local, jev = make(kind="info")
    p = run(h.classify("하나카드사로 연결해 드리겠습니다. 개인 회원은 2번, 가맹점은 3번, 기업 회원은 4번을 눌러주세요."))
    assert p.kind == "menu" and [o.key for o in p.options] == ["2", "3", "4"]
    p = run(h.classify("하나카드사로 연결해 드리겠습니다. 잠시만 기다려 주세요."))
    assert p.kind == "info"


def test_rule_critical_can_be_turned_off(monkeypatch):
    from jev_interpreter import extract_prompt
    text = "카드 발급은 1번, 신규 가입을 원하시면 2번을 눌러주세요"
    assert extract_prompt(text).critical is True  # 기본: 중요 선택 규칙 적용 (가입 + 원하시면)
    monkeypatch.setenv("JEV_RULE_CRITICAL", "0")
    assert extract_prompt(text).critical is False


def test_rule_critical_ignores_certificate_names():
    """'확인증'·'확인서' 의 '확인' 을 승인 말로 보지 않음 확인 (조회 메뉴 전체를 중요 선택으로 묻는 오동작 방지).
    승인 말로 쓴 '확인' 은 그대로 판정."""
    from jev_interpreter import extract_prompt
    menu = ("거래내역 조회 서비스입니다. 무통장 거래내역 조회는 1번, 일자별 거래내역 조회는 2번, 거래내역을 이메일 팩스로 받기는 3번, "
            "송금 확인증을 이메일 팩스로 받기는 4번, 통장 사본을 팩스로 받기는 5번, 상담원 연결은 0번을 눌러주십시오.")
    assert extract_prompt(menu).critical is False
    assert extract_prompt("송금 확인서 발급은 1번을 눌러주십시오.").critical is False
    assert extract_prompt("송금을 확인하시려면 1번, 취소하시려면 2번을 눌러주십시오.").critical is True


# 받아쓰기 고침 따로 묻기 (LLM_FIX_TRANSCRIPT=separate) --------------------------------------------
HEARD = "KB국민은행입니다. 보이스티싱 신고는 1번, 잔액조회는 3번을 눌러 주십시오."   # 받아쓰기 오인식 예
FIX = {"from": "보이스티싱", "to": "보이스피싱"}


def test_fix_mode_comes_from_local_setting(monkeypatch):
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    h, local, jev = make()
    assert h.fix_mode == "separate" and getattr(h, "fix_mode", "0") == "separate"  # Session 쪽 getattr 읽기와 같은 방식 확인
    h.fix_mode = "0"
    assert h.local.fix_mode == "0" and h.local.fix_transcript is False
    h.fix_mode = "1"
    assert h.local.fix_transcript is True


def test_separate_fix_of_jev_menu_is_asked_locally_and_keeps_labels(monkeypatch):
    """JEV 로 판단한 메뉴의 고침은 로컬 해석기에 요청. JEV 재호출 없음, 판단에 쓰는 선택지 이름은 원문 유지."""
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    h, local, jev = make()
    p = run(h.classify(HEARD))
    assert h.last_route == ("jev", None) and not local.seen
    labels = [o.label for o in p.options]
    n = len(jev.states)
    local.fixes = [FIX]
    rec = run(h.fix_prompt(p.text, p.options, p.kind))
    assert rec["source"] == "model" and rec["applied"] == [("보이스티싱", "보이스피싱")]
    assert "보이스피싱 신고는 1번" in rec["shown"]
    assert len(local.fix_seen) == 1 and "보이스티싱" in local.fix_seen[0]
    assert len(jev.states) == n                       # 고침은 JEV 로 보내지 않음
    assert [o.label for o in p.options] == labels     # 판단 값은 그대로
    rec2 = run(h.fix_prompt(p.text, p.options, p.kind))
    assert rec2["source"] == "cache" and len(local.fix_seen) == 1  # 같은 메뉴는 로컬 저장값


def test_separate_fix_skips_text_sent_local_for_privacy(monkeypatch):
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    h, local, jev = make()
    text = "김철수 고객님, 보이스티싱 신고는 1번, 잔액조회는 3번을 눌러 주십시오."
    p = run(h.classify(text))
    assert h.last_route == ("local", "이름 호칭")
    local.fixes = [FIX]
    rec = run(h.fix_prompt(p.text, p.options, p.kind))
    assert rec["source"] == "private" and rec["applied"] == [] and rec["shown"] is None
    assert not local.fix_seen and not jev.states


@pytest.mark.parametrize("after_input", ["1", "0"])
def test_separate_fix_skips_after_input_request_until_next_call(monkeypatch, after_input):
    """민감정보 입력 요청 뒤 안내의 고침 제외 확인. JEV_PRIVATE_AFTER_INPUT 을 꺼서 판단이 JEV 로 가는 경우도 동일."""
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    monkeypatch.setenv("JEV_PRIVATE_AFTER_INPUT", after_input)
    h, local, jev = make()
    local.fixes = [FIX]
    assert run(h.classify("주민등록번호 13자리를 입력해 주세요.")).kind == "sensitive"
    p = run(h.classify(HEARD))
    assert h.last_route[0] == ("local" if after_input == "1" else "jev")
    rec = run(h.fix_prompt(p.text, p.options, p.kind))
    assert rec["source"] == "private" and not local.fix_seen
    run(h.summarize_goal("거래내역 조회"))  # 같은 통화의 목적 다시 말하기: 여전히 고침 제외
    p = run(h.classify(HEARD))
    assert run(h.fix_prompt(p.text, p.options, p.kind))["source"] == "private" and not local.fix_seen
    h.start_call()  # 새 통화
    run(h.summarize_goal("잔액 조회"))
    p = run(h.classify(HEARD))
    assert run(h.fix_prompt(p.text, p.options, p.kind))["applied"] == [("보이스티싱", "보이스피싱")]


def test_fix_prompt_is_off_unless_separate(monkeypatch):
    monkeypatch.delenv("LLM_FIX_TRANSCRIPT", raising=False)
    h, local, jev = make()
    local.fixes = [FIX]
    for text in (HEARD, "김철수 고객님, 보이스티싱 신고는 1번"):
        rec = run(h.fix_prompt(text, [], "menu"))
        assert rec["source"] == "off" and rec["applied"] == []
    assert not local.fix_seen


def test_fix_mode_1_leaves_jev_classified_prompt_unfixed(monkeypatch):
    """LLM_FIX_TRANSCRIPT=1 은 분류 호출에서 고침을 받는 방식이므로, JEV 로 판단한 안내에는 고침 없음 확인."""
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "1")
    h, local, jev = make()
    local.fixes = [FIX]
    p = run(h.classify(HEARD))
    assert h.last_route == ("jev", None) and p.fixes == [] and p.shown is None and not p.display_fixes
    assert run(h.fix_prompt(p.text, p.options, p.kind))["source"] == "off" and not local.seen

def test_session_shows_local_fix_for_jev_menu_after_key(monkeypatch):
    """Session 흐름: JEV 로 판단해 키를 누른 메뉴의 자막에 로컬 고침 표시. 경로의 선택지 이름(판단 값)은 원문 유지."""
    from session import Session
    from test_flows import ev, new_manager, wait_for

    order: list[str] = []
    real_press = Session._press

    async def press(self, key, *a, **kw):
        order.append("key")
        return await real_press(self, key, *a, **kw)

    monkeypatch.setattr(Session, "_press", press)  # Session 생성 전 교체 (첫 키 누락 방지)

    async def go():
        m = await new_manager()
        root = m.deps.backend.tree["nodes"]["root"]
        root["say"] = root["say"].replace("거래내역조회", "거대내역조회")  # 받아쓰기 오인식 예
        local, jev = FakeLocal(), FakeJev("3")
        local.fixes = [{"from": "거대내역조회", "to": "거래내역조회"}]
        h = HybridJevInterpreter(LocalLlmInterpreter(local), JevInterpreter(jev))
        h.fix_mode = "separate"
        real_fix = h.fix_prompt

        async def fix_prompt(text, options=None, kind="info"):
            order.append("fix")
            return await real_fix(text, options, kind)

        h.fix_prompt = fix_prompt
        m.deps.interp = m.deps.navigator.interp = h
        out = await m.create("15889999", "통장 잔액 알고 싶어")
        s = m.sessions[out["session_id"]]
        await wait_for(s, lambda s: ev(s, "path_updated"))
        await wait_for(s, lambda s: any("3번" in c["text"] for c in ev(s, "caption")))
        cap = next(c["text"] for c in ev(s, "caption") if "3번" in c["text"])
        assert "거래내역조회 4번" in cap and "거대내역조회" not in cap
        assert ev(s, "path_updated")[0]["selected"]["label"] == "잔액조회"
        assert s.layers[0].prompt.options[1].label == "거대내역조회"   # 판단에 쓴 이름은 원문
        assert order[:2] == ["key", "fix"]                              # 키 전송 뒤 고침 요청
        assert local.fix_seen and all("current_ars_transcript" in st for st in jev.states)
        if not s.ended:
            await s.action({"type": "cancel"})
        await wait_for(s, lambda s: ev(s, "ended"))
        assert s.hub.history[-1]["event"] == "ended"

    run(go())


def test_session_revise_goal_after_input_request_keeps_local_and_unfixed():
    """Session 흐름: 첫 번호 전 입력 요청 뒤 목적을 다시 말해도(revise_goal) '입력 요청 뒤' 상태 유지.
    그 뒤 안내는 로컬 판단, 고침 제외. 통화 시작 알림(start_call)은 Session 생성 때 한 번만 호출."""
    from test_flows import ev, new_manager, wait_for

    tree = {
        "start": "root", "timeout_s": 300,
        "nodes": {
            "root": {"type": "input", "say": "주민등록번호 13자리와 우물정자를 눌러 주십시오.", "terminator": "#", "next": "m1"},
            "m1": {"say": "잔액조회는 3번, 거래내역조회는 4번을 눌러 주십시오.", "keys": {"3": "m2", "4": "m2"}},
            "m2": {"say": "보이스티싱 신고는 1번, 계좌 잔액 안내는 2번을 눌러 주십시오.", "keys": {"1": "bye", "2": "bye"}},
            "bye": {"type": "end", "say": "감사합니다."},
        },
    }

    async def go():
        m = await new_manager()
        m.deps.backend.tree.clear()
        m.deps.backend.tree.update(tree)
        local, jev = FakeLocal(), FakeJev("3")
        local.fixes = [FIX]
        h = HybridJevInterpreter(LocalLlmInterpreter(local), JevInterpreter(jev))
        h.fix_mode = "separate"
        routes, starts = [], []
        real_classify, real_start = h.classify, h.start_call

        async def classify(text):
            p = await real_classify(text)
            routes.append((text[:6], h.last_route[0], p.kind))
            return p

        def start_call():
            starts.append(1)
            real_start()

        h.classify, h.start_call = classify, start_call
        m.deps.interp = m.deps.navigator.interp = h
        out = await m.create("15889999", "잔액 조회")
        s = m.sessions[out["session_id"]]
        assert len(starts) == 1
        req = await wait_for(s, lambda s: s.pending if s.pending and s.pending["kind"] == "secure_input" else None)
        assert (await s.action({"type": "revise_goal", "text": "통장 잔액 알고 싶어"}))["accepted"]
        assert len(starts) == 1 and h.input_heard           # 목적 다시 말하기는 새 통화 아님
        await s.action({"type": "secure_input", "request_id": req["request_id"], "digits": "9001011234567"})
        await wait_for(s, lambda s: any("신고" in c["text"] for c in ev(s, "caption")))
        cap = next(c["text"] for c in ev(s, "caption") if "신고" in c["text"])
        assert "보이스티싱" in cap and "보이스피싱" not in cap  # 고침 없음
        assert [r[1] for r in routes[1:]] and all(r[1] == "local" for r in routes[1:])  # 입력 요청 뒤는 모두 로컬
        assert not local.fix_seen
        assert not any("보이스티싱" in st.get("current_ars_transcript", "") for st in jev.states)
        if not s.ended:
            await s.action({"type": "cancel"})
        await wait_for(s, lambda s: ev(s, "ended"))

    run(go())


# 막다른 안내 (메뉴 키 뒤 업무 선택지 없이 이동 키만 남은 안내) --------------------------------------------
DEAD = ("증명서 발급 시 휴대전화 본인인증이 필요합니다. 홈페이지 또는 모바일 앱을 통해 발급받을 수 있습니다. "
        "전 단계로 돌아가시려면 우물정자를 눌러주세요.")


def test_answers_goal_is_always_local():
    """막다른 안내가 목적의 답인지는 그 안내를 JEV 로 분류한 경우에도 로컬에서 판정 (JEV 전송 없음)."""
    h, local, jev = make(kind="info")
    p = run(h.classify(DEAD))
    assert h.last_route == ("jev", None) and p.kind == "info" and p.back_key == "#"
    n = len(jev.states)
    sib = ["건강보험 자격 문의"]
    assert run(h.answers_goal("증명서 발급받고 싶어", DEAD, "증명서 발급", sib)) is True
    local.goal_done = {"best": "건강보험 자격 문의", "verdict": "other"}
    assert run(h.answers_goal("건강보험증 발급", DEAD, "증명서 발급", sib)) is False
    local.goal_done = {"best": "건강보험 자격 문의", "verdict": "answer"}   # 답이라면서 다른 선택지를 고른 경우 재질문(None)
    assert run(h.answers_goal("건강보험증 발급", DEAD, "증명서 발급", sib)) is None
    assert len(jev.states) == n and len(local.goal_done_seen) == 3
    assert "건강보험 자격 문의" in local.goal_done_seen[0]


def test_jev_result_on_dead_end_stays_info():
    """JEV 가 이동 키만 남은 안내를 '결과'로 보아도 일반 안내로 유지 (Session 의 막다른 안내 판단을 거치게 함).
    이동 키가 없는 안내의 '결과' 판정은 그대로 사용."""
    h, local, jev = make(kind="result")
    p = run(h.classify(DEAD))
    assert h.last_route == ("jev", None) and p.kind == "info" and p.back_key == "#" and not p.options
    p = run(h.classify("요청하신 증명서는 홈페이지에서 바로 확인하실 수 있습니다."))
    assert h.last_route == ("jev", None) and p.kind == "result"
    h, local, jev = make(kind="input_error")   # '결과' 가 아닌 판정은 이동 키가 있어도 그대로 사용
    assert run(h.classify(DEAD)).kind == "input_error"


def test_jev_rule_extract_uses_narrowed_input_rule():
    """JEV 쪽 규칙 추출에서 규칙 분류의 입력 요청 판별 우선 사용. 입력을 언급만 하는 안내는 입력 요청이 아니므로
    JEV 에 묻고 이동 키를 담음 확인 (입력 요청으로 잘못 보아 JEV 질의 없이 입력 화면을 띄우는 문제 방지)."""
    from jev_interpreter import extract_prompt
    assert extract_prompt("휴대폰 번호가 입력되지 않아 자세한 내용을 보내지 못했습니다. "
                          "고객님 다른 문의사항이 있으신가요?").kind == "info"
    assert extract_prompt("숫자 키패드로 안내받으실 휴대폰 번호를 눌러주세요.").kind == "sensitive"
    h, local, jev = make(kind="info")
    p = run(h.classify("휴대폰 번호가 입력되지 않아 안내를 마칩니다. 전 단계로 돌아가시려면 우물정자를 눌러 주세요."))
    assert h.last_route == ("jev", None) and p.kind == "info" and p.back_key == "#" and len(jev.states) == 1


def test_session_dead_end_judged_by_jev_goes_through_local_check():
    """Session 흐름: 메뉴 키 뒤 이동 키만 남은 안내를 JEV 가 '결과'로 보아도 막다른 안내 판단을 거침.
    판단은 로컬에서 처리, 다른 선택지 이름도 함께 참고. 목적의 답이면 결과로 처리."""
    from test_flows import ev, new_manager, wait_for

    tree = {
        "start": "root", "timeout_s": 300,
        "nodes": {
            "root": {"say": "국민건강보험공단입니다. 증명서 발급은 1번, 건강보험 자격 문의는 2번을 눌러 주십시오.",
                     "keys": {"1": "dead", "2": "bye"}},
            "dead": {"say": DEAD, "keys": {"#": "root"}},
            "bye": {"type": "end", "say": "감사합니다."},
        },
    }

    class Jev(FakeJev):
        async def decide(self, state, questions):
            self.kind = "result" if "본인인증" in state.get("current_ars_transcript", "") else "menu"
            return await super().decide(state, questions)

    async def go():
        m = await new_manager(dead_end_wait_s=0.05)
        m.deps.backend.tree.clear()
        m.deps.backend.tree.update(tree)
        local, jev = FakeLocal(), Jev("1")
        h = HybridJevInterpreter(LocalLlmInterpreter(local), JevInterpreter(jev))
        m.deps.interp = m.deps.navigator.interp = h
        out = await m.create("15889999", "증명서 발급받고 싶어")
        s = m.sessions[out["session_id"]]
        await wait_for(s, lambda s: s.status == "agent_done")
        assert s.summary["kind"] == "result"
        assert [e["selected"]["label"] for e in ev(s, "path_updated")] == ["증명서 발급"]
        assert len(local.goal_done_seen) == 1 and "건강보험 자격 문의" in local.goal_done_seen[0]
        assert any("본인인증" in st.get("current_ars_transcript", "") for st in jev.states)  # 분류는 JEV
        if not s.ended:
            await s.action({"type": "cancel"})
        await wait_for(s, lambda s: ev(s, "ended"))

    run(go())
