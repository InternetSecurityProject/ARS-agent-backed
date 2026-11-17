"""llm_mode(로컬 모델 판단, 규칙 없음) 시험. 실제 모델·실통화 없음 (가짜 클라이언트).

python -m pytest -q test_llm_mode.py
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from interpreter import AGENT_CATEGORY, Option, Prompt
from llm_mode import LlmInterpreter, LlmNavigator, parse_keys_said, to_prompt
from navigator import NavState
from test_flows import run

HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("LLM_COMBO", "COMBO_DEFAULT_INTERP", "RESULT_JUDGE"):
        monkeypatch.delenv(k, raising=False)


def test_llm_mode_has_no_regex_import():
    tree = ast.parse((HERE / "llm_mode.py").read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            assert all(a.name != "re" for a in n.names)
        if isinstance(n, ast.ImportFrom):
            assert n.module != "re"


def test_keys_must_be_said_in_the_text():
    text = "잔액 조회는 삼 번, 다시 들으시려면 별표를 눌러 주세요."
    assert parse_keys_said(["3|삼 번", "*|별표", "4|사 번", "x|별표", "5"], text) == {"3", "*"}
    assert parse_keys_said([{"key": "3", "said": "삼 번"}, {"key": "repeat_key", "said": "별표"}, {"key": "4", "said": "사 번"},
                            {"key": "*", "said": ""}, {"said": "별표"}], text) == {"3"}   # 칸 이름·빈 원문·원문에 없는 말은 버림


def test_to_prompt_keeps_only_grounded_keys_and_marks_agent():
    text = "잔액 조회는 삼 번, 거래내역은 사 번, 상담원 연결은 영 번, 다시 들으시려면 별표를 눌러 주세요."
    d = {"kind": "menu", "keys_said": ["3|삼 번", "4|사 번", "0|영 번", "*|별표"],
         "options": [{"key": "3", "label": "잔액 조회", "category": "잔액 조회"},
                     {"key": "4", "label": "거래내역", "category": "거래내역 조회"},
                     {"key": "7", "label": "지어낸 메뉴", "category": "x"},
                     {"key": "*", "label": "다시 듣기", "category": "다시 듣기"}],
         "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": "*", "agent_key": "0",
         "input_request": False, "input_label": None, "terminator": None, "critical": False,
         "scores": [{"key": "3", "confidence": 1.4}, {"key": "4", "confidence": 0.2}], "wants_agent": False,
         "same_kind": ["4", "9"], "agent_or_self": False}
    p, hint, scores = to_prompt(d, text)
    assert p.kind == "menu" and [o.key for o in p.options] == ["3", "4", "0"] and p.repeat_key == "*"
    assert p.options[2].category == AGENT_CATEGORY and p.agent_key == "0"
    assert "지어낸 메뉴" in hint["unnumbered"]                         # 원문에 없는 번호는 선택지가 아님
    assert scores == {"3": 1.0, "4": 0.2, "0": 0.0} and hint["top"] == "3" and hint["same_kind"] == ["4"]


def test_to_prompt_input_request_and_empty_menu():
    p, _, _ = to_prompt({"kind": "info", "keys_said": ["#|샵"], "options": [], "input_request": True, "input_label": "주민등록번호",
                         "terminator": "#"}, "주민등록번호 앞 여섯 자리와 샵을 눌러 주세요.")
    assert p.kind == "sensitive" and p.label == "주민등록번호" and p.terminator == "#"
    p, _, _ = to_prompt({"kind": "menu", "keys_said": [], "options": [{"key": "1", "label": "a", "category": "a"}]}, "안내")
    assert p.kind == "info" and not p.options                            # 근거 있는 선택지가 없으면 메뉴가 아님
    p, hint, _ = to_prompt({"kind": "menu", "keys_said": ["1|1번", "2|2번"],
                            "options": [{"key": "1", "label": "가", "category": "가"}, {"key": "1", "label": "나", "category": "나"},
                                        {"key": "2", "label": "다", "category": "다"}]}, "가는 1번 나는 1번 다는 2번")
    assert [o.key for o in p.options] == ["2"] and "가" in hint["unnumbered"]   # 같은 번호에 다른 이름 둘: 둘 다 제외


def test_out_of_hours_with_two_tasks_is_menu_and_repeat_key_is_not_back():
    text = "지금은 상담사 연결이 불가합니다. 자동차 검사는 1번, 위치 안내는 2번, 다시 듣기는 9번을 눌러주십시오."
    d = {"kind": "out_of_hours", "keys_said": ["1|1번", "2|2번", "9|9번"],
         "options": [{"key": "1", "label": "자동차 검사", "category": "검사"}, {"key": "2", "label": "위치 안내", "category": "위치"}],
         "back_key": "9", "home_key": None, "repeat_key": "9", "scores": []}
    p, hint, scores = to_prompt(d, text)
    assert p.kind == "menu" and [o.key for o in p.options] == ["1", "2"]
    assert p.back_key is None and p.repeat_key == "9"
    assert scores is None and hint["top"] is None                         # 점수가 없으면 따로 물음
    d.update(options=[{"key": "1", "label": "자동차 검사", "category": "검사"}])
    assert to_prompt(d, text)[0].kind == "out_of_hours"                   # 업무 하나뿐이면 모델 판단 그대로


class FakeClient:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    async def generate(self, system, user, schema, timeout_s=None, retries=None, max_tokens=None):
        self.calls.append((system, user, schema))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply(system, user, schema) if callable(self.reply) else dict(self.reply)


class FakeBase:
    goal = None
    decide_timeout_s = 4.0

    async def summarize_goal(self, goal):
        self.goal = goal
        return "목적 요약"


MENU2 = "카드 발급 문의 서비스입니다. 모바일 홈페이지 문자 수신을 원하시면 1번, 상담원 연결을 원하시면 0번을 눌러주십시오."


def _reply(agent_or_self, wants_agent=False, top="1"):
    return {"kind": "menu", "keys_said": ["1|1번", "0|0번"],
            "options": [{"key": "1", "label": "모바일 홈페이지 문자 수신", "category": "문자 수신"},
                        {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}],
            "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": "0", "input_request": False,
            "input_label": None, "terminator": None, "critical": False,
            "scores": [{"key": "1", "confidence": 0.8 if top == "1" else 0.2}, {"key": "0", "confidence": 0.2 if top == "1" else 0.8}],
            "wants_agent": wants_agent, "same_kind": [], "agent_or_self": agent_or_self}


def _decide(reply, goal="카드 발급 받고 싶어"):
    it = LlmInterpreter(FakeClient(reply), FakeBase())
    run(it.summarize_goal(goal))
    p = run(it.classify(MENU2))
    nav = LlmNavigator(it)
    return p, run(nav.decide(NavState(goal), p, []))


def test_navigator_asks_when_model_says_user_should_choose_between_task_and_agent():
    p, dec = _decide(_reply(agent_or_self=True))
    assert dec.action == "ask" and [o.key for o in dec.options] == ["1", "0"]
    p, dec = _decide(_reply(agent_or_self=False))
    assert dec.action == "press" and dec.key == "1"                     # 낱말 비교 없이 모델 판단을 따름


def test_navigator_agent_top_without_agent_request_asks_and_with_request_presses():
    _, dec = _decide(_reply(False, wants_agent=False, top="0"))
    assert dec.action == "ask"
    _, dec = _decide(_reply(False, wants_agent=True, top="0"))
    assert dec.action == "press" and dec.key == "0"


def test_classify_failure_is_info_and_cache_is_used():
    it = LlmInterpreter(FakeClient(TimeoutError()), FakeBase())
    assert run(it.classify(MENU2)).kind == "info"
    c = FakeClient(_reply(False))
    it = LlmInterpreter(c, FakeBase())
    run(it.summarize_goal("카드 발급"))
    run(it.classify(MENU2))
    run(it.classify(MENU2))
    assert len(c.calls) == 1
    assert run(it.score("카드 발급", [Option("1", "모바일 홈페이지 문자 수신"), Option("0", "상담원 연결")])) == {"1": 0.8, "0": 0.2}
    assert len(c.calls) == 1                                             # 분류 때 받은 점수 사용


def test_scores_of_another_menu_with_overlapping_keys_are_not_reused():
    """1~8번 메뉴 점수를 받은 뒤 1~6번의 다른 메뉴가 오면, 키가 겹쳐도 앞 메뉴 점수를 쓰지 않음."""
    def reply(system, user, schema):
        if "keys_said" not in schema["properties"]:
            return {"scores": [{"key": k, "confidence": 0.9 if k == "1" else 0.1} for k in "123456"]}
        n = 8 if "가1" in user else 6
        pre = "가" if n == 8 else "나"
        return {"kind": "menu", "keys_said": [f"{i}|{pre}{i}" for i in range(1, n + 1)],
                "options": [{"key": str(i), "label": f"{pre}{i}", "category": f"{pre}{i}"} for i in range(1, n + 1)],
                "scores": [{"key": str(i), "confidence": 0.9 if i == 3 else 0.1} for i in range(1, n + 1)] if n == 8 else [],
                "wants_agent": False, "same_kind": [], "agent_or_self": False}
    c = FakeClient(reply)
    it = LlmInterpreter(c, FakeBase())
    run(it.summarize_goal("목적"))
    first = run(it.classify(" ".join(f"가{i}" for i in range(1, 9))))
    assert run(it.score("목적", first.options)) == {str(i): (0.9 if i == 3 else 0.1) for i in range(1, 9)}
    second = run(it.classify(" ".join(f"나{i}" for i in range(1, 7))))
    got = run(it.score("목적", second.options))
    assert got["1"] == 0.9 and got["3"] == 0.1 and len(c.calls) == 3      # 다른 메뉴는 점수를 새로 물음


def test_confirm_never_auto_presses_critical(monkeypatch):
    it = LlmInterpreter(FakeClient({"key": "1"}), FakeBase())
    p = Prompt(kind="confirm", text="해지하시려면 1번", options=[Option("1", "해지"), Option("2", "취소")], critical=True)
    assert run(it.answer_confirm("해지", p)) is None
    p.critical = False
    assert run(it.answer_confirm("해지", p)) is None                    # 기본: 확인 질문은 사용자에게 물음
    monkeypatch.setenv("LLM_AUTO_CONFIRM", "1")
    assert run(it.answer_confirm("해지", p)) == "1"


class MergeBase(FakeBase):
    last_answer = None
    last_menu = None
    last_path = []

    async def merge_answer(self, goal, question, answer):
        self.last_answer, self.seen_menu = answer, self.last_menu
        return goal


def test_text_answer_is_used_only_for_the_question_it_answered():
    readback = "입력하신 번호는 ****입니다. 맞으면 1번, 틀리면 2번을 눌러주십시오."
    menu_text = "잔액 조회는 1번, 카드 분실 신고는 2번을 눌러주십시오."

    def reply(system, user, schema):
        if "<answer>" in user:
            return {"key": "1"}
        if "맞으면" in user:
            return {"kind": "confirm", "keys_said": ["1|1번", "2|2번"], "options": [
                {"key": "1", "label": "맞음", "category": "맞음"}, {"key": "2", "label": "틀림", "category": "틀림"}]}
        return {"kind": "menu", "keys_said": ["1|1번", "2|2번"], "options": [
            {"key": "1", "label": "잔액 조회", "category": "잔액 조회"}, {"key": "2", "label": "카드 분실 신고", "category": "분실"}]}
    c, base = FakeClient(reply), MergeBase()
    it = LlmInterpreter(c, base)
    menu = run(it.classify(menu_text))
    run(it.merge_answer("목적", "어느 쪽인가요?", "1번"))                    # 메뉴 질문에 쓴 답
    assert base.seen_menu is menu                                         # 로컬 해석기가 '1번' 을 풀 메뉴를 받음
    rb = run(it.classify(readback))
    assert run(it.answer_confirm("목적", rb)) is None                     # 앞 메뉴의 '1번' 은 되읽기 확인에 쓰지 않음
    assert not any("<answer>" in u for _, u, _ in c.calls)
    run(it.merge_answer("목적", "번호가 맞나요?", "네 맞아요"))               # 바로 이 확인 질문에 쓴 답
    assert run(it.answer_confirm("목적", rb)) == "1" and base.last_answer is None
    assert run(it.answer_confirm("목적", rb)) is None                     # 한 번만 사용


def test_dropped_top_option_makes_navigator_ask_instead_of_pressing_the_runner_up():
    text = "잔액 조회는 삼번, 거래내역 조회는 사 번, 이체는 오 번을 눌러주십시오."
    d = {"kind": "menu", "keys_said": ["3|삼 번", "4|사 번", "5|오 번"],          # '삼 번' 은 원문('삼번')에 그대로 없음
         "options": [{"key": "3", "label": "잔액 조회", "category": "잔액"}, {"key": "4", "label": "거래내역 조회", "category": "거래"},
                     {"key": "5", "label": "이체", "category": "이체"}],
         "scores": [{"key": "3", "confidence": 0.9}, {"key": "4", "confidence": 0.5}, {"key": "5", "confidence": 0.1}],
         "wants_agent": False, "same_kind": [], "agent_or_self": False}
    p, hint, scores = to_prompt(d, text)
    assert [o.key for o in p.options] == ["4", "5"] and scores == {"4": 0.4, "5": 0.4}   # 모두 0.4: '확신 없음' 재질문
    it = LlmInterpreter(FakeClient(d), FakeBase())
    run(it.summarize_goal("통장 잔액 조회"))
    p = run(it.classify(text))
    dec = run(LlmNavigator(it).decide(NavState("통장 잔액 조회"), p, []))
    assert dec.action == "ask"
    d2 = {**d, "scores": [{"key": "3", "confidence": 0.1}, {"key": "4", "confidence": 0.9}, {"key": "5", "confidence": 0.1}]}
    assert to_prompt(d2, text)[2] == {"4": 0.9, "5": 0.1}                # 1위가 아닌 선택지가 빠졌으면 그대로


def test_answer_to_a_session_built_confirm_is_used_and_keeps_the_goal():
    """세션이 되읽기 규칙으로 새로 만든 확인 질문(분류를 거치지 않음)에 글로 답해도 그 답으로 키를 고르고, 목적은 그대로 둠."""
    calls = []

    def reply(system, user, schema):
        calls.append(user)
        return {"key": "1"} if "<answer>" in user else {"kind": "info", "keys_said": [], "options": []}
    base = MergeBase()
    it = LlmInterpreter(FakeClient(reply), base)
    run(it.classify("입력하신 번호는 ****입니다. 맞으면 1번, 틀리면 2번을 눌러주십시오."))   # 모델은 info 로 분류
    built = Prompt(kind="confirm", text="입력하신 번호는 ****입니다. 맞으면 1번, 틀리면 2번을 눌러주십시오.",
                   options=[Option("1", "맞음"), Option("2", "틀림")])
    it.note_asked(built, built.text)
    assert run(it.merge_answer("잔액 조회", built.text, "1번")) == "잔액 조회"   # 확인 질문의 답은 목적을 바꾸지 않음
    assert not hasattr(base, "seen_menu")                                  # 로컬 해석기의 번호 풀이에 넘기지 않음
    assert run(it.answer_confirm("잔액 조회", built)) == "1"


def test_stale_text_answer_is_dropped_when_a_new_paragraph_arrives():
    """앞 확인 질문에 한 글 답이 남아 있어도, 새 안내를 들은 뒤 이름이 같은 다른 확인 질문에는 쓰지 않음."""
    calls = []

    def reply(system, user, schema):
        calls.append(user)
        if "<answer>" in user:
            return {"key": "1"}
        return {"kind": "confirm", "keys_said": ["1|1번", "2|2번"], "options": [
            {"key": "1", "label": "예", "category": "예"}, {"key": "2", "label": "아니요", "category": "아니요"}]}
    it = LlmInterpreter(FakeClient(reply), MergeBase())
    q1 = run(it.classify("안내를 문자로 받으시겠습니까? 예는 1번, 아니요는 2번을 눌러주십시오."))
    run(it.merge_answer("목적", q1.text, "네"))                             # 답을 받았지만 다시 판단하지 않고 끝난 경우
    q2 = run(it.classify("자동이체 계좌도 함께 변경하시겠습니까? 예는 1번, 아니요는 2번을 눌러주십시오."))
    assert q1.signature() == q2.signature()
    assert run(it.answer_confirm("목적", q2)) is None and not any("<answer>" in u for u in calls)


def test_navigator_asks_the_model_whether_a_changed_goal_wants_an_agent():
    """목적이 바뀐 뒤 같은 메뉴를 다시 판단하면(분류 때 힌트는 옛 목적의 것) 상담원 요청 여부를 지금 목적으로 모델에 물음."""
    def reply(system, user, schema):
        if "wants_agent" in schema["properties"] and len(schema["properties"]) == 1:
            return {"wants_agent": "상담원" in user}
        if "keys_said" not in schema["properties"]:
            return {"scores": [{"key": "1", "confidence": 0.1}, {"key": "2", "confidence": 0.1}, {"key": "0", "confidence": 0.9}]}
        return {"kind": "menu", "keys_said": ["1|1번", "2|2번", "0|0번"], "agent_key": "0", "options": [
            {"key": "1", "label": "잔액 조회", "category": "잔액"}, {"key": "2", "label": "카드 분실 신고", "category": "분실"},
            {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}],
                "scores": [{"key": "1", "confidence": 0.45}, {"key": "2", "confidence": 0.45}, {"key": "0", "confidence": 0.1}],
                "wants_agent": False, "same_kind": [], "agent_or_self": False}
    it = LlmInterpreter(FakeClient(reply), FakeBase())
    run(it.summarize_goal("카드 문의"))
    p = run(it.classify("잔액 조회는 1번, 카드 분실 신고는 2번, 상담원 연결은 0번을 눌러주십시오."))
    it.set_goal("상담원 연결")
    dec = run(LlmNavigator(it).decide(NavState("상담원 연결"), p, [], arrived=False))
    assert dec.action == "press" and dec.key == "0"                      # 다시 묻지 않고 상담원 연결을 누름


def test_input_request_on_a_menu_with_two_tasks_stays_a_menu():
    p, _, _ = to_prompt({"kind": "menu", "keys_said": ["1|1번", "2|2번"], "input_request": True, "input_label": "번호",
                         "options": [{"key": "1", "label": "잔액 조회", "category": "잔액"},
                                     {"key": "2", "label": "거래내역 조회", "category": "거래"}]},
                        "잔액 조회는 1번, 거래내역 조회는 2번을 눌러주십시오.")
    assert p.kind == "menu" and [o.key for o in p.options] == ["1", "2"]


def test_input_request_with_agent_option_is_sensitive():
    p, _, _ = to_prompt({"kind": "menu", "keys_said": ["0|0번"], "options": [{"key": "0", "label": "상담원 연결", "category": "상담원 연결"}],
                         "agent_key": "0", "input_request": True, "input_label": "카드 비밀번호", "terminator": None},
                        "카드 비밀번호 네 자리를 입력해 주십시오. 상담원 연결은 0번입니다.")
    assert p.kind == "sensitive" and p.label == "카드 비밀번호" and not p.options


def test_hints_of_the_old_goal_are_not_used_after_the_goal_changes():
    it = LlmInterpreter(FakeClient(_reply(agent_or_self=True)), FakeBase())
    run(it.summarize_goal("카드 발급"))
    p = run(it.classify(MENU2))
    assert it.hint_for(p)["agent_or_self"] is True
    it.set_goal("상담원 연결")
    assert not (it.hint_for(p) or {}).get("agent_or_self")


def test_agent_wait_with_agent_option_asks_the_local_agent_check():
    class AgentBase(FakeBase):
        async def agent_wait_menu(self, text, prompt):
            return Prompt(kind="menu", text=text, options=[Option("1", "콜백 예약"), Option("0", "상담원 연결", AGENT_CATEGORY)])
    d = {"kind": "agent_wait", "keys_said": ["1|1번", "0|0번"], "options": [
        {"key": "1", "label": "콜백 예약", "category": "콜백"}, {"key": "0", "label": "상담원 연결", "category": "상담원 연결"}],
         "agent_key": "0"}
    it = LlmInterpreter(FakeClient(d), AgentBase())
    p = run(it.classify("상담원 연결을 원하시면 0번, 콜백 예약은 1번을 눌러주십시오."))
    assert p.kind == "menu" and [o.key for o in p.options] == ["1", "0"]
    it = LlmInterpreter(FakeClient(d), FakeBase())                        # 확인 기능이 없거나 실패하면 상담원 대기 그대로
    assert run(it.classify("상담원 연결을 원하시면 0번, 콜백 예약은 1번을 눌러주십시오.")).kind == "agent_wait"


def test_rule_free_call_records_user_keys_as_digit_counts_only():
    """규칙 안전망이 없는 해석기(llm)는 입력 요청을 놓쳤을 때 키패드 번호가 기록에 남지 않게 사용자 키를 자릿수로만 남김."""
    from test_flows import manual

    async def go(rule_free):
        s, call = await manual("통장 잔액 알고 싶어", record=True)
        if rule_free:
            s.d.interp.rule_free = True
        await s.action({"type": "key", "key": "7413"})                    # 키패드 직접 입력
        await s._press("9", "user", None)                                  # 버튼으로 고른 키(키패드 아님)는 값을 남김
        acts = [e for e in s.recorder.events if e.get("action") == "key"]
        sent = [e for e in s.recorder.events if e.get("type") == "keys_sent" and e.get("actor") == "user"]
        path = [e["data"] for e in s.recorder.events if e.get("type") == "event:path_updated"]
        text_log = s.recorder.text_log() if hasattr(s.recorder, "text_log") else ""
        import json as _j
        blob = _j.dumps(s.recorder.events, ensure_ascii=False)
        await s.finish("user_canceled")
        return acts, sent, path, text_log, blob, call.keys

    acts, sent, path, text_log, blob, keys = run(go(True))
    assert keys[-2:] == ["7413", "9"]                                     # 키는 그대로 ARS 로 전송
    assert "key" not in acts[0] and acts[0]["digit_count"] == 4
    assert sent[0].get("keys") is None and sent[0]["digit_count"] == 4 and sent[1]["keys"] == "9"
    assert path[0]["selected"]["key"] is None and path[0]["selected"]["digit_count"] == 4 and "7413" not in path[0]["text"]
    assert path[1]["selected"]["key"] == "9"
    assert "7413" not in blob and "7413" not in text_log
    acts, sent, path, _, _, _ = run(go(False))
    assert acts[0]["key"] == "7413" and sent[0]["keys"] == "7413" and path[0]["selected"]["key"] == "7413"   # 다른 조합은 키 값을 그대로 기록


def test_answer_to_a_dead_end_question_is_merged_even_after_a_confirm():
    """세션이 '지금 안내가 아닌 질문'(막다른 안내·음성 응답 요구)이라고 알리면(note_asked(None)) 앞서 분류한 확인 질문을 쓰지 않고
    답을 목적에 합침. 로컬 해석기에는 확인 질문을 넘기지 않음."""
    class Base(MergeBase):
        async def merge_answer(self, goal, question, answer):
            self.seen_menu = self.last_menu
            return answer
    def reply(system, user, schema):
        return {"kind": "confirm", "keys_said": ["1|1번", "2|2번"], "options": [
            {"key": "1", "label": "예", "category": "예"}, {"key": "2", "label": "아니요", "category": "아니요"}]}
    base = Base()
    it = LlmInterpreter(FakeClient(reply), base)
    run(it.classify("문자로 받으시겠습니까? 예는 1번, 아니요는 2번을 눌러주십시오."))
    it.note_asked(None, "찾는 업무가 아닌 것 같습니다. 어떤 업무를 원하세요?")
    assert run(it.merge_answer("잔액 조회", "질문", "카드 분실 신고하고 싶어요")) == "카드 분실 신고하고 싶어요"
    assert base.seen_menu is None


def test_dropped_option_tying_low_scores_does_not_raise_them():
    text = "가는 일번, 나는 이 번, 다는 삼 번을 눌러주십시오."
    d = {"kind": "menu", "keys_said": ["1|1번", "2|이 번", "3|삼 번"],                # '1번' 은 원문('일번')에 없음
         "options": [{"key": "1", "label": "가", "category": "가"}, {"key": "2", "label": "나", "category": "나"},
                     {"key": "3", "label": "다", "category": "다"}],
         "scores": [{"key": "1", "confidence": 0.1}, {"key": "2", "confidence": 0.1}, {"key": "3", "confidence": 0.1}]}
    assert to_prompt(d, text)[2] == {"2": 0.1, "3": 0.1}                  # 맞는 갈래가 없는 메뉴: 그대로 (되돌아가기)


def test_keypad_number_is_scrubbed_from_every_later_record():
    """llm 통화에서 키패드로 누른 번호(3자리 이상, 한 자씩 눌러도 이어 붙임)는 뒤의 기록(자막·받아쓰기·안내 문구)에서도 지움."""
    from test_flows import manual

    async def go(rule_free):
        s, _ = await manual("통장 잔액 알고 싶어", record=True)
        if rule_free:
            s.d.interp.rule_free = True
        for k in "7413":
            await s.action({"type": "key", "key": k})
        s.recorder.event("caption", {"text": "입력하신 번호는 7413입니다"})
        s.recorder.stt("입력하신 번호는 7413입니다", 8000, 8000, 0.1, "whisper")
        s.publish("status_changed", {"status": "selecting", "message": "'7413' 은 찾는 업무와 다른 것 같습니다."})
        import json as _j
        blob = _j.dumps(s.recorder.events, ensure_ascii=False) + _j.dumps(s.recorder.captions, ensure_ascii=False) + s.recorder.text_log()
        await s.finish("user_canceled")
        return blob
    assert "7413" not in run(go(True)) and "(입력값)" in run(go(True))
    assert "7413" in run(go(False))                                       # 다른 조합은 번호를 지우지 않음


def test_rejected_keypad_key_is_not_kept_in_the_record():
    from test_flows import manual

    async def go():
        s, _ = await manual("통장 잔액 알고 싶어", record=True)
        s.d.interp.rule_free = True
        await s.action({"type": "key", "key": "5521"})
        out = s._recorded("input_rejected", {"keys": "5521", "actor": "user", "message": "번호가 인식되지 않았습니다."})
        await s.finish("user_canceled")
        return out
    out = run(go())
    assert out["keys"] is None and out["digit_count"] == 4


def test_combo_llm_is_hidden_unless_enabled_and_builds_llm_parts(monkeypatch):
    from combos import ComboFactory
    from session import Deps, SessionManager
    from app import build_deps
    from test_combo import end_all
    from test_flows import make_settings
    from test_hybrid_interpreter import FakeJev, FakeLocal
    from hybrid_interpreter import HybridJevInterpreter
    from jev_interpreter import JevInterpreter
    from local_llm import LocalLlmInterpreter

    base = HybridJevInterpreter(LocalLlmInterpreter(FakeLocal()), JevInterpreter(FakeJev("3")))
    f = ComboFactory(base.local.client, base.jev.client, "hybrid", "0")
    assert "llm" not in [x["id"] for x in f.options()["interp"]]
    monkeypatch.setenv("LLM_COMBO", "1")
    assert "llm" in [x["id"] for x in f.options()["interp"]]
    assert f.normalize("llm", "0") == ("llm", "0") and f.normalize_result("llm", "jev") == "local"
    assert f.normalize("llm", "separate") == ("llm", "separate") and f.normalize("llm", "1") == ("llm", "separate")
    it = f.build("llm", "0")
    assert isinstance(it, LlmInterpreter) and it.base.client is f.local_client and it.fix_mode == "0"
    fx = f.build("llm", "separate")
    assert fx.fix_mode == "separate" and fx.fix_prompt == fx.base.fix_prompt and fx.base.fix_mode == "separate"
    assert f.normalize("llm", "judge") == ("llm", "judge") and f.build("llm", "judge").fix_mode == "judge"

    async def go():
        from navigator import Navigator
        d = build_deps(make_settings(sim_lines=1))
        await d.backend.start()
        m = SessionManager(Deps(d.settings, d.backend, d.recognizer, base, Navigator(base), f))
        s = m.sessions[(await m.create("15889999", "잔액 조회", interp="llm"))["session_id"]]
        try:
            return s
        finally:
            await end_all(m)

    s = run(go())
    assert isinstance(s.d.navigator, LlmNavigator) and isinstance(s.d.interp, LlmInterpreter)
    assert s.combo == {"interp": "llm", "fix": "0", "result": "local"}


def test_llm_call_runs_on_sim_backend(monkeypatch):
    """가상 백엔드 통화 하나: 로컬 모델(가짜)이 원문에서 키를 뽑아 메뉴를 판단하고 키를 누름."""
    from combos import ComboFactory
    from session import Deps, SessionManager
    from app import build_deps
    from test_combo import end_all
    from test_flows import ev, make_settings, wait_for
    from test_hybrid_interpreter import FakeJev, FakeLocal
    from hybrid_interpreter import HybridJevInterpreter
    from jev_interpreter import JevInterpreter
    from local_llm import LocalLlmInterpreter

    def answer(system, user, schema):
        props = schema.get("properties", {})
        if "keys_said" not in props:
            return FakeLocal().reply if "kind" in props else {"text": "목적 요약", "scores": [], "key": None, "fixes": []}
        text = user.split("<ars>\n", 1)[1].split("\n</ars>", 1)[0]
        found = re.findall(r"([가-힣A-Za-z ]{2,20}?)\s*(?:은|는)\s*([0-9])\s*번", text)   # 시험 도구의 가짜 모델만 정규식 사용
        opts, said = [], []
        for label, key in found:
            if key not in [o["key"] for o in opts]:
                opts.append({"key": key, "label": label.strip(), "category": label.strip()})
                said.append(f"{key}|{key}번")
        out = {"kind": "menu" if opts else "info", "keys_said": said, "options": opts, "unnumbered": [], "back_key": None,
               "home_key": None, "repeat_key": None, "agent_key": None, "input_request": False, "input_label": None,
               "terminator": None, "critical": False}
        if "scores" in props:
            out.update({"scores": [{"key": o["key"], "confidence": 0.9 if "잔액" in o["label"] else 0.1} for o in opts],
                        "wants_agent": False, "same_kind": [], "agent_or_self": False})
        return out

    class Local(FakeLocal):
        async def generate(self, system, user, schema, timeout_s=None, retries=None, max_tokens=None):
            return answer(system, user, schema)

    monkeypatch.setenv("LLM_COMBO", "1")

    async def go():
        from navigator import Navigator
        base = HybridJevInterpreter(LocalLlmInterpreter(Local()), JevInterpreter(FakeJev("3")))
        f = ComboFactory(base.local.client, base.jev.client, "hybrid", "0")
        d = build_deps(make_settings(sim_lines=1))
        d.backend.tree["timeout_s"] = 300
        await d.backend.start()
        m = SessionManager(Deps(d.settings, d.backend, d.recognizer, base, Navigator(base), f))
        s = m.sessions[(await m.create("15889999", "통장 잔액 알고 싶어", interp="llm"))["session_id"]]
        try:
            await wait_for(s, lambda s: ev(s, "path_updated"), timeout=10)
            return s
        finally:
            await end_all(m)

    s = run(go())
    first = __import__("test_flows").ev(s, "path_updated")[0]
    assert first["selected"]["key"] == "3"
