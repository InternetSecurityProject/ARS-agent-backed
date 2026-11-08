"""
local_llm 시험 (실제 LLM 서버 호출 없음, httpx 가짜 전송으로 응답 모사)

실제 모델의 정확도·속도는 bench/menu_eval.py 로 측정. 이 시험은 요청 형식과 실패 시 대체 경로만 확인.
python -m pytest -q test_local_llm.py
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from gemini_interpreter import SCHEMA_CLASSIFY
from local_llm import LocalLlmInterpreter, OpenAICompatClient, parse_json

ROOT = "KB국민은행입니다. 잔액조회는 3번, 거래내역조회는 4번, 상담직원 연결은 0번을 눌러 주십시오."


def run(coro):
    return asyncio.run(coro)


def client_with(handler, api: str = "ollama", **kw) -> tuple[OpenAICompatClient, list]:
    """handler(요청 본문 dict) → httpx.Response. 보낸 본문은 seen 에 수집."""
    seen: list = []

    def h(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        seen.append((str(req.url), body))
        return handler(body)

    c = OpenAICompatClient("http://llm.test" + ("/v1" if api == "openai" else ""), "m", api=api, **kw)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(h))
    return c, seen


def ollama_reply(obj, **extra) -> httpx.Response:
    return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
                                     "prompt_eval_count": 1500, "eval_count": 100, **extra})


MENU = {"kind": "menu", "options": [{"key": "3", "label": "잔액조회", "category": "잔액 조회"},
                                    {"key": "0", "label": "상담직원 연결", "category": "상담원 연결"}],
        "unnumbered": [], "back_key": None, "home_key": None, "repeat_key": None, "agent_key": "0",
        "label": None, "terminator": None, "critical": False}


# 응답 읽기 ---------------------------------------------------------------
def test_parse_json_strips_thinking_and_fences():
    assert parse_json('<think>음...</think>\n{"a": 1}') == {"a": 1}
    assert parse_json('```json\n{"a": 2}\n```') == {"a": 2}
    with pytest.raises(ValueError):
        parse_json("[1, 2]")


# 요청 형식 ---------------------------------------------------------------
def test_ollama_request_has_schema_ctx_think_keepalive():
    c, seen = client_with(lambda b: ollama_reply({"text": "ok"}), num_ctx=8192, keep_alive="24h")
    assert run(c.generate("sys", "user", {"type": "object"})) == {"text": "ok"}
    url, body = seen[0]
    assert url.endswith("/api/chat")
    assert body["format"] == {"type": "object"} and body["think"] is False and body["stream"] is False
    assert body["options"]["num_ctx"] == 8192 and body["options"]["temperature"] == 0 and body["keep_alive"] == "24h"
    assert body["options"]["presence_penalty"] == 0 and body["options"]["repeat_penalty"] == 1.0  # 모델 기본 반복 벌점 끔
    assert body["shift"] is False  # 문맥 초과 시 조용히 자르지 않고 오류 (→ 대체 경로)
    assert c.last["prompt_tokens"] == 1500


def test_openai_request_uses_json_schema_and_disables_thinking():
    reply = {"choices": [{"message": {"content": '{"text": "ok"}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    c, seen = client_with(lambda b: httpx.Response(200, json=reply), api="openai")
    assert run(c.generate("sys", "user", {"type": "object"})) == {"text": "ok"}
    url, body = seen[0]
    assert url.endswith("/v1/chat/completions")
    assert body["response_format"]["type"] == "json_schema" and body["response_format"]["json_schema"]["strict"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_model_without_think_option_is_retried_without_it():
    def h(b):
        if "think" in b:
            return httpx.Response(400, json={"error": '"m" does not support thinking'})
        return ollama_reply({"text": "ok"})

    c, seen = client_with(h)
    assert run(c.generate("sys", "user", {"type": "object"}, retries=0)) == {"text": "ok"}
    assert "think" in seen[0][1] and "think" not in seen[1][1] and c.think is None


# 실패 경로 ---------------------------------------------------------------
@pytest.mark.parametrize("resp", [
    httpx.Response(500, text="model crashed"),
    httpx.Response(200, json={"message": {"content": "죄송하지만 JSON 이 아닙니다"}}),
    httpx.Response(200, json={"message": {"content": "[]"}}),
])
def test_bad_reply_raises_after_retries(resp):
    c, seen = client_with(lambda b: resp)
    with pytest.raises(RuntimeError):
        run(c.generate("sys", "user", {"type": "object"}, retries=1))
    asked = [b for _, b in seen if b["messages"][1]["content"] == "user"]  # 실패 뒤 백그라운드 다시 올리기 요청 제외
    assert len(asked) == 2  # 한 번 재시도


def test_connection_refused_raises():
    def h(b):
        raise httpx.ConnectError("refused")

    c, _ = client_with(h)
    with pytest.raises(RuntimeError):
        run(c.generate("sys", "user", {"type": "object"}, retries=0))


def test_server_down_falls_back_to_rules_and_asks():
    """LLM 서버가 꺼져 있을 때 규칙 분류로 메뉴를 읽고 점수는 재질문 값(0.40)으로 두는지 확인 (번호 선택 없음)."""
    def h(b):
        raise httpx.ConnectError("refused")

    c, _ = client_with(h)
    ai = LocalLlmInterpreter(c)
    ai.goal = "카드 결제일 바꾸고 싶어"
    p = run(ai.classify(ROOT))
    assert p.kind == "menu" and {o.key for o in p.options} >= {"3", "4", "0"}
    s = run(ai.score(ai.goal, p.options, []))
    assert set(s.values()) == {0.40}


def test_interpreter_uses_model_reply_and_scores():
    reply = {**MENU, "scores": [{"key": "3", "confidence": 0.95}, {"key": "0", "confidence": 0.2}]}
    c, seen = client_with(lambda b: ollama_reply(reply))
    ai = LocalLlmInterpreter(c)
    ai.goal = "잔액 조회"
    p = run(ai.classify(ROOT))
    assert [o.key for o in p.options] == ["3", "0"]
    assert run(ai.score("잔액 조회", p.options, [])) == {"3": 0.95, "0": 0.2}
    assert len(seen) == 1  # 점수는 분류 때 함께 받았으므로 재호출 없음
    assert "scores" in seen[0][1]["format"]["properties"]  # 분류+점수 스키마로 보냄


def test_decide_timeout_env(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_DECIDE_TIMEOUT_S", "2.5")
    ai = LocalLlmInterpreter(OpenAICompatClient("http://llm.test", "m"))
    assert ai.decide_timeout_s == 2.5


def test_missing_model_is_explicit_error(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="LOCAL_LLM_MODEL"):
        OpenAICompatClient.from_env()


def test_warmup_failure_does_not_raise():
    def h(b):
        raise httpx.ConnectError("refused")

    c, _ = client_with(h)
    run(LocalLlmInterpreter(c).warmup())  # 예외 없이 종료되어야 함


def test_schema_sent_unchanged():
    c, seen = client_with(lambda b: ollama_reply(MENU))
    run(c.generate("sys", "user", SCHEMA_CLASSIFY))
    assert seen[0][1]["format"] == SCHEMA_CLASSIFY


# 출력 형식 설정 -------------------------------------------------------------
def test_compact_instruction_added_by_default():
    c, seen = client_with(lambda b: ollama_reply({"text": "ok"}))
    run(c.generate("SYS", "user", {"type": "object"}))
    system = seen[0][1]["messages"][0]["content"]
    assert system.startswith("SYS") and system.endswith("한 줄로 씁니다.") and "format" in seen[0][1]


def test_prompt_schema_mode_sends_no_grammar():
    c, seen = client_with(lambda b: ollama_reply({"text": "ok"}), schema_mode="prompt")
    run(c.generate("SYS", "user", {"type": "object", "properties": {"text": {"type": "string"}}}))
    body = seen[0][1]
    assert "format" not in body and '응답 JSON 스키마: {"type":"object"' in body["messages"][0]["content"]


def test_prompt_mode_bad_json_raises_for_fallback():
    c, _ = client_with(lambda b: httpx.Response(200, json={"message": {"content": '{"kind": "menu", '}}), schema_mode="prompt")
    with pytest.raises(RuntimeError):
        run(c.generate("SYS", "user", {"type": "object"}, retries=0))


def test_compact_off_and_schema_env(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_COMPACT", "0")
    monkeypatch.setenv("LOCAL_LLM_SCHEMA", "prompt")
    c = OpenAICompatClient.from_env("m")
    assert c.compact is False and c.schema_mode == "prompt"
    with pytest.raises(ValueError):
        OpenAICompatClient("http://llm.test", "m", schema_mode="loose")


# 설정값 경계 사례·다시 올리기 ----------------------------------------------------
def test_blank_decide_timeout_falls_back(monkeypatch):
    """.env 의 'LOCAL_LLM_DECIDE_TIMEOUT_S=' (빈 값)은 GEMINI_DECIDE_TIMEOUT_S 로 대체 확인 (서버 시작 실패 없음)."""
    monkeypatch.setenv("LOCAL_LLM_DECIDE_TIMEOUT_S", "")
    monkeypatch.setenv("GEMINI_DECIDE_TIMEOUT_S", "3")
    assert LocalLlmInterpreter(OpenAICompatClient("http://llm.test", "m")).decide_timeout_s == 3.0


def test_openai_mode_requires_base_url(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_API", "openai")
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="LOCAL_LLM_BASE_URL"):
        OpenAICompatClient.from_env("m")


@pytest.mark.parametrize("raw, want", [("24h", "24h"), ("", "24h"), ("-1", -1), ("300", 300)])
def test_keep_alive_values(monkeypatch, raw, want):
    monkeypatch.setenv("LOCAL_LLM_KEEP_ALIVE", raw)
    assert OpenAICompatClient.from_env("m").keep_alive == want


def test_invalid_think_value_stops_startup(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_THINK", "none")
    with pytest.raises(ValueError, match="LOCAL_LLM_THINK"):
        OpenAICompatClient.from_env("m")


def test_failed_call_restarts_warmup_in_background():
    """모델이 내려가 판단 호출이 실패하면 긴 대기 시간의 미리 올리기를 백그라운드에서 한 번 다시 시작하는지 확인 (30초 간격)."""
    seen = []

    def h(b):
        seen.append(b)
        raise httpx.ReadTimeout("loading")

    c, _ = client_with(h)

    async def go():
        with pytest.raises(RuntimeError):
            await c.generate("sys", "user", {"type": "object"}, timeout_s=0.1, retries=0)
        await asyncio.sleep(0)      # 다시 올리기 작업 시작
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError):
            await c.generate("sys", "user", {"type": "object"}, timeout_s=0.1, retries=0)
        await asyncio.sleep(0.05)

    run(go())
    warm_calls = [b for b in seen if "잔액조회는 3번을 눌러 주십시오" in b["messages"][1]["content"]]
    assert len(warm_calls) == 1  # 두 번 실패해도 30초 안에는 한 번만
    assert c._warming is False


# 받아쓰기 고침만 따로 묻는 호출 (LLM_FIX_TRANSCRIPT=separate) ------------------------------------------
def test_side_call_failure_does_not_restart_warmup():
    """rewarm=False 로 부른 부가 호출(받아쓰기 고침)은 실패해도 미리 올리기 없음 확인.
    미리 올리기 요청이 다음 판단 호출과 겹치면 판단 대기 시간(기본 4초) 초과 우려."""
    seen = []

    def h(b):
        seen.append(b)
        raise httpx.ReadTimeout("slow")

    c, _ = client_with(h)

    async def go():
        with pytest.raises(RuntimeError):
            await c.generate("sys", "user", {"type": "object"}, timeout_s=0.1, retries=0, rewarm=False, max_tokens=128)
        await asyncio.sleep(0.05)

    run(go())
    assert len(seen) == 1 and seen[0]["options"]["num_predict"] == 128  # 재시도·미리 올리기 없음, 출력 상한 반영
    assert c._warming is False and c._warm_at == 0.0


def test_fix_prompt_failure_keeps_decisions_on_model(monkeypatch):
    """고침 호출이 실패해도 규칙 판단 전환(_slow_until) 없음, 다음 분류도 모델 사용 확인."""
    monkeypatch.setenv("LLM_FIX_TRANSCRIPT", "separate")
    calls = []

    def h(b):
        calls.append(b["messages"][0]["content"])
        if "음성 인식으로 받아 적은 글을 검토" in b["messages"][0]["content"]:
            raise httpx.ReadTimeout("slow")
        return ollama_reply(MENU)

    c, _ = client_with(h)
    it = LocalLlmInterpreter(c)

    async def go():
        r = await it.fix_prompt(ROOT, [], "menu")
        assert r["source"] == "error" and r["applied"] == []
        assert it._slow_until == 0.0
        p = await it.classify(ROOT)
        assert p.kind == "menu" and [o.key for o in p.options] == ["3", "0"]

    run(go())
    assert c._warming is False and c._warm_at == 0.0
    assert "fixes" not in calls[-1]  # separate 의 분류 지시문에는 고침 규칙 없음
