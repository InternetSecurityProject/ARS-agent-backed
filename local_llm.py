"""
로컬 LLM 해석기 (Interpreter 프로토콜 구현)

로컬에서 실행하는 LLM 서버(Ollama · llama.cpp server · vLLM) 호출. 프롬프트·응답 스키마·실패 시 대체 규칙은
gemini_interpreter 와 동일, 모델 호출부만 교체 (GeminiInterpreter 에 OpenAICompatClient 전달).

.env 예
  INTERPRETER=local_llm:LocalLlmInterpreter
  LOCAL_LLM_MODEL=qwen3:8b                     (필수, LLM 서버에 올라가 있는 모델 이름)
  LOCAL_LLM_API=ollama                         ollama: Ollama 고유 API(/api/chat, 기본) / openai: OpenAI 호환 API(/v1/chat/completions)
                                               Ollama 서버에는 ollama 방식 사용 (OpenAI 방식으로는 문맥 길이·생각 끔의 요청별 지정 불가)
  LOCAL_LLM_BASE_URL=http://127.0.0.1:11434    LLM 서버 주소. ollama 방식은 비우면 이 값. openai 방식은 필수, /v1 까지 포함
                                               (예: http://<호스트>:<포트>/v1)
  LOCAL_LLM_TIMEOUT_S=6                        판단이 아닌 호출(목적 요약·결과 요약 등)의 대기 시간
  LOCAL_LLM_DECIDE_TIMEOUT_S=4                 번호를 눌러야 하는 판단(분류·점수·확인 질문)의 대기 시간
  LOCAL_LLM_THINK=0                            생각(추론) 모드. 0=끔(기본), 1=켬, low/medium/high=수준 지정, 빈 값=모델 기본
  LOCAL_LLM_NUM_CTX=8192                       ollama 방식 문맥 길이. 짧으면 지시문 앞부분 잘림
  LOCAL_LLM_KEEP_ALIVE=24h                     ollama 방식 모델 상주 시간 (마지막 호출 뒤 이 시간 동안 모델 유지). 숫자는 초, -1 은 계속 상주
  LOCAL_LLM_COMPACT=1                          JSON 을 공백·줄바꿈 없는 한 줄로 쓰도록 지시 (1=켬, 기본). 출력 토큰 감소
  LOCAL_LLM_SCHEMA=grammar                     grammar: LLM 서버의 스키마 기반 출력 제한 (기본, 형식 보장)
                                               prompt: 스키마를 지시문에 글로 넣고 출력 제한 없음 (더 빠름, 형식 오류는 대체 경로)
                                               생각 끔을 지키지 않는 모델(qwen3 계열)에는 prompt 방식 사용 금지 (아래 참고)

안전 원칙 (gemini_interpreter 와 동일)
  - 호출 실패·시간 초과·형식 오류는 예외로 전달. GeminiInterpreter 가 규칙 분류·재질문 값으로 대체.
  - 응답은 기본적으로 JSON 스키마로 제한 (서버 쪽 문법 제한). 스키마 제한 없이 보내면 think=false 에도
    본문에 긴 추론 문장을 써서 시간 초과될 수 있는 모델(qwen3 계열) 존재. 스키마 제한 시 JSON 만 출력.
    그래도 생각 문장(<think>)이 섞여 오면 제거 후 해석.
  - 서버 시작 시 모델 미리 올리기(warmup). 첫 판단이 모델 로드 시간 때문에 시간 초과되는 문제 방지.
    호출 실패 시(LLM 서버 늦은 시작, 모델 내려감) 긴 대기 시간의 미리 올리기를 백그라운드에서 다시 시작.
    Ollama 는 요청이 끊기면 모델 로드도 멈추므로, 짧은 판단 호출(기본 4초)만으로는 모델 로드 완료 불가.
  - 빈 설정값(.env 의 'X=')은 기본값으로 처리. 잘못된 LOCAL_LLM_THINK 값은 서버 시작 시 오류로 중단.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time

from gemini_interpreter import GeminiInterpreter

log = logging.getLogger("local_llm")
_BACKGROUND: set = set()  # 백그라운드 다시 올리기 작업

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)

# 한 줄 JSON 지시: 들여쓴 JSON 대비 출력 토큰·응답 시간 감소.
# 스키마 제한 디코딩은 토큰당 속도도 느림 (prompt 방식이 더 빠른 이유)
COMPACT = "\n- JSON 은 공백과 줄바꿈 없이 한 줄로 씁니다."


def parse_json(text: str) -> dict:
    """모델 응답 문장 → dict. 생각 문장·코드 블록 표시 제거, 객체가 아니면 예외."""
    t = _THINK_BLOCK.sub("", text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    out = json.loads(t)
    if not isinstance(out, dict):
        raise ValueError(f"JSON 객체가 아님: {type(out).__name__}")
    return out


def _think_value(raw: str):
    v = raw.strip().lower()
    if v in ("", "default"):
        return None
    if v in ("0", "false", "off", "no"):
        return False
    if v in ("1", "true", "on", "yes"):
        return True
    if v in ("low", "medium", "high"):
        return v  # 수준을 받는 모델 (gpt-oss 등)
    # 'none'·'minimal' 같은 값은 모델의 400 거절 대상. 그 뒤 think 를 빼고 재전송하면 생각 모드가 오히려 켜짐.
    # 따라서 잘못된 값은 서버 시작 시 중단
    raise ValueError(f"LOCAL_LLM_THINK 값이 올바르지 않음: {raw!r} (0 / 1 / low / medium / high / 빈 값)")


def _keep_alive_value(raw: str):
    """Ollama keep_alive: 숫자는 초(음수 = 계속 상주), 그 밖에는 기간 문자열(24h 등). 빈 값은 24h.
    '-1' 을 문자열로 보내면 Ollama 가 기간으로 읽지 못해 모든 요청이 HTTP 400 이 되므로 숫자는 정수로 변환."""
    v = raw.strip()
    if not v:
        return "24h"
    try:
        return int(v)
    except ValueError:
        return v


def _env_first(*names: str, default: str) -> str:
    """여러 설정 가운데 비어 있지 않은 첫 값 (.env 의 'X=' 처럼 빈 값은 미설정으로 처리)."""
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return default


class LlmHttpError(RuntimeError):
    """LLM 서버의 HTTP 오류 응답 (상태 코드 보관). RuntimeError 하위 클래스이므로 RuntimeError 처리에 그대로 포함."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class OpenAICompatClient:
    """GeminiClient 와 같은 generate() 계약. 실패 시 예외 발생."""

    def __init__(self, base_url: str, model: str, *, api: str = "ollama", timeout_s: float = 6.0, retries: int = 1,
                 think=False, num_ctx: int = 8192, keep_alive: str = "24h", api_key: str = "",
                 compact: bool = True, schema_mode: str = "grammar") -> None:
        if api not in ("ollama", "openai"):
            raise ValueError("LOCAL_LLM_API 는 ollama 또는 openai")
        if schema_mode not in ("grammar", "prompt"):
            raise ValueError("LOCAL_LLM_SCHEMA 는 grammar 또는 prompt")
        if not model:
            raise RuntimeError("LOCAL_LLM_MODEL 이 .env 에 없습니다. (INTERPRETER=local_llm:LocalLlmInterpreter 사용 시 필요)")
        self.base_url, self.model, self.api = base_url.rstrip("/"), model, api
        self.timeout_s, self.retries, self.think = timeout_s, retries, think
        self.num_ctx, self.keep_alive, self.api_key = num_ctx, keep_alive, api_key
        self.compact, self.schema_mode = compact, schema_mode
        self._http = None
        self.last: dict = {}  # 마지막 호출의 시간·토큰 수 (벤치마크·로그용)
        self._warming = False  # 미리 올리기 진행 여부 (실패 시 다시 올리기 중복 방지)
        self._warm_at = 0.0    # 마지막으로 다시 올리기를 시작한 시각
        self._inflight = 0     # 진행 중인 generate 수 (동시 통화 중 다시 올리기 생략 판단용)

    @classmethod
    def from_env(cls, model: str | None = None) -> "OpenAICompatClient":
        """LOCAL_LLM_* 설정으로 생성. model 을 주면 LOCAL_LLM_MODEL 대신 사용 (벤치마크용)."""
        api = _env_first("LOCAL_LLM_API", default="ollama").lower()
        base = os.environ.get("LOCAL_LLM_BASE_URL", "").strip()
        if not base:
            if api != "ollama":
                # OpenAI 호환 서버는 표준 포트 없음 (vLLM 기본 포트 8000 도 app.py 기본 포트와 겹침) → 주소 지정 필수
                raise RuntimeError("LOCAL_LLM_API=openai 이면 LOCAL_LLM_BASE_URL 을 지정해야 합니다 (예 http://127.0.0.1:8080/v1)")
            base = "http://127.0.0.1:11434"
        return cls(base, model or os.environ.get("LOCAL_LLM_MODEL", "").strip(), api=api,
                   timeout_s=float(_env_first("LOCAL_LLM_TIMEOUT_S", default="6")),
                   think=_think_value(os.environ.get("LOCAL_LLM_THINK", "0")),
                   num_ctx=int(_env_first("LOCAL_LLM_NUM_CTX", default="8192")),
                   keep_alive=_keep_alive_value(os.environ.get("LOCAL_LLM_KEEP_ALIVE", "24h")),
                   api_key=os.environ.get("LOCAL_LLM_API_KEY", ""),
                   compact=os.environ.get("LOCAL_LLM_COMPACT", "1").strip() != "0",
                   schema_mode=os.environ.get("LOCAL_LLM_SCHEMA", "grammar").strip().lower() or "grammar")

    # 요청 본문 ---------------------------------------------------------------
    def _body(self, system: str, user: str, schema: dict | None, max_tokens: int = 1024) -> tuple[str, dict]:
        if schema is not None and self.schema_mode == "prompt":
            # 서버의 출력 제한이 없으므로 형식을 글로 안내. 받은 뒤 parse_json·해석기 검증으로 거름
            system += "\n- 응답 JSON 스키마: " + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            schema = None
        if self.compact:
            system += COMPACT
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.api == "ollama":
            # 반복 벌점 끔. 일부 모델은 기본값으로 presence_penalty·반복 벌점이 켜져 있어, JSON 의 같은 키 이름
            # ("key", "label" …)을 되풀이할 때마다 벌점을 받아 형식·내용이 흐트러질 우려
            # shift=False: 입력이 문맥 길이를 넘으면 Ollama 가 가운데를 조용히 잘라내는 대신 HTTP 400 반환 (→ 대체 경로).
            # 이 값과 num_ctx 는 요청마다 바꾸면 모델을 다시 올리므로 항상 같은 값 전송
            body = {"model": self.model, "messages": messages, "stream": False, "keep_alive": self.keep_alive, "shift": False,
                    "options": {"temperature": 0, "num_ctx": self.num_ctx, "num_predict": max_tokens,
                                "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1.0}}
            if schema is not None:
                body["format"] = schema
            if self.think is not None:
                body["think"] = self.think
            return f"{self.base_url}/api/chat", body
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                "presence_penalty": 0, "frequency_penalty": 0}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "result", "strict": True, "schema": schema}}
        if self.think is False:
            # vLLM·SGLang 의 Qwen 계열 생각 끔. 미지원 서버는 이 값 무시
            body["chat_template_kwargs"] = {"enable_thinking": False}
        elif isinstance(self.think, str):
            body["reasoning_effort"] = self.think
        return f"{self.base_url}/chat/completions", body

    def _content(self, data: dict) -> str:
        if self.api == "ollama":
            msg = data.get("message") or {}
            self.last.update(prompt_tokens=data.get("prompt_eval_count"), output_tokens=data.get("eval_count"),
                             load_ms=round((data.get("load_duration") or 0) / 1e6))
            return msg.get("content", "")
        u = data.get("usage") or {}
        self.last.update(prompt_tokens=u.get("prompt_tokens"), output_tokens=u.get("completion_tokens"))
        return data["choices"][0]["message"].get("content") or ""

    # 호출 ------------------------------------------------------------------
    async def generate(self, system: str, user: str, schema: dict, timeout_s: float | None = None,
                       retries: int | None = None, rewarm: bool = True, max_tokens: int | None = None) -> dict:
        """rewarm=False: 실패해도 모델 다시 올리기 없음 (판단이 아닌 부가 호출용. 다시 올리기 요청이
        다음 판단 호출과 겹치면 판단 대기 시간 초과). max_tokens: 출력 상한 (기본 1024).
        진행 중인 요청 수 집계: 다른 요청이 처리 중이면 모델은 이미 올라가 있으므로, 시간 초과는 동시 통화의
        모델 공유로 인한 지연일 가능성. 이 경우 다시 올리기(180초짜리 요청) 생략 (_rewarm)."""
        self._inflight = getattr(self, "_inflight", 0) + 1
        try:
            return await self._generate(system, user, schema, timeout_s, retries, rewarm, max_tokens)
        finally:
            self._inflight -= 1

    async def _generate(self, system: str, user: str, schema: dict, timeout_s: float | None,
                        retries: int | None, rewarm: bool, max_tokens: int | None) -> dict:
        import httpx

        if self._http is None:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            self._http = httpx.AsyncClient(headers=headers, timeout=self.timeout_s)
        url, body = self._body(system, user, schema, max_tokens or 1024)
        last: Exception | None = None
        for attempt in range((self.retries if retries is None else retries) + 1):
            t0 = time.perf_counter()
            self.last = {}
            try:
                r = await self._http.post(url, json=body, timeout=timeout_s or self.timeout_s)
                if r.status_code == 400 and "think" in body and "think" in r.text.lower():
                    # 생각 모드 미지원 모델: 옵션을 빼고 재전송 (재시도 횟수와 무관)
                    log.info("모델 %s 은 think 옵션을 받지 않음 — 빼고 보냄", self.model)
                    self.think = None
                    body.pop("think", None)
                    r = await self._http.post(url, json=body, timeout=timeout_s or self.timeout_s)
                if r.status_code >= 400:
                    raise LlmHttpError(r.status_code, f"LLM 서버 HTTP {r.status_code}: {r.text[:200]}")
                out = parse_json(self._content(r.json()))
                ms = (time.perf_counter() - t0) * 1000
                self.last["ms"] = round(ms)
                log.info("local-llm %dms (입력 %s, 출력 %s 토큰)", ms, self.last.get("prompt_tokens"), self.last.get("output_tokens"))
                if self.api == "ollama" and self.last.get("prompt_tokens") and self.last["prompt_tokens"] >= self.num_ctx:
                    log.warning("입력이 문맥 길이(%d)에 닿음 — 지시문이 잘렸을 수 있음. LOCAL_LLM_NUM_CTX 를 늘릴 것", self.num_ctx)
                return out
            except Exception as e:  # 시간 초과·연결 실패·HTTP 오류·JSON 오류
                last = e
                log.warning("local-llm 실패 (%d회): %s", attempt + 1, (str(e) or type(e).__name__)[:200])
        if rewarm:
            self._rewarm()
        raise RuntimeError(f"로컬 LLM 호출 실패: {last}") from last   # 원래 예외(시간 초과·JSON 오류 등)를 __cause__ 로 남김

    def _rewarm(self) -> None:
        """호출 실패 시 긴 대기 시간의 미리 올리기를 백그라운드에서 다시 시작 (30초에 한 번까지).
        모델이 내려가 있으면(LLM 서버 늦은 시작·재시작, 상주 시간 만료) 짧은 판단 호출(기본 4초)은 로드 완료 전에 끊기고,
        Ollama 는 요청이 끊기면 로드도 멈춤. 따라서 판단 호출만으로는 모델 로드 불가."""
        if self._warming or time.monotonic() - self._warm_at < 30 or getattr(self, "_inflight", 0) > 1:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._warm_at = time.monotonic()
        task = loop.create_task(self.warmup())
        _BACKGROUND.add(task)  # 작업 종료 전에 사라지지 않도록 참조 보관
        task.add_done_callback(_BACKGROUND.discard)

    async def warmup(self) -> None:
        """모델 로드와 지시문 앞부분 캐시 준비. 실패해도 서버는 계속 실행, 이후 호출 실패 시 다시 시도 (_rewarm)."""
        from gemini_interpreter import CLASSIFY_SCORE, SCHEMA_CLASSIFY_SCORE

        if self._warming:
            return
        self._warming = True
        t0 = time.perf_counter()
        try:
            await self.generate(CLASSIFY_SCORE, "<goal>\n잔액 조회\n</goal>\n<path>\n[]\n</path>\n<ars>\n잔액조회는 3번을 눌러 주십시오.\n</ars>",
                                SCHEMA_CLASSIFY_SCORE, timeout_s=180, retries=0)
            log.info("로컬 LLM 준비 완료: %s (%.1fs)", self.model, time.perf_counter() - t0)
        except Exception as e:
            log.warning("로컬 LLM 미리 올리기 실패 (%s): %s", self.model, e)
        finally:
            self._warming = False


class LocalLlmInterpreter(GeminiInterpreter):
    """GeminiInterpreter 와 같은 판단 규칙, 모델 호출만 로컬 LLM 서버 사용."""

    def __init__(self, client: OpenAICompatClient | None = None) -> None:
        super().__init__(client=client or OpenAICompatClient.from_env())
        self.decide_timeout_s = float(_env_first("LOCAL_LLM_DECIDE_TIMEOUT_S", "GEMINI_DECIDE_TIMEOUT_S", default="4"))

    async def warmup(self) -> None:
        warm = getattr(self.client, "warmup", None)
        if warm:
            await warm()
