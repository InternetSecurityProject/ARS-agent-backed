"""
AI 슬롯필링 연동 지점 (팀원 담당)

서버는 이 파일의 Interpreter 프로토콜만 호출한다. 판단 로직(목적 이해, 안내 분류, 선택지 일치율,
요약 문장 생성)은 AI 담당 팀원이 구현하며, 입출력 형식은 API 명세서 v0.3 의 9장을 따른다.

현재 기본값은 MockInterpreter 이다. 판단 로직 없이 mock_ai.json 의 목 데이터만 돌려준다.
팀원 구현이 들어오면 .env 에 INTERPRETER=모듈명:클래스명 을 지정하여 교체한다. 서버 코드는 바뀌지 않는다.

입력 규칙
  - 민감정보 입력값은 이 모듈에 전달되지 않는다.
  - ARS 가 입력값을 다시 읽어 주는 구간의 문장은 숫자를 '*' 로 가린 뒤 전달된다.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

HERE = Path(__file__).parent


def _norm(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", text)


# ---------------------------------------------------------------------------
# 데이터 구조 (서버와 AI 사이의 계약, 명세서 9.2)
# ---------------------------------------------------------------------------
@dataclass
class Option:
    key: str
    label: str                    # ARS 가 말한 선택지 이름 그대로 (예: "상담직원 연결")
    category: str | None = None   # [AI] 선택지를 정규화한 카테고리 (예: "상담원 연결")

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "category": self.category}


@dataclass(frozen=True)
class RouteAssessment:
    """Optional internal routing extension, separate from menu-match scores.

    unknown is fail-safe: a failed inference is not evidence of service absence.
    """
    status: str = "unknown"  # proceed / needs_info / unavailable / unknown
    confidence: float = 0.0
    clarification_keys: tuple[str, ...] = ()  # Independently verified user-detail alternatives, internal only.
    clarification_dimension: str = ""  # purpose can require a free answer without two menu choices.


# Prompt.kind 값 (명세서 9.2)
#   menu          번호 선택 메뉴
#   confirm       확인 질문 ("맞으면 1번")
#   sensitive     민감정보 입력 요청 (주민번호·계좌번호·비밀번호 등)
#   result        조회 결과·처리 완료 안내
#   agent_wait    상담원 연결 대기 안내
#   agent_human   상담원 육성 응답
#   input_error   입력 오류·입력 시간 초과 안내
#   voice_request 음성으로 답하라는 요청
#   out_of_hours  업무시간 외 안내
#   message_sent  문자·알림톡 발송 후 종료 안내
#   rejected      본인 확인 실패 등 거절
#   info          그 밖의 안내 (판단 불필요)
@dataclass
class Prompt:
    kind: str
    text: str
    options: list[Option] = field(default_factory=list)   # 업무 선택지 (아래 이동 키 제외)
    back_key: str | None = None      # 이전 메뉴
    home_key: str | None = None      # 처음으로
    repeat_key: str | None = None    # 다시 듣기
    agent_key: str | None = None     # 상담원 연결 (options 에도 포함)
    label: str | None = None         # sensitive: 입력 항목명
    terminator: str | None = None    # sensitive: 입력 뒤 종료 키
    critical: bool = False           # 결제·계약·해지 최종 승인 등
    readback_of: str | None = None   # confirm: 입력값 되읽기 확인이면 그 입력 항목명 (경로 표시용, 값은 아님)

    def all_options(self) -> list[Option]:
        out = list(self.options)
        for key, label in ((self.back_key, "이전 메뉴"), (self.home_key, "처음으로"), (self.repeat_key, "다시 듣기")):
            if key and all(o.key != key for o in out):
                out.append(Option(key, label))
        return out

    def label_of(self, key: str) -> str:
        return next((o.label for o in self.all_options() if o.key == key), key)

    def category_of(self, key: str) -> str:
        o = next((o for o in self.all_options() if o.key == key), None)
        return (o.category or o.label) if o else key

    def signature(self) -> str:
        return "|".join(sorted(f"{o.key}:{_norm(o.label)}" for o in self.options))

    @classmethod
    def from_dict(cls, text: str, d: dict) -> "Prompt":
        return cls(kind=d.get("kind", "info"), text=text,
                   options=[Option(str(o["key"]), o["label"], o.get("category")) for o in d.get("options", [])],
                   back_key=d.get("back_key"), home_key=d.get("home_key"), repeat_key=d.get("repeat_key"),
                   agent_key=d.get("agent_key"), label=d.get("label"), terminator=d.get("terminator"),
                   critical=bool(d.get("critical", False)))


class Interpreter(Protocol):
    """[AI 담당] 서버가 호출하는 7개 함수. 입출력은 명세서 9.1 표를 따른다."""

    async def summarize_goal(self, goal: str) -> str:
        """AI 가 이해한 목적을 한 문장으로 요약 (승인 질문 아님). 예: '수원 오늘 날씨를 찾아드릴게요.'"""

    async def classify(self, text: str) -> Prompt:
        """ARS 안내 한 단락을 분류하고 선택지를 추출한다.
        선택지마다 ARS 가 말한 이름(label)을 카테고리(category)로 정규화한다. 미리 정한 단어 목록이 아니라
        그 단계에서 들은 선택지를 매번 정규화한다."""

    async def score(self, goal: str, options: list[Option], path: list[str] | None = None) -> dict[str, float]:
        """사용자 목적을 이 단계 선택지의 정규화 카테고리에 맞춘 신뢰도 (선택지 키 → 0~1).
        서버는 0.50 이상을 일치, 0.30~0.50 을 불확실(low_confidence 재질문), 0.30 미만을 불일치로 본다.
        path: 지금까지 선택한 메뉴의 카테고리 (첫 메뉴부터 순서대로)."""

    async def answer_confirm(self, goal: str, prompt: Prompt) -> str | None:
        """확인 질문에 누를 키. 판단할 수 없으면 None (사용자에게 재질문)."""

    async def merge_answer(self, goal: str, question: str, answer: str) -> str:
        """재질문 답변을 목적에 반영한 새 목적."""

    async def summarize_result(self, goal: str, captions: list[str]) -> dict:
        """업무 결과. {"summary": 한 문장, "key_info": [{"label": 항목, "value": 값}, ...]}"""

    async def handoff_script(self, goal: str, path_texts: list[str]) -> str:
        """상담사에게 전달할 문장."""


# ---------------------------------------------------------------------------
# 목 구현: mock_ai.json 의 값을 그대로 돌려준다 (판단 로직 없음)
# ---------------------------------------------------------------------------
class MockInterpreter:
    def __init__(self, path: str | Path = HERE / "mock_ai.json") -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.goals: dict[str, dict] = data.get("goals", {})
        self.prompts = {_norm(t): (t, d) for t, d in data.get("prompts", {}).items()}

    def _goal(self, goal: str) -> dict:
        return self.goals.get(goal.strip(), {})

    async def summarize_goal(self, goal: str) -> str:
        return self._goal(goal).get("summary") or f"'{goal.strip()}' 업무를 찾아 드릴게요."

    async def classify(self, text: str) -> Prompt:
        hit = self.prompts.get(_norm(text))
        return Prompt.from_dict(text.strip(), hit[1]) if hit else Prompt(kind="info", text=text.strip())

    async def score(self, goal: str, options: list[Option], path: list[str] | None = None) -> dict[str, float]:
        match = self._goal(goal).get("match", {})  # 목 데이터: 목적별 카테고리 일치율 표
        return {o.key: float(match.get(o.category or o.label, 0.0)) for o in options}

    async def answer_confirm(self, goal: str, prompt: Prompt) -> str | None:
        return None

    async def merge_answer(self, goal: str, question: str, answer: str) -> str:
        return answer.strip()  # 목 데이터: 답변 문장을 새 목적으로 사용 (goals 표에서 조회)

    async def summarize_result(self, goal: str, captions: list[str]) -> dict:
        g = self._goal(goal)
        return {"summary": g.get("result_summary") or (captions[-1] if captions else "결과 안내를 받지 못했습니다."),
                "key_info": g.get("key_info", [])}

    async def handoff_script(self, goal: str, path_texts: list[str]) -> str:
        route = " > ".join(path_texts) if path_texts else "첫 메뉴"
        return f"{goal.strip()} 건으로 연락드렸습니다. ARS 에서 {route} 까지 진행했습니다."

    def goal_examples(self) -> list[str]:
        return list(self.goals)


def load_interpreter(spec: str) -> Interpreter:
    import importlib

    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls or "MockInterpreter")()
