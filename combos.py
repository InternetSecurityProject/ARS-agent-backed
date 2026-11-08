"""통화별 판단 해석기 조합 선택.

선택 항목
  판단 모델 interp : hybrid = JEV 나눔 판단 (번호 판단은 JEV, 민감정보·결과 요약 등은 로컬 gemma4)
                    local  = 로컬 gemma4 단독
  받아쓰기 고침 fix : 0 = 끔, 1 = 분류에 묶음 (분류 요청에 고침 질문 포함. hybrid 에서는 로컬 판단 안내에만 해당),
                    separate = 따로 묻기 (판단 뒤 로컬 모델에 별도 질문. 고침 결과는 화면 표시에만 적용)
선택 대상 아님
  받아쓰기(STT): 서버 설정(로컬 Whisper) 사용. JEV 중요 선택 규칙(JEV_RULE_CRITICAL): 서버 설정(기본 켬) 사용.
  Gemini 판단: 선택 불가.

통화마다 해석기 객체를 새로 생성하고 모델 연결(로컬 LLM 클라이언트·JEV 클라이언트)만 공유.
따라서 통화별 목적·최근 메뉴·'입력 요청 뒤' 상태·번호 없이 들린 선택지가 다른 통화와 섞이지 않음.
분류 결과 캐시((안내, 목적) → 결과)와 고침 제안 캐시는 같은 조합끼리 공유: 같은 ARS 를 같은 목적으로 다시 걸 때
이미 들은 안내의 재질문 방지.
고침 방식이 다르면 분류 지시문·응답 모양이 달라 캐시 분리.
모델 지연 표시(급한 판단이 시간을 넘긴 뒤 20초 동안 규칙으로 판단, _slow_until)는 모델 연결의 상태이므로 모든 통화가 공유
(다음 통화로도 이어짐).

요청에 조합이 없으면(Android 앱 등) 서버 설정(INTERPRETER, LLM_FIX_TRANSCRIPT)과 같은 조합으로 생성.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("ars.combo")

# 화면 표시 이름. hybrid 의 JEV 버전 번호는 JEV 연결의 모델 이름(JEV_MODEL)에서 추출
INTERPS = {"hybrid": "JEV 1.14 + Local STT", "local": "all local"}
# STT 후보정: 1단계 = 분류 요청에 고침을 함께 묻기, 2단계 = 판단 뒤 따로 묻기
FIXES = {"0": "끔", "1": "1단계", "separate": "2단계"}


class ComboError(ValueError):
    """잘못된 조합 값 요청 (문구는 사용자에게 그대로 표시)."""


class _SlowFlag:
    """모델 지연 표시 (monotonic 시각). 같은 모델 연결을 쓰는 통화별 해석기가 공유."""

    def __init__(self) -> None:
        self.until = 0.0


def _call_local_cls():
    from local_llm import LocalLlmInterpreter

    class CallLocalLlm(LocalLlmInterpreter):
        """통화별 로컬 해석기. _slow_until 만 공유 표시로 연결 (그 밖의 상태는 통화별)."""

        def __init__(self, client, slow: _SlowFlag) -> None:
            self._slow = None
            super().__init__(client)   # 부모 초기화의 _slow_until = 0.0 대입은 공유 표시에 영향 없음 (이 시점 _slow 는 None)
            self._slow = slow

        @property
        def _slow_until(self) -> float:
            return self._slow.until if self._slow is not None else 0.0

        @_slow_until.setter
        def _slow_until(self, value: float) -> None:
            if getattr(self, "_slow", None) is not None:
                self._slow.until = value

    return CallLocalLlm


class ComboFactory:
    def __init__(self, local_client, jev_client, default_interp: str, default_fix: str) -> None:
        self.local_client = local_client
        self.jev_client = jev_client                 # None: hybrid 선택 불가 (TYPESAFE_API_KEY 없음)
        self.default_interp = default_interp
        self.default_fix = default_fix
        self._caches: dict[tuple[str, str], tuple[dict, dict]] = {}
        self._slow = _SlowFlag()
        self._local_cls = None

    @classmethod
    def from_interp(cls, interp) -> "ComboFactory | None":
        """서버 설정으로 만든 해석기에서 모델 연결 추출. 조합 선택이 불가한 해석기(목·Gemini)면 None."""
        from hybrid_interpreter import HybridJevInterpreter
        from local_llm import LocalLlmInterpreter

        if isinstance(interp, HybridJevInterpreter):
            return cls(interp.local.client, interp.jev.client, "hybrid", interp.fix_mode)
        if isinstance(interp, LocalLlmInterpreter):
            return cls(interp.client, _jev_client_from_env(), "local", interp.fix_mode)
        return None

    def options(self) -> dict:
        """health 에 싣는 고를 수 있는 값과 기본값."""
        interps = [k for k in INTERPS if k != "hybrid" or self.jev_client is not None]
        return {"interp": [{"id": k, "label": self._hybrid_label() if k == "hybrid" else INTERPS[k]} for k in interps],
                "fix": [{"id": k, "label": v} for k, v in FIXES.items()],
                "default": {"interp": self.default_interp, "fix": self.default_fix}}

    def _hybrid_label(self) -> str:
        """'JEV 1.14 + Local STT' — 버전 번호는 JEV 모델 이름(jev-1.14.0)에서 추출. 추출 실패 시 기본 이름."""
        m = re.match(r"jev-(\d+)\.(\d+)", str(getattr(self.jev_client, "model", "") or ""))
        return f"JEV {m.group(1)}.{m.group(2)} + Local STT" if m else INTERPS["hybrid"]

    def normalize(self, interp, fix) -> tuple[str, str]:
        """요청 값 검사. 비어 있으면 서버 기본값. 틀리면 ComboError."""
        i = self.default_interp if interp in (None, "") else str(interp).strip().lower()
        f = self.default_fix if fix in (None, "") else str(fix).strip().lower()
        if i not in INTERPS:
            raise ComboError("interp 는 hybrid 또는 local 입니다.")
        if i == "hybrid" and self.jev_client is None:
            raise ComboError("이 서버에는 JEV 연결 설정이 없어 hybrid 를 고를 수 없습니다.")
        if f not in FIXES:
            raise ComboError("fix 는 0, 1, separate 중 하나입니다.")
        return i, f

    def build(self, interp: str, fix: str):
        """통화별 해석기 생성 (normalize 거친 값 기준)."""
        from hybrid_interpreter import HybridJevInterpreter
        from jev_interpreter import JevInterpreter

        if self._local_cls is None:
            self._local_cls = _call_local_cls()
        lo = self._local_cls(self.local_client, self._slow)
        lo._cache, lo._fix_cache = self._caches.setdefault((interp, fix), ({}, {}))
        lo.fix_mode, lo.fix_transcript = fix, fix == "1"
        if interp == "hybrid":
            return HybridJevInterpreter(local=lo, jev=JevInterpreter(self.jev_client))
        return lo


def _jev_client_from_env():
    """로컬 단독이 기본인 서버에서도 JEV 선택이 가능하도록 JEV 연결 생성. 키가 없으면 None."""
    from jev_interpreter import JevInterpreter

    try:
        return JevInterpreter().client
    except RuntimeError:
        log.info("TYPESAFE_API_KEY 가 없어 JEV 나눔 판단은 고를 수 없다")
        return None
    except Exception as e:  # JEV_TIMEOUT_S 가 빈 값·숫자가 아님 등. 이 경우에도 로컬 단독 서버는 정상 시작 필요
        log.warning("JEV 연결 설정을 읽지 못해 JEV 나눔 판단은 고를 수 없다: %s", type(e).__name__)
        return None
