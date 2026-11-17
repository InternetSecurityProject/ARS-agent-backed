"""통화별 판단 해석기 조합 선택.

선택 항목
  판단 모델 interp : hybrid = JEV 나눔 판단 (번호 판단은 JEV, 민감정보·결과 요약 등은 로컬 gemma4)
                    local  = 로컬 gemma4 단독
                    som    = som 방식 JEV 단독 (해석기·탐색기는 som_mode.py).
                             받아쓰기 고침·결과 안내 판단은 해당 없음 (fix 는 0, result 는 jev 로 기록)
                    llm    = 로컬 모델 판단 (규칙 없음, llm_mode.py). LLM_COMBO=1 인 서버만 선택 가능. fix 는 0, result 는 local
                    jevllm = JEV 판단 + 규칙 자리만 로컬 모델 (jevllm_mode.py). 로컬·JEV 연결이 있고 JEVLLM_COMBO=1 인 서버만 선택 가능.
                             fix 는 0, result 는 jev 로 기록
                    서버 기본 판단 모델은 COMBO_DEFAULT_INTERP(hybrid|local|som|llm|jevllm, 선택 가능할 때만)로 변경 가능
  받아쓰기 고침 fix : 0 = 끔, 1 = 분류에 묶음 (분류 요청에 고침 질문 포함. hybrid 에서는 로컬 판단 안내에만 해당),
                    separate = 따로 묻기 (판단 뒤 로컬 모델에 별도 질문. 고침 결과는 화면 표시에만 적용)
  결과 안내 판단 result : hybrid 에서만. jev = '발송됩니다·접수되었습니다' 같은 결과 말만 있는 안내(민감정보 모양·값이 붙는 결과 말
                    없음, 입력 요청 전)도 JEV 가 판단 (로컬 사전 확인에서 개인정보로 보면 로컬, JEV 가 정하지 못하거나 일반 안내라고
                    하면 로컬 재판단), local = 결과 말이 있으면 로컬 판단.
                    local 판단 모델이면 늘 local. 기본값은 RESULT_JUDGE (비었으면 jev). 자세한 내용은 hybrid_interpreter 머리말
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
import os
import re

log = logging.getLogger("ars.combo")

# 화면 표시 이름. hybrid 의 JEV 버전 번호는 JEV 연결의 모델 이름(JEV_MODEL)에서 추출
# som 은 내부 값, 화면 이름은 'JEV v2'
INTERPS = {"hybrid": "JEV 1.14 + Local STT", "local": "all local", "som": "JEV v2", "llm": "로컬 모델 (규칙 없음)",
           "jevllm": "JEV + 로컬 (규칙 자리 로컬)"}
# STT 후보정: 1 = 판단(분류) 호출 안에서 함께 고침(고친 선택지 이름이 뒤 단계 판단에도 쓰임), separate = 판단 뒤 따로 고쳐 화면 안내만 바꿈,
# judge = 판단 앞에 따로 고치고 고친 글로 판단
FIXES = {"0": "끔", "1": "판단 중 후보정", "separate": "안내 표시 후보정", "judge": "후보정으로 판단"}
# 결과 안내 판단 (hybrid 만): 결과 말만 있는 안내를 JEV 가 판단할지, 로컬이 판단할지
RESULTS = {"jev": "JEV", "local": "로컬"}


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
    def __init__(self, local_client, jev_client, default_interp: str, default_fix: str, default_result: str | None = None,
                 som_client=None) -> None:
        self.local_client = local_client
        self.jev_client = jev_client                 # None: hybrid 선택 불가 (TYPESAFE_API_KEY 없음)
        self.som_client = som_client                 # som_jev 의 JevClient. None: som 선택 불가
        self.default_interp = default_interp
        self.default_fix = default_fix
        if default_result not in RESULTS:
            from hybrid_interpreter import result_judge_from_env
            default_result = result_judge_from_env()
        self.default_result = default_result         # hybrid 선택 시 결과 안내 판단 기본값
        env = os.environ.get("COMBO_DEFAULT_INTERP", "").strip().lower()
        if env and env in INTERPS and self._selectable(env):
            self.default_interp = env                # 요청에 조합이 없을 때(Android 앱 등)와 웹 앱 첫 선택값
        elif env:
            log.warning("COMBO_DEFAULT_INTERP=%s 는 이 서버에서 고를 수 없어 %s 를 기본으로 둔다", env, self.default_interp)
        self._caches: dict[tuple[str, str], tuple[dict, dict]] = {}
        self._slow = _SlowFlag()
        self._ex_slow = _SlowFlag()                  # jevllm 의 로컬 추출 전용 지연 표시 (공용 지연 표시와 별도)
        self._local_cls = None

    @classmethod
    def from_interp(cls, interp) -> "ComboFactory | None":
        """서버 설정으로 만든 해석기에서 모델 연결 추출. 조합 선택이 불가한 해석기(목·Gemini)면 None."""
        from hybrid_interpreter import HybridJevInterpreter
        from local_llm import LocalLlmInterpreter

        from som_mode import som_client_from_env

        if isinstance(interp, HybridJevInterpreter):
            return cls(interp.local.client, interp.jev.client, "hybrid", interp.fix_mode, interp.result_judge,
                       som_client=som_client_from_env())
        if isinstance(interp, LocalLlmInterpreter):
            jev = _jev_client_from_env()
            return cls(interp.client, jev, "local", interp.fix_mode, som_client=som_client_from_env() if jev else None)
        return None

    def _selectable(self, interp: str) -> bool:
        if interp == "llm":   # 로컬 모델 판단(규칙 없음): LLM_COMBO=1 인 서버만 선택 가능 (llm_mode.py)
            return self.local_client is not None and os.environ.get("LLM_COMBO", "0").strip() == "1"
        if interp == "jevllm":   # JEV 판단 + 규칙 자리만 로컬: 로컬·JEV 연결이 모두 있고 JEVLLM_COMBO=1 인 서버만 선택 가능 (jevllm_mode.py)
            return (self.local_client is not None and self.jev_client is not None
                    and os.environ.get("JEVLLM_COMBO", "0").strip() == "1")
        return {"hybrid": self.jev_client is not None, "som": self.som_client is not None}.get(interp, interp in INTERPS)

    def options(self) -> dict:
        """health 에 싣는 고를 수 있는 값과 기본값. result(결과 안내 판단)는 hybrid 선택 가능 서버만 포함."""
        interps = [k for k in INTERPS if self._selectable(k)]
        out = {"interp": [{"id": k, "label": self._hybrid_label() if k == "hybrid" else INTERPS[k]} for k in interps],
               "fix": [{"id": k, "label": v} for k, v in FIXES.items()],
               "default": {"interp": self.default_interp, "fix": self.default_fix}}
        if "hybrid" in interps:
            out["result"] = [{"id": k, "label": v} for k, v in RESULTS.items()]
            out["default"]["result"] = self.default_result
        return out

    def _hybrid_label(self) -> str:
        """'JEV 1.14 + Local STT' — 버전 번호는 JEV 모델 이름(jev-1.14.0)에서 추출. 추출 실패 시 기본 이름."""
        m = re.match(r"jev-(\d+)\.(\d+)", str(getattr(self.jev_client, "model", "") or ""))
        return f"JEV {m.group(1)}.{m.group(2)} + Local STT" if m else INTERPS["hybrid"]

    def normalize(self, interp, fix) -> tuple[str, str]:
        """요청 값 검사. 비어 있으면 서버 기본값. 틀리면 ComboError."""
        i = self.default_interp if interp in (None, "") else str(interp).strip().lower()
        f = self.default_fix if fix in (None, "") else str(fix).strip().lower()
        if i not in INTERPS:
            raise ComboError("interp 는 hybrid, local, som, llm, jevllm 중 하나입니다.")
        if i == "hybrid" and self.jev_client is None:
            raise ComboError("이 서버에는 JEV 연결 설정이 없어 hybrid 를 고를 수 없습니다.")
        if i == "som" and self.som_client is None:
            raise ComboError("이 서버에는 JEV 연결 설정이 없어 som 을 고를 수 없습니다.")
        if i == "llm" and not self._selectable("llm"):
            raise ComboError("이 서버에서는 로컬 모델(규칙 없음) 판단을 고를 수 없습니다.")
        if i == "jevllm" and not self._selectable("jevllm"):
            raise ComboError("이 서버에서는 JEV + 로컬(규칙 자리 로컬) 판단을 고를 수 없습니다.")
        if f not in FIXES:
            raise ComboError("fix 는 0, 1, separate 중 하나입니다.")
        if i == "som":
            return i, "0"   # som 방식(원본 해석기·탐색기 그대로)에는 받아쓰기 고침 없음
        # llm·jevllm 은 판단을 로컬 분류 호출이 하지 않으므로 '1'(분류 호출 안에서 함께 고침) 없음. 고침을 고르면 판단 뒤
        # 화면 자막만 따로 고치는 separate 로 적용 (판단 입력은 원문 그대로)
        if i in ("llm", "jevllm"):
            return i, ("separate" if f in ("1", "separate") else f)
        return i, f

    def normalize_result(self, interp: str, result) -> str:
        """결과 안내 판단 값 검사 (normalize 를 거친 interp). 비어 있으면 서버 기본값, local·llm 판단 모델이면 늘 local,
        som 이면 늘 jev (모든 안내를 JEV 가 판단), jevllm 이면 늘 jev (결과 말만 있는 안내도 JEV 가 판단, hybrid 의 jev 와 같음).
        틀리면 ComboError."""
        r = self.default_result if result in (None, "") else str(result).strip().lower()
        if r not in RESULTS:
            raise ComboError("result 는 jev 또는 local 입니다.")
        if interp == "hybrid":
            return r
        return "jev" if interp in ("som", "jevllm") else "local"   # llm 은 모두 로컬

    def build(self, interp: str, fix: str, result: str | None = None):
        """통화별 해석기 생성 (normalize·normalize_result 를 거친 값 기준. result 가 없으면 서버 기본값)."""
        from hybrid_interpreter import HybridJevInterpreter
        from jev_interpreter import JevInterpreter

        if interp == "som":
            from som_mode import new_som_interpreter
            return new_som_interpreter(self.som_client)
        lo = self._local(interp, fix)
        if interp == "llm":
            from llm_mode import LlmInterpreter
            # 메뉴 판단은 LlmInterpreter 가 정규식 없이, 그 밖의 일(목적 요약·결과 요약 등)은 같은 연결의 로컬 해석기(lo)가 담당
            it = LlmInterpreter(self.local_client, lo, cache=self._caches.setdefault(("llm-judge", "0"), ({}, {}))[0])
            return self._with_display_fix(it, lo, fix)
        if interp == "jevllm":
            from jevllm_mode import JevLlmInterpreter, JevLlmJev
            # 메뉴 판단은 JEV, 정규식 규칙이 판단하던 자리는 로컬 모델이 담당. 목적 요약·결과 요약 등과 로컬 추출 실패 때의 대체
            # 판단은 같은 연결의 로컬 해석기(lo)가 담당. 캐시는 llm 조합과 분리
            it = JevLlmInterpreter(self.local_client, lo, JevLlmJev(self.jev_client),
                                   cache=self._caches.setdefault(("jevllm-judge", "0"), ({}, {}))[0],
                                   ex_cache=self._caches.setdefault(("jevllm-extract", "0"), ({}, {}))[0],
                                   ex_slow=self._ex_slow)
            return self._with_display_fix(it, lo, fix)
        if interp == "hybrid":
            # 로컬 분류 캐시는 결과 안내 판단과 무관하게 같은 로컬 모델 답이므로 분리하지 않음 (판단 쪽만 달라짐)
            return HybridJevInterpreter(local=lo, jev=JevInterpreter(self.jev_client),
                                        result_judge=result if result in RESULTS else self.default_result)
        return lo

    def _local(self, interp: str, fix: str):
        """조합 하나가 쓸 로컬 해석기 (캐시는 판단 모델·고침 값별로 나눔)."""
        if self._local_cls is None:
            self._local_cls = _call_local_cls()
        lo = self._local_cls(self.local_client, self._slow)
        lo._cache, lo._fix_cache = self._caches.setdefault((interp, fix), ({}, {}))
        lo.fix_mode, lo.fix_transcript = fix, fix == "1"
        return lo

    @staticmethod
    def _with_display_fix(it, lo, fix: str):
        """llm·jevllm 해석기에 받아쓰기 고침 부착: 세션은 해석기의 fix_mode·fix_prompt 를 보고 separate 면 판단 뒤 자막만,
        judge 면 판단 앞에 고쳐 고친 글로 판단."""
        if fix in ("separate", "judge"):
            it.fix_mode, it.fix_prompt = fix, lo.fix_prompt
        return it


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
