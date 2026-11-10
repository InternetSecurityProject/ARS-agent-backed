"""som 방식 판단 조합 (combo.interp = som).

som_jev 패키지의 해석기 UnifiedJevInterpreter 와 탐색기 Navigator 를 서버 세션에 연결함.
해석·탐색 판단은 som_jev 코드 그대로이고, 세션은 som_jev 해석기용 호출 흐름만 따름
(session.py 의 system_recovery 분기: 이력과 함께 분류 classify_history, 탐색기의 'recover' 판단과 시스템 재판단·다시 듣기 복구,
확인 질문의 경로 평가). 받아쓰기(STT)·번호 여러 개·웹 앱·통화 기록·민감정보 입력과 되읽기 가림은 서버 것을 사용.

hybrid 조합과 다른 점
- 모든 안내 단락과 목적 원문을 JEV 로 전달. 로컬 사전 확인(개인정보)·민감정보 모양 규칙·입력 요청 뒤 로컬 판단 없음.
  (서버 세션이 되읽기 숫자를 가린 문장을 넘기는 것은 같음)
- 선택지 문구를 JEV 가 원문에서 추출 (서버의 규칙 추출 _candidate_labels 미사용).
- 결과 요약·상담원 인계 문구·목적 요약도 som_jev 해석기 방식 (로컬 모델 미사용).
- 받아쓰기 고침·막다른 안내 판단(answers_goal)·상담원 대기 확인 없음. 세션은 해석기에 이 함수가 없으면 호출하지 않음.
"""
from __future__ import annotations

import logging
import re

from gemini_interpreter import clean_label
from jev_interpreter import _strip_copula
from som_jev.jev_menu_interpreter import UnifiedJevInterpreter
from som_jev.navigator import Navigator as _SomNavigator
from som_jev.recovery_budget import MenuRecoveryBudgets  # noqa: F401  (세션에서 사용)

log = logging.getLogger("ars.som")


class SomNavigator(_SomNavigator):
    """som_jev 탐색기. 서버 세션이 넘기는 추가 인자(revised·user_branch)는 som_jev decide 에 없으므로 버리고 그대로 호출."""

    async def decide(self, st, current, history, arrived: bool = True, revised: bool = False, user_branch=None):
        return await super().decide(st, current, history, arrived)


def som_navigator_like(base, interp) -> SomNavigator:
    """서버 탐색기(base)와 같은 기준값(일치·차이·불확실 기준, 반복·키 횟수 한도)으로 som_jev 탐색기 생성."""
    return SomNavigator(interp, base.threshold, base.margin, base.max_revisit, base.max_presses, base.low)


def som_client_from_env():
    """som_jev 의 JevClient (TYPESAFE_API_KEY, JEV_MODEL, JEV_TIMEOUT_S). 키가 없거나 설정을 읽지 못하면 None."""
    try:
        return UnifiedJevInterpreter().client
    except RuntimeError:
        log.info("TYPESAFE_API_KEY 가 없어 som 방식은 고를 수 없다")
        return None
    except Exception as e:  # JEV_TIMEOUT_S 가 숫자가 아님 등
        log.warning("som 방식 JEV 연결 설정을 읽지 못했다: %s", type(e).__name__)
        return None


def new_som_interpreter(client) -> UnifiedJevInterpreter:
    """통화 하나가 쓸 som_jev 해석기 (som_jev 의 new_session 과 같이 연결만 공유하고 상태는 새로 생성)."""
    return UnifiedJevInterpreter(client)


def is_som(interp) -> bool:
    return isinstance(interp, UnifiedJevInterpreter)


_DISPLAY_TAIL = re.compile(r"(?:을|를)\s*원하시면$|(?:으시면|하시려면|시려면|려면|으면|면|은|는)$")


def display_label(label: str) -> str:
    """화면에 보일 선택지 이름. som 해석기는 JEV 가 원문 구절을 그대로 뽑아 '상담원 연결을 원하시면' 처럼 조건 꼬리가 남음.
    서버 규칙 추출(jev_interpreter._candidate_labels)과 같은 방식으로 서술격 조건 끝말·'을 원하시면'·끝의 키 표현을 제거.
    다 지워지면 원래 이름."""
    out = _DISPLAY_TAIL.sub("", _strip_copula(clean_label(label).strip())).strip(" ,.")
    return out or label


def display_fixes_for(prompt) -> list[tuple[str, str]]:
    """그 안내의 화면용 고침 목록 (원래 이름, 다듬은 이름). 세션은 이것을 Prompt.display_fixes 로 두어 자막·경로·재질문·상태 문구에만
    적용. som 해석기·탐색기와 JEV 요청은 원래 이름 그대로임."""
    out = []
    for o in prompt.options:
        new = display_label(o.label)
        if new != o.label:
            out.append((o.label, new))
    return out
