"""
메뉴 탐색 판단 (7장 처리 흐름의 '선택지 안내' 분류)

탐색 규칙 (v0.3)
  1. 현재 머물러 있는 ARS 계층(가장 최근에 들은 메뉴)에서 판단한다. 처음에는 첫 메뉴, 다시 말하기 후에는 그 시점의 메뉴.
  2. 신뢰도(AI score: 목적을 선택지 카테고리에 맞춘 값) 구간에 따라 판단한다.
       0.50 이상           일치. 1·2순위 차이 0.10 이상이면 키 전송, 미만이면 재질문(ambiguous)
       0.30 이상 0.50 미만  불확실. 재질문(low_confidence), 계층당 1회
       0.30 미만           불일치 (low_confidence 답변 뒤에도 0.50 미만이면 불일치로 본다)
  3. 불일치이면 바로 이전 계층의 다른 선택지를 확인한다.
     이전 계층에 0.50 이상인 선택지가 있으면 '이전 메뉴' 키(없으면 '처음으로' 키 + 지나온 키)로 돌아가 다시 판단한다.
  4. 이전 계층에도 없으면 탐색을 종료한다 (end_reason=not_found). 이전 계층 복귀는 탐색당 1회만 허용한다.

판단 결과(Decision)만 반환하고, 키 전송·이벤트 전송은 session 모듈이 수행한다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from interpreter import Interpreter, Option, Prompt

_GENERIC = {"서비스", "안내", "업무", "문의", "연결", "메뉴", "기타", "다른", "신청", "조회", "발급", "변경", "확인"}


def _chunks(label: str) -> list[str]:
    """선택지 이름을 낱말 조각으로: '신용 체크카드' → ['신용', '체크', '카드'], '현금카드' → ['현금', '카드']."""
    import re

    out: list[str] = []
    for w in re.split(r"[\s·,/및과와]+", label):
        w = re.sub(r"[^가-힣A-Za-z0-9]", "", w)
        if not w:
            continue
        # '체크카드'·'현금카드'·'집주소' 처럼 붙은 복합어는 뒤의 공통 명사를 떼어 둔다 (2글자 명사 기준)
        m = re.match(r"^(.{2,})(카드|주소|계좌|통장|보험|요금|번호|예금|대출|수표)$", w)
        out += [m.group(1), m.group(2)] if m else [w]
    return out


def sibling_choices(goal: str, best: Option, others: list[Option]) -> list[Option]:
    """best 가 다른 선택지와 같은 대상의 '종류'를 고르는 선택지인데(신용·체크카드 / 현금카드, 집 주소 / 회사 주소,
    국민은행 송금 / 타행 송금), 목적에 그 종류를 가르는 말이 없으면 AI 가 대신 고르지 않고 사용자에게 묻는다.
    같은 대상 = 선택지 이름의 마지막 명사(머리말)가 같다. '결제일 변경 / 결제 대금 조회' 처럼 머리말이 다르면 다른 업무다."""
    g = goal.replace(" ", "")

    def same(a: str, b: str) -> bool:  # '주소지'·'주소' 처럼 한쪽이 다른 쪽으로 시작하면 같은 명사로 본다
        return len(a) >= 2 and len(b) >= 2 and (a.startswith(b) or b.startswith(a))

    bc = _chunks(best.label)
    if len(bc) < 2 or bc[-1] in _GENERIC:
        return []
    twins: list[Option] = []
    for o in others:
        oc = _chunks(o.label)
        if len(oc) < 2 or not same(bc[-1], oc[-1]):
            continue
        if any(c in g for c in bc[:-1] if len(c) >= 2):
            continue  # 목적이 best 쪽을 가리킨다 ("신용카드 …" → 신용 체크카드)
        if any(c in g for c in oc[:-1] if len(c) >= 2):
            continue  # 목적이 다른 쪽을 가리킨다
        twins.append(o)
    return twins


@dataclass
class Layer:
    """키를 눌러 지나온 메뉴 한 단계."""
    prompt: Prompt
    key: str
    actor: str  # ai / user


@dataclass
class Decision:
    action: str                 # press / ask / backtrack / end
    key: str | None = None      # press: 누를 키
    keys: str | None = None     # backtrack: 돌아가기 위해 누를 키 열
    reason: str = ""            # ask: clarification reason / end: end_reason
    note: str = ""              # 사람이 읽는 판단 근거
    options: list[Option] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class NavState:
    goal: str
    backtracked: bool = False           # 이전 계층 복귀를 이미 사용했는지
    exclude: set[str] = field(default_factory=set)  # 복귀한 계층에서 제외할 키 (이미 시도한 키)
    exclude_at: str | None = None       # exclude 를 적용할 계층 (메뉴 signature). 다른 메뉴에는 적용하지 않는다
    presses: int = 0                    # AI 가 이번 탐색에서 누른 업무 선택 키 수
    visits: Counter = field(default_factory=Counter)
    revisits: int = 0                   # 키 전송 결과로 이미 지나온 메뉴에 다시 도착한 횟수
    low_asked: set[str] = field(default_factory=set)  # low_confidence 재질문을 이미 한 계층
    sibling_asked: set[str] = field(default_factory=set)  # 같은 대상의 종류를 고르는 재질문을 이미 한 계층

    def reset_loop(self) -> None:
        """반복(loop) 재질문에 사용자가 응답하면 반복 판정을 처음부터 다시 센다."""
        self.visits.clear()
        self.revisits = 0
        self.presses = 0


class Navigator:
    def __init__(self, interp: Interpreter, threshold: float = 0.5, margin: float = 0.1,
                 max_revisit: int = 2, max_presses: int = 5, low_threshold: float = 0.3) -> None:
        self.interp = interp
        self.threshold, self.margin, self.low = threshold, margin, low_threshold
        self.max_revisit, self.max_presses = max_revisit, max_presses

    async def _rank(self, goal: str, opts: list[Option], path: list[str]) -> tuple[list[tuple[Option, float]], dict[str, float]]:
        scores = await self.interp.score(goal, opts, path) if opts else {}
        ranked = sorted(((o, scores.get(o.key, 0.0)) for o in opts), key=lambda x: x[1], reverse=True)
        return ranked, scores

    async def decide(self, st: NavState, current: Prompt, history: list[Layer], arrived: bool = True) -> Decision:
        """arrived=True: 키 전송 결과로 새로 도착한 메뉴. 재질문 답변 후 같은 메뉴를 다시 판단할 때는 False."""
        sig = current.signature()
        excluded = st.exclude if st.exclude_at == sig else set()
        opts = [o for o in current.options if o.key not in excluded]

        # 반복 감지: 키를 눌러 이미 지나온 메뉴에 다시 도착한 경우만 센다 (7장 5항).
        # 명세 6.3: '이미 지나온 메뉴가 다시 나오는 상황이 2회 발생' 또는 키 전송 5회
        # 확인 질문("맞으면 1번/틀리면 2번")은 여러 단계에서 같은 모양으로 되풀이되므로 반복으로 세지 않는다
        if arrived and current.kind == "menu":
            if st.visits[sig]:
                st.revisits += 1
            st.visits[sig] += 1
        # MAX_REVISIT=0 / MAX_PRESSES=0 이면 그 판정을 끈다
        if (self.max_revisit and st.revisits >= self.max_revisit) or (self.max_presses and st.presses >= self.max_presses):
            note = (f"지나온 메뉴에 다시 도착 {st.revisits}회" if st.revisits >= self.max_revisit
                    else f"키 {st.presses}회 입력 후 미도달")
            return Decision("ask", reason="loop", note=note, options=current.options)

        if current.critical:
            return Decision("ask", reason="critical_action", note="중요 선택", options=current.options)

        path = [layer.prompt.category_of(layer.key) for layer in history]
        ranked, scores = await self._rank(st.goal, opts, path)
        if ranked and ranked[0][1] >= self.threshold:
            best, s1 = ranked[0]
            s2 = ranked[1][1] if len(ranked) > 1 else 0.0
            if s1 - s2 >= self.margin:
                twins = sibling_choices(st.goal, best, [o for o, _ in ranked[1:]]) if sig not in st.sibling_asked else []
                if twins:
                    # 같은 대상의 종류를 고르는 선택지인데 목적에 그 구분이 없다. 모델이 높게 줘도 사용자가 고른다 (계층당 1회)
                    st.sibling_asked.add(sig)
                    return Decision("ask", reason="ambiguous", note=f"'{best.label}' 과 '{twins[0].label}' 중 목적만으로 구분 불가",
                                    options=[best] + twins, scores=scores)
                return Decision("press", key=best.key, note=f"'{best.label}' 일치율 {s1:.2f}", scores=scores)
            cands = [o for o, s in ranked if s >= self.threshold and s1 - s < self.margin]
            return Decision("ask", reason="ambiguous", note=f"1·2순위 차이 {s1 - s2:.2f}", options=cands, scores=scores)

        # 불확실 → 사용자에게 재질문 (계층당 1회)
        best_s = ranked[0][1] if ranked else 0.0
        if best_s >= self.low and sig not in st.low_asked:
            st.low_asked.add(sig)
            cands = [o for o, s in ranked if s >= self.low][:3]
            return Decision("ask", reason="low_confidence", note=f"최고 신뢰도 {best_s:.2f} (기준 {self.threshold:.2f} 미만)",
                            options=cands, scores=scores)

        # 불일치 → 이전 계층 확인
        if st.backtracked:
            return Decision("end", reason="not_found", note=f"이전 계층 확인 후에도 일치 선택지 없음 (최고 {best_s:.2f})", scores=scores)
        if not history:
            return Decision("end", reason="not_found", note=f"현재 계층에 일치 선택지 없음, 이전 계층 정보 없음 (최고 {best_s:.2f})", scores=scores)

        prev = history[-1]
        alts = [o for o in prev.prompt.options if o.key != prev.key]
        alt_ranked, _ = await self._rank(st.goal, alts, path[:-1])
        if not alt_ranked or alt_ranked[0][1] < self.threshold:
            return Decision("end", reason="not_found", note="현재·이전 계층 모두 일치 선택지 없음", scores=scores)

        if current.back_key:
            keys = current.back_key
        elif current.home_key and all(layer.key for layer in history[:-1]):
            keys = current.home_key + "".join(layer.key for layer in history[:-1])
        else:
            return Decision("end", reason="not_found",
                            note="이전 계층에 후보가 있으나 돌아가는 키(이전 메뉴·처음으로)가 안내되지 않음", scores=scores)
        return Decision("backtrack", keys=keys,
                        note=f"이전 계층의 '{alt_ranked[0][0].label}' 확인을 위해 복귀", scores=scores)
