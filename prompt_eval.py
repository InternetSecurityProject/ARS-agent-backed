"""
프롬프트 버전 비교 평가 (실제 Gemini 호출, 무료 키 사용량 주의)

  python prompt_eval.py                         현재 gemini_interpreter.py
  python prompt_eval.py --module 이전버전.py      다른 버전 파일 (예: git show <커밋>:gemini_interpreter.py > old.py)

서버(navigator) 판단 규칙과 같은 방식으로 한 단계씩 판단한다.
  0.50 이상·1·2위 차이 0.10 이상 → 번호 누름 / 0.30~0.50 또는 차이 부족 → 재질문 / 0.30 미만 → 이전 단계 확인·종료(서버)
사례
  nhis_*  건강보험증 발급 평가 10건 (feat/ai-evaluation 의 nhis_card_v0_1, 소민님)
  real_*  실제 통화 녹취 STT 문장 17개 (국민건강보험공단·KB국민은행·하나은행)
  nomatch_*  맞는 선택지가 없는 경우 4개 (프롬프트 v0.4)
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import config  # noqa: F401,E402  (.env 로드)

PRESS, ASK, NONE, PAUSE, END = "번호 누름", "재질문", "이전 단계·종료", "직접 입력", "결과·종료"

NHIS_MAIN = ("지금은 셀프 발급 및 조회 업무만 이용 가능합니다. 상담사 연결이 필요하시면 평일 오전 9시부터 오후 6시 사이에 다시 걸어주시기 바랍니다. "
             "화면을 보면서 이용할 수 있는 디지털 ars는 9번 증명서 발급은 1번, 건강보험증 발급은 2번 고지서 발급은 3번 건강검진 추가 신청은 4번 "
             "의료급여 진료 확인번호 요청은 5번 팩스 발송 결과 조회는 6번 각종 서신 팩스 발송 서비스는 7번 홈페이지 및 모바일 앱 설치의 문자 안내는 8번 "
             "내 보험료 계산하기는 샵 버튼, 다시 듣기는 별표를 눌러주십시오.")
KB_MAIN = ("분실 신고 금융 사기는 별표 KB 청년 미래 적금 안내는 8번, 상담 직원 연결은 0번 빠른 말 또는 쉬운 말 서비스는 샵 버튼 "
           "잔액 조회는 3번, 거래 내역 조회는 4번 카드 및 기타 서비스는 5번 폰뱅킹 전용 메뉴는 7번 폴 포딩 랭귀지 서비 프렉스 스 다시 듣기는 9번을 눌러주십시오.")
NHIS_QUAL = ("건강보험 자격 관련 서비스입니다. 건강보험증 발급은 1번, 자격 취득 상실 관련 문의는 2번, 서로의 출입국 신고 관련 문의는 3번, "
             "상담사 연결은 0번, 다시 듣기는 별표 전 단계는 우물 정자를 눌러주십시오.")
KB_ARS = "KB 스타뱅킹을 통해 간편 조회를 하기 위한 카카오 알림톡 전송은 1번 ars로 계속 진행은 2번 다시 듣기는 별표 이전 단계는 샵 버튼을 눌러주십시오."

# (id, 목적, 사용자 답변, 앞서 들은 안내, 안내, 기대 동작, 기대 키)
NHIS = [
    ("nhis_card_001", "건강보험증을 발급받고 싶다", None, None, "화면을 보면서 이용할 수 있는 디지털 ARS는 9번, 증명서 발급은 1번, 건강보험증 발급은 2번, 고지서 발급은 3번, 건강검진 추가 신청은 4번, 의료급여 진료 확인번호 요청은 5번, 팩스 발송 결과 조회는 6번, 각종 서식 팩스 발송 서비스는 7번, 홈페이지 및 모바일 앱 설치 문자 안내는 8번, 내 보험료 계산하기는 샵 버튼, 다시 듣기는 별표를 눌러주십시오.", PRESS, "2"),
    ("nhis_card_002", "건강보험증을 새로 받고 싶어", None, None, "디지털 알스는 구번, 증명서 발급 일번, 건강 보헙증 발급 이번, 고지서 발급 삼번, 건강검진 추가 신청 사번을 눌러주십시오.", PRESS, "2"),
    ("nhis_card_003", "건강보험증을 발급받고 싶다", None, None, "증명서 발급은 1번, 건강보험증 발급은, 고지서 발급은 3번을 눌러주십시오.", ASK, None),
    ("nhis_card_004", "건강보험증을 발급받고 싶다", None, None, "건강보험증 발급입니다. 타인의 주민등록번호를 이용할 경우 법적인 처벌을 받을 수 있습니다. 본인의 주민등록번호 13자리를 눌러주십시오.", PAUSE, None),
    ("nhis_card_005", "건강보험증을 발급받고 싶다", None, None, "입력하신 주민등록번호는 마스킹된 번호입니다. 맞으면 1번, 틀리면 2번을 눌러주십시오.", ASK, None),
    ("nhis_card_006", "건강보험증을 발급받고 싶다", "맞아", None, "입력하신 주민등록번호가 맞으면 1번, 틀리면 2번을 눌러주십시오.", PRESS, "1"),
    ("nhis_card_007", "건강보험증을 발급받고 싶다", None, None, "지역가입자는 주민등록상 주소지로 발급되고 직장가입자는 회사 또는 집 주소지를 선택하실 수 있습니다. 발급을 원하시면 1번, 원하지 않으시면 2번을 눌러주십시오.", PRESS, "1"),
    ("nhis_card_008", "건강보험증을 주민등록상 주소로 발급받고 싶다", "나는 지역가입자야", None, "주민등록상 주소지 발급은 1번, 회사 주소지 발급은 2번을 눌러주십시오.", PRESS, "1"),
    ("nhis_card_009", "건강보험증을 회사로 받고 싶다", "회사 주소로 보내줘", None, "주민등록상 주소지 발급은 1번, 회사 주소지 발급은 2번을 눌러주십시오.", PRESS, "2"),
    ("nhis_card_010", "건강보험증을 주민등록상 주소로 발급받고 싶다", None, None, "고객님의 주민등록상 주소지로 발송됩니다. 다시 듣기는 별표, 전 단계는 우물정자를 눌러주십시오.", END, None),
]
REAL = [
    ("real_nhis_main", "건강보험증 발급받고 싶어", None, None, NHIS_MAIN, PRESS, "2"),
    ("real_nhis_rrn", "건강보험증 발급", None, None, "건강보험증 발급입니다. 타인의 주민등록번호를 이용할 경우 법적인 처벌을 받을 수 있습니다. 본인의 주민등록번호 13자리를 눌러주십시오.", PAUSE, None),
    ("real_nhis_readback", "건강보험증 발급", None, None, "입력하신 주민등록번호는 *************입니다. 맞으면 1번, 틀리면 2번을 눌러주십시오.", ASK, None),
    ("real_nhis_agree", "건강보험증 발급받고 싶어", None, None, "지역 가입자는 주민등록상 주소지로 발급되고, 직장 가입자는 회사 또는 집 주소지를 선택하실 수 있습니다. 발급을 원하시면 1번 원하지 않으시면 2번을 눌러 주십시오.", PRESS, "1"),
    ("real_nhis_address", "건강보험증을 회사로 받고 싶어", None, None, "주민등록상 주소지 발급은 1번, 회사 주소지 발급은 2번을 눌러주십시오.", PRESS, "2"),
    ("real_nhis_sent", "건강보험증 발급", None, None, "고객님의 주민등록상 주소지로 발송됩니다. 다시 듣기는 별표 전 단계는 우물 정자를 눌러주십시오.", END, None),
    ("real_nhis_qual", "건강보험증 발급받고 싶어", None, NHIS_MAIN, NHIS_QUAL, PRESS, "1"),
    ("real_nhis_msg", "보험료 계산", None, None, "발송 완료되었습니다. 국민건강보험공단 알림톡 또는 문자 메시지를 확인해 주시기 바랍니다. 다시 듣기는 별표 전 단계는 우물 정자 이용 종료를 원하시면 0번을 눌러주십시오.", END, None),
    ("real_nhis_hours", "고지서 발급", None, None, "고객님의 고지서 발급을 위해서는 상담사 연결이 필요합니다. 저희 고객센터 업무 시간은 평일 오전 9시부터 6시까지이며, 현재는 팩스 및 일부 조회 발급 서비스만 가능합니다. 건강보험 고객센터를 이용해 주셔서 감사합니다.", END, None),
    ("real_nhis_timeout", "진료 확인번호", None, None, "입력 시간이 초과되었습니다. 요양 기관 코드 8자리를 입력하여 주십시오.", PAUSE, None),
    ("real_nhis_fax_typo", "팩스 발송 결과 조회", None, None, "직장 가입자 자격 취득 신고서는 1번 직장 가입자 자격 상실 신고서는 2번 피부양자 취득 및 상실신고서는 3번 텍스 발송 결과 조회는 0번 다시 듣기는 별표, 전단계는 우물 강자를 눌러주십시오.", PRESS, "0"),
    ("real_kb_main", "통장 잔액 알고 싶어", None, None, KB_MAIN, PRESS, "3"),
    ("real_kb_ars", "전화로 통장 잔액 듣고 싶어", None, KB_MAIN, KB_ARS, PRESS, "2"),
    ("real_kb_rrn_typo", "카드 분실 신고", None, None, "고객님의 주민등록번호 앞 여섯자리와 체업 버튼 또는 사업자 등록번호 10자리와 취업 버튼을 눌러주십시오. 다른 서비스는 체업 버튼을 눌러주십시오.", PAUSE, None),
    ("real_kb_pw", "통장 잔액", None, None, "계좌 비밀번호 4자리", PAUSE, None),
    ("real_kb_balance", "통장 잔액 알고 싶어", None, None, "현재 잔액은 2만 7856원, 출금 가능액은 2만 7856원 다시 듣기는 영 번 국민은행으로 송금은 일 번 다른 은행으로 송금은 이번 예금 펀드 대출 잔액 조회는 3번, 거래 내역 조회는 4번 다른 서비스는 샵버등.", END, None),
    ("real_hana_rrn", "송금", None, None, "본 서비스는 폰뱅킹 가입자만 이용 가능합니다. 손님의 주민등록번호 13자리를 눌러주십시오.", PAUSE, None),
]
NOMATCH = [
    # 첫 메뉴에 맞는 업무가 없고 상담원 연결이 있으면 → 상담원 연결 (돌아갈 이전 단계가 없음)
    ("nomatch_kb_first_agent", "자동차 보험 가입하고 싶어", None, None, KB_MAIN, PRESS, "0"),
    # 하위 메뉴에 맞는 업무가 없고 전 단계 키가 있으면 → 상담원보다 이전 단계 확인이 먼저 (서버가 이전 계층 확인)
    ("nomatch_nhis_sub_back", "건강검진 결과 확인하고 싶어", None, NHIS_MAIN, NHIS_QUAL, NONE, None),
    ("nomatch_kb_sub_back", "외화 환전하고 싶어", None, KB_MAIN, KB_ARS, NONE, None),
    # 첫 메뉴에 맞는 업무도 상담원 연결도 없으면 → 누르지 않음 (서버 not_found)
    ("nomatch_nhis_first_none", "자동차 보험 가입하고 싶어", None, None, NHIS_MAIN, NONE, None),
]


def load(module: str | None):
    if not module:
        import gemini_interpreter as m
        return m
    spec = importlib.util.spec_from_file_location("interp_under_test", module)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


async def decide(ai, goal: str, answer: str | None, before: str | None, text: str):
    for attr, v in (("last_answer", None), ("last_menu", None), ("last_path", []), ("goal", goal), ("_scores", {}),
                    ("_slow_until", 0.0)):
        setattr(ai, attr, v)
    for attr in ("_heard", "_menus", "_cache"):
        if hasattr(ai, attr):
            setattr(ai, attr, [] if attr == "_heard" else {})
    if answer:
        goal = await ai.merge_answer(goal, "확인이 필요합니다.", answer)
        ai.goal = goal
    if before:
        await ai.classify(before)
    t0 = time.perf_counter()
    p = await ai.classify(text)
    key = None
    if p.kind == "sensitive":
        act = PAUSE
    elif p.kind in ("result", "message_sent", "out_of_hours", "rejected"):
        act = END
    elif p.kind == "confirm" and not p.critical:
        key = await ai.answer_confirm(goal, p)
        act = PRESS if key else ASK
    elif p.kind in ("menu", "confirm"):
        s = await ai.score(goal, p.options, [])
        r = sorted(s.items(), key=lambda x: -x[1])
        s1 = r[0][1] if r else 0.0
        s2 = r[1][1] if len(r) > 1 else 0.0
        if s1 >= 0.5 and s1 - s2 >= 0.1:
            act, key = PRESS, r[0][0]
        elif s1 >= 0.3:
            act = ASK
        else:
            act = NONE
    else:
        act = ASK
    return act, key, p.kind, (time.perf_counter() - t0) * 1000


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", help="비교할 gemini_interpreter 파일 (없으면 현재 버전)")
    ap.add_argument("--only", choices=["nhis", "real", "nomatch"])
    ap.add_argument("--sleep", type=float, default=5.0, help="사례 사이 대기(초). 무료 키 분당 호출 제한")
    ap.add_argument("--timeout", type=float, default=20.0,
                    help="판단 대기 시간(초). 프롬프트 정확도를 볼 때는 길게(기본 20), 실제 통화 조건은 4")
    a = ap.parse_args()
    m = load(a.module)
    ai = m.GeminiInterpreter()
    if hasattr(ai, "decide_timeout_s"):
        ai.decide_timeout_s = a.timeout
    timeouts = 0
    cases = {"nhis": NHIS, "real": REAL, "nomatch": NOMATCH}
    rows, lat = [], []
    for group, items in cases.items():
        if a.only and a.only != group:
            continue
        for cid, goal, answer, before, text, exp_act, exp_key in items:
            slow_before = getattr(ai, "_slow_until", 0.0)
            act, key, kind, ms = await decide(ai, goal, answer, before, text)
            if getattr(ai, "_slow_until", 0.0) != slow_before and getattr(ai, "_slow_until", 0.0) > 0:
                timeouts += 1
            ok = act == exp_act and key == exp_key
            danger = key is not None and key != exp_key
            rows.append((cid, ok, danger, act, key, exp_act, exp_key, kind, ms))
            lat.append(ms)
            print(f"{'PASS' if ok else 'FAIL'}{' 위험' if danger else ''} {cid}: {act} {key or ''} (기대 {exp_act} {exp_key or ''}) "
                  f"kind={kind} {ms:.0f}ms", flush=True)
            await asyncio.sleep(a.sleep)
    lat.sort()
    n = len(rows)
    print(f"\n프롬프트 {getattr(m, 'PROMPT_VERSION', 'v0.3')}: 정답 {sum(r[1] for r in rows)}/{n}, 위험 자동입력 {sum(r[2] for r in rows)}건, "
          f"p50 {lat[n // 2]:.0f}ms, p95 {lat[min(n - 1, int(n * 0.95))]:.0f}ms, Gemini 시간 초과 {timeouts}건 (대기 {a.timeout:g}초)")
    for g in cases:
        sub = [r for r in rows if r[0].startswith(g)]
        if sub:
            print(f"  {g}: {sum(r[1] for r in sub)}/{len(sub)}")


if __name__ == "__main__":
    asyncio.run(main())
