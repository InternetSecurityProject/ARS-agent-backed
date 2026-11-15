"""Approval safety regressions; every call/model is an in-process test double."""
import pytest

from gemini_interpreter import rule_keys
from jev_interpreter import extract_prompt, rule_critical
from navigator import NavState, Navigator
from test_hybrid_interpreter import make, run


MIXED_OR_AMBIGUOUS = [
    '결제 취소 조회 및 신청은 1번, 이전 메뉴는 2번을 눌러주세요.',
    '결제 취소 안내 및 신청은 1번, 이전 메뉴는 2번을 눌러주세요.',
    '송금 취소 내역을 처리하려면 1번, 이전 메뉴는 2번입니다.',
    '결제 취소 내역입니다. 위 내용으로 처리할까요? 예는 1번, 아니요는 2번입니다.',
    '송금취소 관련 문의는 1번, 송금 취소는 2번을 눌러주세요.',
    '결제 취소 조회 또는 신청은 1번, 이전 메뉴는 2번입니다.',
    '1번은 결제 취소 조회 및 신청, 2번은 이전 메뉴입니다.',
    '결제 취소 조회 관련 처리는 1번, 이전 메뉴는 2번입니다.',
    '결제 취소 문의입니다. 처리를 원하면 1번, 아니면 2번입니다.',
    '결제 취소 조회는 1번으로 처리됩니다.',
    '결제 취소 조회는 1번을 누르시면 바로 접수됩니다.',
    '결제 취소 조회는 1번입니다만 바로 신청됩니다.',
    '결제 취소 조회는 1번입니다. 위 내용으로 처리할까요? 예는 3번, 아니요는 4번입니다.',
    '결제 취소 안내는 1번입니다. 위 내용으로 신청을 할까요? 예는 3번, 아니요는 4번입니다.',
]


@pytest.mark.parametrize('text', MIXED_OR_AMBIGUOUS)
def test_mixed_execution_or_unclear_inquiry_keeps_the_approval_guard(text, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')
    assert rule_critical(rule_keys(text))
    assert extract_prompt(text).critical


@pytest.mark.parametrize('text', MIXED_OR_AMBIGUOUS[:6])
def test_real_navigator_requires_confirmation_even_when_model_says_not_critical(text, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')

    async def go():
        interp, _, _ = make(pick='1')  # FakeJev reports critical=0; the real rule/gate must protect it.
        interp.bench_reset('결제 취소 처리')
        prompt = await interp.classify(text)
        decision = await Navigator(interp).decide(NavState('결제 취소 처리'), prompt, [])
        assert prompt.critical
        assert decision.action == 'ask' and decision.reason == 'critical_action'
    run(go())


@pytest.mark.parametrize('text', [
    '결제 취소 조회는 1번입니다.',
    '결제 취소 조회는 1번을 눌러주세요.',
    '결제 취소 내역은 12번, 이용 금액은 2번입니다.',
    '송금취소 관련 문의는 일 번, 송금 한도 문의는 삼 번을 눌러주세요.',
    '자동이체 취소 안내는 1 번을 눌러 주십시오.',
    '결제 취소 조회는 별표를 눌러주세요.',
    '결제 취소 내역은 우물정자를 눌러주십시오.',
])
def test_only_clear_inquiry_key_assignments_are_exempt(text, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')
    assert not extract_prompt(text).critical


@pytest.mark.parametrize('text', [
    '결제 취소 조회는 별표를 누르시면 바로 접수됩니다.',
    '결제 취소 내역은 우물정자로 처리됩니다.',
])
def test_symbol_keys_do_not_exempt_execution_instructions(text, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')
    assert extract_prompt(text).critical


def test_processing_question_after_an_inquiry_is_not_auto_selected(monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')

    async def go():
        interp, _, _ = make(pick='3')
        prompt = await interp.classify(MIXED_OR_AMBIGUOUS[-2])
        decision = await Navigator(interp).decide(NavState('결제 취소 처리'), prompt, [])
        assert decision.action == 'ask' and decision.reason == 'critical_action'
    run(go())


@pytest.mark.parametrize('text', MIXED_OR_AMBIGUOUS[:4])
def test_real_session_does_not_send_a_digit_before_explicit_selection(text, monkeypatch):
    from jev_interpreter import JevInterpreter
    from test_flows import manual
    from test_hybrid_interpreter import FakeJev
    from session import ApiError

    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')

    async def go():
        s, call = await manual('결제 취소 처리', clarify_expire_s=5, clarify_drop_s=10)
        interp = JevInterpreter(FakeJev(pick='1'))
        s.d.interp = s.d.navigator.interp = interp
        try:
            await s.on_paragraph(text)
            assert call.keys == []
            assert s.pending and s.pending['reason'] == 'critical_action'
            request_id = s.pending['request_id']
            with pytest.raises(ApiError):
                await s.action({'type': 'answer', 'request_id': request_id, 'text': '네'})
            assert call.keys == []
            await s.action({'type': 'select', 'request_id': request_id, 'key': '1'})
            assert call.keys == ['1'] and s.path[-1]['actor'] == 'user'
        finally:
            await s.action({'type': 'cancel'})
    run(go())


def _public_ars_menus():
    import json
    from pathlib import Path
    return json.loads((Path(__file__).parent / 'testdata' / 'approval_guard_ars_menus.json').read_text())


@pytest.mark.parametrize('record', _public_ars_menus(), ids=lambda r: r['institution'])
def test_researched_public_service_menus_are_not_mistaken_for_transaction_approval(record, monkeypatch):
    # Hand-reviewed ordinary menu listings, not an ASR/model accuracy benchmark.
    # Original STT spelling is preserved. No user goals/input/readbacks are included.
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')
    prompt = extract_prompt(record['text'])
    assert prompt.options
    assert prompt.critical is record['requires_transaction_confirmation']


@pytest.mark.parametrize('text', MIXED_OR_AMBIGUOUS)
def test_alternate_jev_path_keeps_its_existing_approval_protection(text):
    from som_jev.jev_interpreter import extract_prompt as alternate_extract
    assert alternate_extract(text).critical


@pytest.mark.parametrize('text,pick', [
    ('결제 취소 조회는 1번, 바로 신청하려면 2번을 눌러주세요.', '2'),
    ('결제 취소 조회는 1번을 눌러주세요. 해당 번호를 누르면 바로 접수됩니다.', '1'),
    ('결제 취소 조회는 1번입니다. 위 내용대로 처리해 드릴까요? 예는 3번, 아니요는 4번입니다.', '3'),
    ('결제 취소 내역은 1번입니다. 이 내용대로 신청하시겠습니까? 예는 3번, 아니요는 4번입니다.', '3'),
    ('결제 취소 조회는 1번, 신청 및 조회는 2번입니다.', '2'),
    ('결제 취소 안내는 1번, 처리는 2번입니다.', '2'),
    ('결제 취소 내역은 1번, 접수 또는 안내는 2번입니다.', '2'),
])
def test_execution_in_another_option_or_sentence_still_requires_confirmation(text, pick, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')

    async def go():
        interp, _, _ = make(pick=pick)
        prompt = await interp.classify(text)
        decision = await Navigator(interp).decide(NavState('결제 취소 처리'), prompt, [])
        assert prompt.critical
        assert decision.action == 'ask' and decision.reason == 'critical_action'
    run(go())


@pytest.mark.parametrize('text', [
    '송금 취소 문의는 1번 송금 한도 문의는 2번입니다.',
    '송금취소 관련 문의는 1번송금 한도 조회는 2번입니다.',
    '결제 취소 조회는 별표 결제 금액 문의는 2번입니다.',
    '결제 취소 조회는 1번 결제 취소 내역은 2번입니다.',
])
def test_unpunctuated_but_separate_inquiry_choices_can_navigate(text, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')

    async def go():
        interp, _, _ = make(pick='2')
        prompt = await interp.classify(text)
        decision = await Navigator(interp).decide(NavState('한도 조회'), prompt, [])
        assert not prompt.critical
        assert decision.action == 'press' and decision.key == '2'
    run(go())


@pytest.mark.parametrize('text', [
    '송금 취소 문의는 1번 신청 및 조회는 2번입니다.',
    '송금 취소 문의는 1번 바로 접수 및 안내는 2번입니다.',
    '송금 취소 문의는 1번 처리 후 내역은 2번입니다.',
])
def test_unpunctuated_execution_cannot_masquerade_as_the_next_inquiry(text, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')
    assert extract_prompt(text).critical


@pytest.mark.parametrize('text', [
    '결제 취소 조회는 1번, 신청 방법 안내는 2번입니다.',
    '결제 취소 문의는 1번, 접수 현황 조회는 2번입니다.',
])
def test_method_guides_and_status_queries_are_not_execution_predicates(text, monkeypatch):
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')
    assert not extract_prompt(text).critical


@pytest.mark.parametrize('text', [
    '서식 팩스 전송 서비스 신청입니다. 근로자 자격관리 서식은 1번, 보험가입지원 서식은 2번, '
    '재증명 신청은 7번을 눌러주십시오. 다시 듣기를 원하시면 별표를 눌러주십시오.',
    '이체한도 조회는 1번, 이체한도 줄이기 신청은 2번, 입금계좌 지정내역 조회는 3번입니다.',
])
def test_cancellation_fix_does_not_reclassify_unrelated_application_menus(text, monkeypatch):
    # Actual public archive wording. Preserve the existing rule result outside
    # cancellation context; the model's independent critical flag still wins.
    monkeypatch.setenv('JEV_RULE_CRITICAL', '1')
    assert not extract_prompt(text).critical
