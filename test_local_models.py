"""Local-copy model adapters; all remote transports are replaced, no real calls."""
import asyncio
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from audio import hum_ulaw
from interpreter import Option, Prompt
from stt import OpenAIRecognizer


def run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize('model,expected', [
    ('gpt-transcribe', {'extra_body': {'languages': ['ko', 'en']}}),
    ('gpt-4o-transcribe', {'language': 'ko'}),
])
def test_openai_model_specific_language_request(monkeypatch, model, expected):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    r = OpenAIRecognizer(model)
    async def create(**kwargs):
        assert {k: kwargs[k] for k in ('language', 'extra_body') if k in kwargs} == expected
        assert kwargs['model'] == model
        assert kwargs['file'].read(4) == b'RIFF'
        return SimpleNamespace(text='조회는 2번입니다.')
    r.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    assert run(r.recognize(hum_ulaw(500))) == '조회는 2번입니다.'


def test_openai_context_hint_is_bounded_and_not_retained(monkeypatch):
    from stt import ARS_PROMPT
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    r=OpenAIRecognizer('gpt-transcribe');prompts=[]
    async def create(**kwargs):
        prompts.append(kwargs['prompt'])
        return SimpleNamespace(text='조회는 2번입니다.')
    r.client=SimpleNamespace(audio=SimpleNamespace(transcriptions=SimpleNamespace(create=create)))
    async def go():
        await r.recognize(hum_ulaw(500),'가'*500)
        await r.recognize(hum_ulaw(500))
    run(go())
    assert prompts[0]==ARS_PROMPT+'\n앞선 음성의 전사문(문맥 참고용): '+'가'*240
    assert prompts[1]==ARS_PROMPT


def test_menu_labels_keys_navigation_and_symbol_business():
    from jev_interpreter import extract_prompt
    p = extract_prompt('국민은행입니다. 잔액 조회는 3번, 카드 업무는 5번, 분실 신고는 별표, '
                       '다른 서비스는 샵 버튼, 상담원 연결은 0번, 다시 듣기는 9번을 눌러주세요.')
    assert [(o.key, o.label) for o in p.options] == [
        ('3', '잔액 조회'), ('5', '카드 업무'), ('*', '분실 신고'), ('#', '다른 서비스'), ('0', '상담원 연결')]
    assert p.repeat_key == '9' and p.agent_key == '0'


def test_duplicate_keys_are_not_selectable():
    from jev_interpreter import extract_prompt
    p = extract_prompt('잔액 조회는 3번, 카드 업무는 3번, 상담원 연결은 0번입니다.')
    assert [o.key for o in p.options] == ['0']


def test_sensitive_and_readback_do_not_become_business_star_options():
    from jev_interpreter import extract_prompt
    p = extract_prompt('주민등록번호 앞 여섯 자리를 입력하고 샵 버튼을 눌러 주세요.')
    assert p.kind == 'sensitive' and p.terminator == '#' and not p.options
    p = extract_prompt('입력하신 주민등록번호는 ******입니다. 맞으면 1번, 틀리면 2번을 눌러 주세요.')
    assert p.kind == 'confirm' and {o.key for o in p.options} == {'1', '2'}


def test_unpunctuated_spoken_menu_and_long_labels():
    from jev_interpreter import extract_prompt
    p = extract_prompt('입출금 거래내역은 일 번 입금 거래내역은 이 번 출금 거래내역은 삼 번을 눌러 주세요')
    assert [(o.key, o.label) for o in p.options] == [('1', '입출금 거래내역'), ('2', '입금 거래내역'), ('3', '출금 거래내역')]
    p = extract_prompt('인터넷·스마트 상담원 연결 및 텔레뱅킹 장기 미사용자 해제·등록 이체내역 통제 신청·해제는 2번입니다.')
    assert p.options[0].label.startswith('인터넷·스마트')


def client(handler):
    from jev_interpreter import JevClient
    return JevClient('test-SECRET', http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def reply(questions, *, kind='menu', key='dtmf_5', available=.97, critical=.01):
    answers = {}
    for name, q in questions.items():
        if q['type'] == 'choice':
            chosen = kind if name == 'kind' else 'proceed' if name == 'route_status' else key
            answers[name] = {'type': 'choice', 'choice': chosen,
                'probabilities': {k: (.98 if k == chosen else .001) for k in q['criteria']}, 'confidence': .98}
        else:
            answers[name] = {'type': 'noul', 'noul': critical if name == 'critical' else available}
    return {'model': 'jev-1.14.0', 'answers': answers, 'usage': {'input_tokens': 100}}


def test_jev_decision_reaches_navigator_and_uses_current_context():
    from jev_interpreter import JevInterpreter
    from navigator import Navigator, NavState
    def handler(req):
        body = json.loads(req.content)
        assert body['state']['user_goal'] == '카드 결제일 변경'
        assert body['model'] == 'jev-1.14.0'
        return httpx.Response(200, json=reply(body['questions']))
    async def go():
        i = JevInterpreter(client(handler))
        await i.summarize_goal('카드 결제일 변경')
        p = await i.classify('잔액 조회는 3번, 카드 업무는 5번을 눌러 주세요.')
        d = await Navigator(i).decide(NavState('카드 결제일 변경'), p, [])
        assert d.action == 'press' and d.key == '5'
        await i.summarize_goal('새 목적')
        assert i.last_menu is None and i.last_path == [] and i.last_answer is None
    run(go())


@pytest.mark.parametrize('available', [.4, .1])
def test_uncertain_or_absent_service_never_presses(available):
    from jev_interpreter import JevInterpreter
    from navigator import Navigator, NavState
    def handler(req):
        b = json.loads(req.content)
        return httpx.Response(200, json=reply(b['questions'], available=available))
    async def go():
        i = JevInterpreter(client(handler))
        await i.summarize_goal('그거 해줘')
        p = await i.classify('잔액 조회는 3번, 카드 업무는 5번입니다.')
        d = await Navigator(i).decide(NavState('그거 해줘'), p, [])
        assert d.action != 'press'
    run(go())


def test_critical_confirmation_requires_user_even_if_model_says_no():
    from jev_interpreter import JevInterpreter
    from navigator import Navigator, NavState
    def handler(req):
        b = json.loads(req.content)
        return httpx.Response(200, json=reply(b['questions'], kind='confirm', key='dtmf_1'))
    async def go():
        i = JevInterpreter(client(handler))
        p = await i.classify('카드 해지를 진행하려면 1번, 취소하려면 2번을 눌러 주세요.')
        assert p.critical
        assert await i.answer_confirm('카드 해지', p) is None
        assert (await Navigator(i).decide(NavState('해지'), p, [])).reason == 'critical_action'
    run(go())


@pytest.mark.parametrize('bad', ['unknown_key', 'nan', 'http_error', 'timeout'])
def test_invalid_or_failed_jev_does_not_press_or_log_secret(bad, caplog):
    from jev_interpreter import JevInterpreter
    def handler(req):
        if bad == 'http_error': return httpx.Response(401, text='test-SECRET')
        if bad == 'timeout': raise httpx.ReadTimeout('test-SECRET')
        b = json.loads(req.content)
        d = reply(b['questions'], key='dtmf_9' if bad == 'unknown_key' else 'dtmf_5')
        if bad == 'nan' and 'has_matching_service' in d['answers']:
            d['answers']['has_matching_service']['noul'] = float('nan')
        return httpx.Response(200, json=d)
    async def go():
        i = JevInterpreter(client(handler))
        await i.summarize_goal('카드')
        p = await i.classify('잔액 조회는 3번, 카드 업무는 5번입니다.')
        scores = await i.score('카드', p.options, [])
        assert scores and all(math.isfinite(x) and x < .5 for x in scores.values())
    run(go())
    assert 'test-SECRET' not in caplog.text


def test_missing_jev_key_fails_before_any_call(monkeypatch):
    from jev_interpreter import JevInterpreter
    monkeypatch.delenv('TYPESAFE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='TYPESAFE_API_KEY'):
        JevInterpreter()


def test_answer_and_result_protocol_without_generative_api():
    from jev_interpreter import JevInterpreter
    def fail(_): raise AssertionError('No API needed for local summaries/explicit answers')
    async def go():
        i = JevInterpreter(client(fail))
        p = Prompt('confirm', '맞으면 1번 틀리면 2번', [Option('1', '확인'), Option('2', '다시 입력')])
        i.last_menu = p
        await i.merge_answer('잔액 조회', p.text, '아니요')
        assert await i.answer_confirm('잔액 조회', p) == '2'
        assert await i.answer_confirm('잔액 조회', p) is None
        assert (await i.summarize_result('잔액 조회', ['조회 결과입니다.']))['summary'] == '조회 결과입니다.'
    run(go())
