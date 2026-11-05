"""Visible ARS captions preserve STT wording, not navigation placeholders."""
from interpreter import Option, Prompt
from session import caption_text
from test_jev_phrase import item, run
from test_system_recovery import RecoveryAI, make_session


def test_caption_does_not_replace_a_spoken_service_with_an_unknown_menu_label():
    text='카드 문의는 3번, 자기앞수표 조회는 1번, 다른 서비스는 샵 버튼을 눌러 주십시오.'
    p=Prompt('menu',text,[Option('1','1번 · 메뉴 확인 필요'),Option('3','카드 문의')])
    assert caption_text(p,text)==text


def test_caption_keeps_source_order_wording_and_navigation_not_in_ai_options():
    text='먼저 안내를 끝까지 들어 주세요. 배송 조회는 별표, 반품은 샵, 다시 듣기는 9번입니다.'
    p=Prompt('menu',text,[Option('#','반품'),Option('*','배송 조회')],repeat_key='9')
    assert caption_text(p,text)==text


def test_session_emits_raw_caption_even_when_menu_interpretation_needs_repair():
    async def go():
        text='예약 문의는 3번, 주차 요금은 1번, 다시 듣기는 9번.'
        s=make_session(RecoveryAI({'3':item('예약 문의는',confidence=.4),'1':item('주차 요금은'),
                                  '9':item('다시 듣기는','repeat')},route='unknown',key='3'))
        try:
            await s.on_paragraph(text)
            captions=[e['data']['text'] for e in s.hub.history if e['event']=='caption']
            assert captions==[text]
            assert s.captions[-1]['text']==text
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
