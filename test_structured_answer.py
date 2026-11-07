"""A clarified preference stays explicit and cannot answer a different menu."""
from test_system_recovery import make_session
from test_jev_phrase import AI,item,run

TEXT='문자 안내는 1번, 상담원 연결은 2번.'
class AnswerContextAI(AI):
    async def decide(self,state,questions):
        if 'matching_key' in questions:
            resolved=(state.get('current_user_answer',{}).get('answer')=='문자로 받고 싶어요')
            self.route='proceed' if resolved else 'needs_info'
        return await super().decide(state,questions)


def test_spoken_preference_is_explicit_when_menu_is_rejudged():
    async def go():
        c=AnswerContextAI({'1':item('문자 안내는'),'2':item('상담원 연결은')})
        s=make_session(c)
        try:
            await s.on_paragraph(TEXT)
            assert s.pending and not s.call.keys
            await s.action({'type':'answer','request_id':s.pending['request_id'],'text':'문자로 받고 싶어요'})
            assert s.call.keys==['1'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_answer_to_previous_menu_does_not_silently_resolve_a_new_question():
    async def go():
        c=AnswerContextAI({'1':item('문자 안내는'),'2':item('상담원 연결은')})
        s=make_session(c)
        try:
            await s.on_paragraph(TEXT)
            await s.action({'type':'answer','request_id':s.pending['request_id'],'text':'문자로 받고 싶어요'})
            assert s.call.keys==['1']
            c.items={'1':item('서울 지역은'),'2':item('부산 지역은')}
            await s.on_paragraph('서울 지역은 1번, 부산 지역은 2번.')
            assert s.call.keys==['1'] and s.pending
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_pending_old_question_answer_is_not_attached_to_newly_heard_menu():
    async def go():
        c=AnswerContextAI({'1':item('문자 안내는'),'2':item('상담원 연결은')});s=make_session(c)
        try:
            await s.on_paragraph(TEXT);rid=s.pending['request_id']
            c.items={'1':item('서울 지역은'),'2':item('부산 지역은')}
            await s.on_paragraph('서울 지역은 1번, 부산 지역은 2번.')
            await s.action({'type':'answer','request_id':rid,'text':'문자로 받고 싶어요'})
            assert not s.call.keys and s.pending
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_unmasked_readback_question_is_never_retained_as_model_context():
    from interpreter import Prompt
    from test_jev_phrase import engine
    async def go():
        i=engine(AnswerContextAI({}));i.last_menu=Prompt('menu','입력하신 번호는 ******입니다. 맞으면 1번, 틀리면 2번.')
        await i.merge_answer('확인','입력하신 번호는 01012345678입니다. 맞으면 1번.','잘 모르겠어요')
        assert '01012345678' not in repr(i._clarification_answer)
    run(go())


def test_numeric_old_request_answer_uses_the_asked_menu_not_latest_menu():
    async def go():
        c=AnswerContextAI({'1':item('문자 안내는'),'2':item('상담원 연결은')});s=make_session(c)
        try:
            await s.on_paragraph(TEXT);rid=s.pending['request_id']
            c.items={'1':item('서울 지역은'),'2':item('부산 지역은')}
            await s.on_paragraph('서울 지역은 1번, 부산 지역은 2번.')
            await s.action({'type':'answer','request_id':rid,'text':'1번'})
            assert '서울' not in s.goal and '문자 안내' in s.goal
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
