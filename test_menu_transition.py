"""Regression tests for automatic ARS transfers. Fake provider/phone only."""
import pytest

from test_jev_phrase import item, run, engine
from test_system_recovery import RecoveryAI, make_session


@pytest.mark.parametrize('text',[
    '카드 ARS로 연결해 드리겠습니다.',
    '병원 예약 센터로 연결합니다.',
    '담당 부서의 자동 안내로 연결하겠습니다.',
])
@pytest.mark.parametrize('kind_conf',[.4,.99])
def test_transfer_without_actionable_keys_is_listening_not_an_empty_menu(text,kind_conf):
    async def go():
        c=RecoveryAI({},kind='menu',kind_conf=kind_conf);s=make_session(c)
        try:
            await s.on_paragraph(text)
            assert s.current is None or s.current.kind=='info'
            assert not c.repair_seen and s.pending is None and not s.call.keys
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_transfer_cannot_spend_the_following_real_menus_reanalysis():
    async def go():
        c=RecoveryAI({},kind_conf=.4);s=make_session(c)
        try:
            await s.on_paragraph('예약 센터로 연결하겠습니다.')
            c.kind_conf=.99;c.items={'4':item('예약 변경은',confidence=.4),'6':item('검사 결과는')}
            c.repair_items={'4':item('예약 변경은'),'6':item('검사 결과는')};c.key='4';c.repair_seen=False
            await s.on_paragraph('예약 변경은 4번, 검사 결과는 6번.')
            assert c.repair_seen and s.call.keys==['4'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_a_new_menu_without_intervening_dtmf_gets_its_own_reanalysis():
    async def go():
        c=RecoveryAI({'1':item('예약 문의는',confidence=.4)},route='unknown')
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 1번.')
            c.items={'4':item('검진 일정 변경은',confidence=.4),'6':item('검사 결과는')}
            c.repair_items={'4':item('검진 일정 변경은'),'6':item('검사 결과는')}
            c.route='proceed';c.key='4';c.repair_seen=False
            await s.on_paragraph('검진 일정 변경은 4번, 검사 결과는 6번.')
            assert c.repair_seen and s.call.keys==['4'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_new_menu_with_same_numbers_but_different_trusted_services_can_recover():
    async def go():
        c=RecoveryAI({'1':item('주차 요금은'),'2':item('진료 시간은')},route='unknown')
        s=make_session(c)
        try:
            await s.on_paragraph('주차 요금은 1번, 진료 시간은 2번.')
            c.items={'1':item('수술 접수는'),'2':item('진료 예약은')};c.repair_seen=False
            await s.on_paragraph('수술 접수는 1번, 진료 예약은 2번.')
            assert c.repair_seen and not s.call.keys and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_asr_preface_and_formatting_drift_does_not_buy_a_second_replay():
    async def go():
        c=RecoveryAI({'1':item('예약 문의는',confidence=.4),'9':item('다시 듣기는','repeat')})
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 1번, 다시 듣기는 9번.')
            c.repair_seen=False
            await s.on_paragraph('입력 시간이 경과했습니다. 예약 문의는 1번. 다시 듣기는 9번.')
            assert s.call.keys==['9'] and not c.repair_seen and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_confidence_improvement_in_same_options_does_not_reset_repair():
    async def go():
        c=RecoveryAI({'1':item('예약 문의는',confidence=.4),'2':item('주차 요금은')},route='unknown')
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 1번, 주차 요금은 2번.')
            c.items={'1':item('예약 문의는'),'2':item('주차 요금은')};c.repair_seen=False
            await s.on_paragraph('안녕하세요. 예약 문의는 1번, 주차 요금은 2번.')
            assert not c.repair_seen and s.pending is None and not s.call.keys
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_unlimited_changed_menu_numbers_do_not_cause_unlimited_reanalysis():
    async def go():
        c=RecoveryAI({},route='unknown');s=make_session(c)
        repairs=0
        try:
            for key in '12345678':
                c.items={key:item('예약 문의는',confidence=.4)};c.key=key;c.repair_seen=False
                await s.on_paragraph(f'예약 문의는 {key}번.')
                repairs+=int(c.repair_seen)
            assert 1<repairs<=3 and not s.call.keys and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_a_business_menu_about_connecting_is_still_actionable():
    async def go():
        c=RecoveryAI({'4':item('예약 센터 연결은')},key='4');s=make_session(c)
        try:
            await s.on_paragraph('예약 센터 연결은 4번.')
            assert s.call.keys==['4'] and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_first_extraction_transport_error_still_allows_source_grounded_repair():
    class FirstError(RecoveryAI):
        failed=False
        async def decide(self,state,questions):
            if not self.failed:
                self.failed=True
                raise TimeoutError('not-a-real-private-error')
            return await super().decide(state,questions)
    async def go():
        c=FirstError({'4':item('예약 변경은')},key='4');s=make_session(c)
        try:
            await s.on_paragraph('예약 변경은 4번.')
            assert s.call.keys==['4'] and c.repair_seen and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_repair_correcting_labels_in_identical_audio_does_not_create_another_menu():
    async def go():
        c=RecoveryAI({'1':item('주차 요금은'),'2':item('예약 문의는')},route='unknown',
            repair_items={'1':item('예약 문의는'),'2':item('주차 요금은')})
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 1번, 주차 요금은 2번.')
            assert sum(bool(state.get('system_reanalysis')) for state,_ in c.calls)==1
            assert not s.call.keys and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_corrected_same_menu_with_added_greeting_cannot_replay_twice():
    async def go():
        c=RecoveryAI({'1':item('주차 요금은'),'2':item('예약 문의는'),
                      '9':item('다시 듣기는','repeat')},route='unknown',
            repair_items={'1':item('예약 문의는'),'2':item('주차 요금은'),
                          '9':item('다시 듣기는','repeat')})
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 1번, 주차 요금은 2번, 다시 듣기는 9번.')
            await s.on_paragraph('안녕하세요. 예약 문의는 1번, 주차 요금은 2번, 다시 듣기는 9번.')
            assert s.call.keys==['9']
            assert sum(bool(state.get('system_reanalysis')) for state,_ in c.calls)==1
        finally:await s.finish('user_canceled',hangup=False)
    run(go())


def test_swapped_star_hash_business_bindings_are_a_distinct_menu():
    async def go():
        c=RecoveryAI({'*':item('예약 문의는'),'#':item('주차 요금은')},route='unknown',key='*')
        s=make_session(c)
        try:
            await s.on_paragraph('예약 문의는 *, 주차 요금은 #.')
            c.items={'#':item('예약 문의는'),'*':item('주차 요금은')};c.repair_seen=False
            await s.on_paragraph('예약 문의는 #, 주차 요금은 *.')
            assert c.repair_seen and not s.call.keys and s.pending is None
        finally:await s.finish('user_canceled',hangup=False)
    run(go())
