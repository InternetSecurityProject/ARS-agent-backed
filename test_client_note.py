"""앱이 버린 누름(client_note)을 통화 기록에만 남기고 아무 동작도 하지 않는지 확인."""
from test_flows import manual, run


def test_client_note_is_recorded_and_does_nothing():
    async def go():
        s, call = await manual("통장 잔액 알고 싶어", record=True)
        keys_before, pending_before, status_before = list(call.keys), s.pending, s.status
        out = await s.action({"type": "client_note", "what": "select", "reason": "request_replaced", "request_id": "req_9", "key": "1"})
        await s.action({"type": "client_note", "what": "secure_key", "reason": "expired", "key": "7"})
        notes = [e for e in s.recorder.events if e.get("action") == "client_note"]
        text = s.recorder.text_log()
        state = (list(call.keys), s.pending, s.status)
        await s.finish("user_canceled")
        return out, notes, text, (keys_before, pending_before, status_before), state

    out, notes, text, before, after = run(go())
    assert out == {"accepted": True} and before == after                      # 키·요청·상태 그대로
    assert notes[0]["what"] == "select" and notes[0]["reason"] == "request_replaced" and notes[0]["key"] == "1"
    assert notes[0]["request_id"] == "req_9"
    assert notes[1]["what"] == "secure_key" and "key" not in notes[1]          # 메뉴 선택이 아니면 키 값은 남기지 않음
    assert "앱이 무시한 누름 select (이유 request_replaced) 키 1" in text
