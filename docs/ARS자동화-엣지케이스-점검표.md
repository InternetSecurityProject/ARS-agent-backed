작성일: 2026-10-25

본 문서는 통화 중 발생할 수 있는 사용자 입력과 ARS 반응의 경우의 수를 나열하고, 각 경우에 대한 서버(`session.py`)의 처리와 이를 확인하는 자동 시험(`test_flows.py`)을 대응시킨 점검표이다. 서버 동작을 바꿀 때는 해당 행의 처리와 시험을 함께 갱신한다. 명세 조항 번호는 API 명세서 v0.3 기준이다.

---

## 1. 경우의 수를 나누는 축

### 1.1 사용자 입력 (앱 → 서버)

| 입력 | API | 비고 |
| --- | --- | --- |
| DTMF 키 직접 입력 | `actions(type=key)` | 안내에서 들은 키일 수도, 없는 키일 수도 있다 |
| 재질문 음성 답변 | `actions(type=answer)` | `request(kind=clarification)` 에 대한 응답 |
| 재질문 후보 선택 | `actions(type=select)` | 위와 같음 (보조 수단) |
| 민감정보 입력 | `actions(type=secure_input)` | `request(kind=secure_input)` 에 대한 응답 |
| 다시 말하기 | `actions(type=revise_goal)` | 첫 번호 입력 전까지 |
| 통화 종료 | `actions(type=cancel)` | 모든 상태에서 가능 |
| 무응답 | — | 요청에 답하지 않음 (15초·30초·60초 경과) |
| 채널 연결·해제 | `WSS /events`, `WSS /audio` | 상태에 영향 없음 |

### 1.2 ARS 반응 (상대 → 서버)

안내 한 단락의 분류(AI `classify`)와 시간 관계로 나눈다.

| 반응 | 분류 또는 사건 |
| --- | --- |
| 다음 메뉴·확인 질문 | `menu`, `confirm` |
| 민감정보 입력 요청 | `sensitive` |
| 입력값 되읽기 | `secure_window` 중 "입력하신 … 맞으면 1번" |
| 결과 안내 | `result` |
| 입력 오류 안내 | `input_error` ("잘못 누르셨습니다", "입력 시간이 초과되었습니다") |
| 같은 메뉴 재생 | ARS 자체 입력 대기 시간 경과 후 반복 |
| 종료성 안내 | `message_sent`, `out_of_hours`, `rejected`, "입력이 없어 통화를 종료합니다"(`info`) |
| 상담원 | `agent_wait`, `agent_human` |
| 음성 응답 요구 | `voice_request` |
| 무음 | 키를 보냈으나 아무 소리가 없음 |
| 긴 안내 | 키를 보낸 뒤 다음 안내가 8초를 넘김 |
| 키보다 먼저 시작된 안내 | 안내 도중 키를 보낸 경우의 그 안내 (키 이전 상태의 안내) |
| 상대가 끊음 | `remote_hangup` (결과 직후, 응답 대기 중, 상담원 대기 중 등) |

### 1.3 서버 상태

`status`(3.1)와 다음 내부 플래그의 조합이다.

| 플래그 | 의미 |
| --- | --- |
| `pending` | 열려 있는 사용자 요청 (`clarification` 또는 `secure_input`) |
| `key_outstanding` | 키를 보냈고 아직 ARS 의 반응 안내를 듣지 못함 |
| `last_key_at` | 마지막 키 전송 시각. 이보다 먼저 시작된 안내는 키 이전 상태의 안내로 본다 |
| `last_send_secure` | 마지막으로 보낸 키가 민감정보 입력값 |
| `secure_window` | 민감정보 입력 직후 (되읽기 마스킹 구간) |
| `ai_hold_until` | 사용자 직접 선택 뒤 AI 메뉴 판단 보류 (10초) |
| `key_queue` | 이전 계층 복귀 중 차례로 누를 키 |
| `hangup_seen` | 상대가 끊었고 마지막 안내·요약을 처리하는 중 |
| `result_heard` | 결과 안내를 들음 (요약 완료 여부와 무관) |

---

## 2. 점검표

표기: 처리 열의 조항 번호는 명세서 조항. 시험 열은 `test_flows.py` 의 함수명.

### A. DTMF 키 직접 입력 (`key`)

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| 상대가 받기 전 | `409 invalid_state` | — (코드 `_act_key`) |
| 민감정보 입력 요청이 열려 있음 | `409 sensitive_input_blocked`. 경로에 숫자가 남지 않는다 | `test_key_rejected_during_secure_input_request` |
| 재질문이 열려 있음 | 요청을 닫고 키 전송, `awaiting_user` 해제, AI 판단 10초 보류 | `test_user_direct_key_and_input_rejected` |
| AI 탐색 중, 보류 중 결과 안내 수신 | 보류는 메뉴 판단만 미루므로 결과는 처리한다 | `test_result_during_user_key_hold_is_not_lost` |
| AI 탐색 중, 보류가 끝남 | 보류 중 들은 가장 최근 메뉴를 판단한다 | `test_result_during_user_key_hold_is_not_lost` (보류 해제 경로 `_resume_after_hold`) |
| 이전 계층 복귀 중 | 복귀 대기열을 비우고 사용자 키를 따른다 | — (코드 `_act_key`) |
| 안내에 없던 키 → ARS 오류 안내 | `input_rejected(actor=user)`, 같은 메뉴 재판단 | `test_user_direct_key_and_input_rejected` |
| 안내에 없던 키 → ARS 가 새 메뉴로 이동 | 새 메뉴를 판단한다. 경로에는 `"9 입력"` 으로 남는다 | (ARS 숨은 키와 구분 불가. 3장 1번) |
| 마지막 키 전송 뒤 새 메뉴를 듣기 전에 누름 (민감정보 안내 중, 안내가 끝나기 전 등) | 지난 메뉴의 선택지로 해석하지 않고 `"5 입력"` 으로 기록, 지나온 계층에 넣지 않음 | `test_key_pressed_before_next_menu_is_not_labeled_as_old_menu_choice` |
| 업무 완료(`agent_done`) 뒤 | 키만 전송. 무응답 재전송(다시 듣기 키)을 걸지 않는다 | `test_no_auto_repeat_after_agent_done` |
| 상담원 연결 중 | 키만 전송. 자동 재전송 없음 | `test_no_auto_repeat_when_connected_to_agent` |
| 키 전송 실패 (회선 끊김) | `409 invalid_state`, 세션은 유지 (종료는 `_on_end` 가 처리) | `test_key_send_failure_is_409_for_user_and_harmless_for_ai` |
| 상대가 끊은 뒤 | `409 invalid_state` | — (코드 `action`) |

### B. 재질문 응답 (`answer`, `select`)

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| `low_confidence` 에 답변 | 목적 보충 후 같은 메뉴 재판단 | `test_low_confidence_then_answer` |
| `ambiguous` 에 답변 | 위와 같음 | `test_ambiguous_then_answer` |
| 후보 선택 (`select`) | 고른 키 전송, `awaiting_user` 해제 | `test_select_releases_awaiting_user` |
| 후보에 없는 키 `select` | `400 invalid_request` | `test_select_rejects_non_candidate_and_answer_rejects_stale` |
| 닫힌·만료된 요청에 답변 | `409 invalid_state` | `test_select_rejects_non_candidate_and_answer_rejects_stale` |
| `critical_action` 에 `answer` | `409` (`select` 만 허용) | `test_critical_confirm_is_never_pressed_by_ai` |
| 음성 응답 요구(`unknown_prompt`, 후보 없음)에 답변 | 지난 메뉴로 판단하지 않고 다음 안내를 기다린다 | `test_answer_to_voice_request_waits_for_next_menu` |
| `loop` 재질문에 응답 | 반복 횟수 초기화 | `test_loop_after_two_revisits` (발생), 코드 `reset_loop` |
| 답변 처리 중 AI(`merge_answer`) 실패 | 답변 문장을 새 목적으로 사용 | `test_ai_goal_summary_and_merge_failure_fallbacks` |
| 재질문 대기 중 ARS 가 같은 메뉴를 다시 읽음 | 계속 대기. 반복으로 세지 않음 | `test_timeout_notice_while_clarification_pending_is_caption_only` |
| 재질문 대기 중 ARS 의 "입력 시간이 초과되었습니다" | 키 오류가 아니다. 자막만 전송, 요청 유지 | `test_timeout_notice_while_clarification_pending_is_caption_only` |

### C. 민감정보 입력 (`secure_input`)

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| 정상 입력 | 톤 전송 후 폐기, `200`. 자막·경로·이벤트에 숫자 없음 | `test_goal_first_secure_input_completed` |
| 사용자가 종료 키(#)까지 누름 | 종료 키를 중복해 붙이지 않는다 | `test_secure_input_terminator_is_not_duplicated` |
| 입력 뒤 ARS 오류 안내 | `input_rejected(keys=null, actor=user)`. 메뉴 키 오류로 세지 않아 재질문이 열리지 않는다. ARS 가 입력 안내를 반복하면 새 요청 | `test_secure_input_error_is_not_a_menu_error_and_digits_stay_out` |
| 입력 뒤 ARS 가 입력값을 되읽음 (확인 표현 포함) | 숫자를 가린 뒤 판단만 하고 자막 생략. 확인 질문은 사용자에게 묻는다 | `test_mask_digits` (마스킹), 코드 `_paragraph` 되읽기 분기 |
| 입력 뒤 같은 자릿수의 숫자열이 들림 (확인 표현 없음) | 입력 자릿수와 같은 숫자열만 가린다. 금액 등 다른 자릿수는 남긴다 | `test_digits_matching_secure_input_length_are_masked_but_amounts_are_not` |
| 입력 대기 중 아무 키도 안 보냈는데 "입력 시간이 초과되었습니다" | 키 오류가 아니다. 자막만, 요청 유지 | `test_timeout_notice_while_secure_pending_is_caption_only` |
| 입력 대기 중 ARS 가 입력 안내를 반복 | 같은 요청 유지 (새 요청 없음) | `test_timeout_notice_while_secure_pending_is_caption_only` |
| 입력 대기 중 ARS 가 메뉴로 돌아감 | 입력 요청을 닫고 그 메뉴를 판단한다 | `test_secure_request_closes_when_ars_returns_to_menu` |
| 입력 대기 중 `revise_goal` | 요청 유지, 목적만 변경 | `test_revise_goal_before_first_key` (일반), 코드 `keep_secure` |
| 입력 대기 중 상대가 끊음 | `user_input_timeout` | `test_remote_hangup_while_waiting_for_user_is_user_input_timeout` |
| 외부 STT 사용 시 되읽기 구간 | 인식하지 않는다 (`is_local=False`) | — (코드 `_stt_worker`) |

### D. 다시 말하기 (`revise_goal`)

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| 첫 메뉴 재질문 중 | 요청을 닫고 새 목적으로 첫 메뉴 재판단, 새 `goal_summary` | `test_revise_goal_before_first_key` |
| 첫 번호를 누른 뒤 | `409 invalid_state` | `test_revise_goal_before_first_key` |
| 상대가 받기 전 | 목적만 바꾸고 응답 뒤 첫 메뉴부터 탐색 | — (코드 `_activate_agent`) |
| AI(`summarize_goal`) 실패 | 대체 문장으로 진행 | `test_ai_goal_summary_and_merge_failure_fallbacks` |

### E. 통화 종료 (`cancel`)

| 상황 | `end_reason` | 시험 |
| --- | --- | --- |
| 탐색 중 | `user_canceled` | `test_cancel_reason_by_state` |
| 결과 안내를 들은 뒤 (요약 완료 여부 무관) | `completed` | `test_cancel_reason_by_state`, `test_goal_first_secure_input_completed` |
| 상담원 연결 중 | `agent_call_ended` | `test_agent_connection_handoff` |
| 이미 종료됨 | `409 invalid_state` | `test_rest_and_websocket` |
| 상대가 끊은 뒤 마무리 처리 중 | 바로 종료 (`hangup` 재요청 없음) | — (코드 `action`) |

### F. 사용자 무응답

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| 재질문 30초 | 지금 메뉴의 다시 듣기 키 1회. ARS 가 말하는 중이면 끝난 뒤. 요청 유지 | `test_clarify_repeat_key_waits_until_ars_is_quiet` |
| 재질문 60초, 상담원 선택지 있음 | 요청 무효, 상담원 연결 키 전송 | `test_low_confidence_no_answer_goes_to_agent_menu` |
| 재질문 60초, 상담원 선택지 없음 | 요청 유지, 계속 대기 | `test_ambiguous_no_answer_keeps_waiting` |
| 민감정보 입력 무응답 | 서버 동작 없음. ARS 가 끊으면 `user_input_timeout` | `test_remote_hangup_while_waiting_for_user_is_user_input_timeout` |
| ARS 가 "입력이 없어 통화를 종료합니다" 후 끊음 | 자막 전송 후 `user_input_timeout` | `test_ars_gives_up_and_hangs_up_while_waiting_for_user` |

### G. ARS 쪽 사건 (AI 탐색 중)

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| 키 전송 뒤 8초 동안 무음 | 키를 누른 그 메뉴의 다시 듣기 키 1회 (같은 키 재전송 없음) | `test_no_response_after_silence_presses_that_menus_repeat_key` |
| 키 전송 뒤 다음 안내가 8초보다 김 | ARS 가 말하는 동안 아무 키도 누르지 않는다 | `test_no_repeat_key_while_next_menu_is_long` |
| 키 전송 뒤 소리는 났으나 인식 문장이 없음 | 이전 메뉴의 다시 듣기 키를 누르지 않는다 | `test_no_previous_menu_repeat_key_after_ars_spoke` |
| 다시 듣기 키로 같은 메뉴 재생 | 반복으로 세지 않음 | `test_no_response_replay_is_not_counted_as_revisit` |
| 키보다 먼저 시작된 메뉴 안내 (AI 판단이 ARS 입력 대기보다 늦은 경우) | 자막만 남기고 판단하지 않는다 (같은 키 두 번 전송 방지) | `test_stale_menu_heard_before_key_is_not_decided_again` |
| 키보다 먼저 시작된 오류 안내 | 지금 키에 대한 거부로 세지 않는다 | `test_stale_error_heard_before_key_is_not_a_rejection` |
| 키 전송 뒤 "잘못 누르셨습니다" 1회 | `input_rejected`, 같은 메뉴 재판단 | `test_input_rejected_needs_two_consecutive_errors` |
| 같은 단계 오류 2회 연속 | `request(clarification, input_rejected)` | `test_input_rejected_needs_two_consecutive_errors` |
| 다른 단계로 넘어간 뒤 오류 | 횟수를 다시 센다 | `test_input_rejected_needs_two_consecutive_errors` |
| 이전 계층 복귀 중 대기열 키 거부 | 메뉴가 다시 나오면 같은 키 재전송 | `test_backtrack_queue_retries_key_rejected_by_ars` |
| 키 전송 결과로 지나온 메뉴 재도착 2회 | `request(clarification, loop)` | `test_loop_after_two_revisits` |
| 맞는 선택지 없음 → 이전 계층에 있음 | '이전 메뉴' 키로 복귀 후 재판단 | `test_backtrack_to_previous_layer` |
| 맞는 선택지 없음 → 이전 계층에도 없음 | `ended(not_found)` | `test_not_found_after_checking_previous_layer`, `test_not_found_at_first_layer` |
| 중요 선택(결제·계약·해지) 확인 질문 | 키를 보내지 않고 `critical_action` 재질문 | `test_critical_confirm_is_never_pressed_by_ai` |
| 결과 안내 | `summary(result)` → `agent_done`, 통화 유지 | `test_goal_first_secure_input_completed` |
| 결과 안내 직후 상대가 끊음 (요약 생성 중) | 요약을 마친 뒤 `summary` → `ended(completed)`. 종료 뒤 상태 되돌림 없음 | `test_hangup_during_result_summary_is_completed` |
| 결과를 말하고 무음 없이 바로 끊음 | 끊기 전까지의 음성을 인식하여 `completed` | `test_hangup_right_after_result_speech_is_recognized` |
| 상담원 대기 → 상담원 음성 | `summary(handoff)` → `connected_to_agent` | `test_agent_connection_handoff` |
| 상담원 대기 중 끊김 | `failed` | `test_remote_hangup_while_waiting_for_user_is_user_input_timeout` |
| 문자 발송·업무시간 외·거절 안내 | `message_sent`(partial) / `out_of_hours` / `ars_rejected`(failed) 로 종료 | `test_ars_ending_notices_end_the_call_with_matching_reason` |
| 음성 응답 요구 | `request(clarification, unknown_prompt)` (TTS 미구현) | `test_answer_to_voice_request_waits_for_next_menu` |
| 업무 완료 뒤의 안내 | 자막만 전송 | `test_no_auto_repeat_after_agent_done` |

### H. AI 모듈·회선 오류

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| `classify` 실패 | 그 안내는 자막만 남기고 계속 듣는다 | `test_ai_classify_failure_keeps_call_alive` |
| `score`(메뉴 판단) 실패 | `request(clarification, unknown_prompt)` 로 사용자에게 묻는다 | `test_ai_score_failure_asks_user_instead_of_stalling` |
| `answer_confirm` 실패 또는 판단 불가(`null`) | `unknown_prompt` 재질문 | `test_confirm_question_without_answer_asks_user` |
| `summarize_goal` 실패 | 대체 문장 "'목적' 업무를 찾아 드릴게요." | `test_ai_goal_summary_and_merge_failure_fallbacks` |
| `merge_answer` 실패 | 답변 문장을 새 목적으로 사용 | `test_ai_goal_summary_and_merge_failure_fallbacks` |
| `summarize_result` 실패 | 결과 안내 원문을 요약으로 사용 | — (코드 `_paragraph` 결과 분기) |
| `handoff_script` 실패 | 대체 문장 "목적 건으로 연락드렸습니다." | — (코드 `_handoff`) |
| AI 판단 중 통화 종료 | 판단 결과를 버린다 (키 전송 없음) | — (코드 `_decide_inner` 의 종료 확인) |
| 키 전송 실패 (AI 경로) | 예외 없이 멈추고 종료는 `_on_end` 가 처리 | `test_key_send_failure_is_409_for_user_and_harmless_for_ai` |
| 발신 실패·무응답·통화 중 | `failed` / `no_answer` / `busy` | `test_dial_failures_map_to_end_reasons`, `test_clawops_backend_wiring` (연결부) |
| 최대 통화 시간(600초) | `max_duration` 으로 종료 | `test_max_call_duration_ends_call` |

### I. 채널과 세션

| 상황 | 서버 처리 | 시험 |
| --- | --- | --- |
| 이벤트 채널 연결·재연결 | `snapshot` 1회 후 이벤트 전송 | `test_rest_and_websocket` |
| 종료된 세션에 이벤트 채널 연결 | `snapshot(ended)` 후 서버가 닫는다 | `test_rest_and_websocket` |
| 종료된 세션에 오디오 채널 연결 | 즉시 닫는다 | `test_rest_and_websocket` |
| 같은 세션에 여러 클라이언트 | 모두 같은 이벤트·오디오 수신 | `test_live_console_against_live_server` (붙어서 듣기) |
| 다른 통화 진행 중 세션 생성 | `409 invalid_state` | `test_one_call_at_a_time` |
| 상대가 끊은 뒤 마무리 처리 중 세션 생성 | 허용 (회선은 비어 있음) | `test_new_call_allowed_while_previous_finishes_after_hangup` |
| 종료 뒤 토큰 | 24시간 뒤 `401` | — (코드 `SessionManager.get`) |
| 종료 뒤 늦게 끝난 처리 | 상태·이벤트를 바꾸지 않는다 (`ended` 가 마지막 이벤트) | `test_hangup_during_result_summary_is_completed` |

---

## 3. 미해결·검증 필요 항목

1. 안내에 없던 키로 ARS 가 새 메뉴로 이동한 경우, STT·AI 가 선택지를 놓친 것인지 ARS 의 숨은 키인지 서버는 구분할 수 없다. 통화 동작은 바꾸지 않으며, 발생 빈도는 실통화 로그로 센다.
2. 민감정보 요청 자체를 AI 가 알아보지 못한 경우는 서버가 감지할 수 없다. 사용자가 `key` 로 숫자를 누르면 경로(PathStep)에는 남는다. 누른 키 값이 AI 모듈로 전달되는 경로를 막는 수정은 별도 브랜치에서 검토한다.
3. 실제 ARS 의 입력 대기 시간(KB 첫 메뉴 4.5초)보다 AI `classify`+`score` 가 늦으면 ARS 가 메뉴를 다시 읽는다. 키보다 먼저 시작된 안내를 판단에서 제외하는 규칙으로 중복 전송은 막았으나, 응답 시간 자체(목표 2초, 9.3)는 실측이 필요하다.
4. 음성 응답 요구(음성봇)에 대한 음성 송출(TTS)은 미구현이다.
5. 되읽기 마스킹은 STT 가 숫자를 숫자 또는 한글 숫자로 적는 경우만 다룬다. 다른 표기("구공공일영일" 등 혼용)는 실통화 인식 결과를 보고 보완한다.
6. 이전 계층 복귀를 '처음으로' 키로 하는 경우, 지나온 키를 되밟는 경로에 사용자가 실수로 누른 키가 포함될 수 있다 (`layers` 기록 기준). 해당 키는 ARS 에서 오류 안내 후 메뉴 재생으로 이어지므로 탐색은 계속되지만, 경로 재구성 시 제외할지 검토한다.
