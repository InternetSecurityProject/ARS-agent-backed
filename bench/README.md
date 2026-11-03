# bench — 실통화 시나리오 벤치마크

모델별 음성 인식(STT) 정확도, 안내 해석·판단(되묻기 판정 포함) 정확도, 단계별 속도를 같은 실통화 녹음으로 비교하는 도구.

## 구성

| 파일 | 역할 |
| --- | --- |
| `scenarios.json` | 시나리오 정의: 전화번호, 목적, 키 경로, 목적 변형별 기대 판단, 가짜 번호 안전 규칙 |
| `record.py` | 경로 고정 녹음: 시나리오 키 경로대로 실제 ARS 에 발신·녹음 (`raw/<ID>-<n>.wav`, `.json`) |
| `make_clips.py` | 녹음 → 단락 WAV + `clips.json` 초안 (구역 A/B, 판단 지점, 숫자 정답 자동 표시) |
| `stt_eval.py` | 모델별 음성 인식 평가 |
| `interp_eval.py` | 모델별 안내 해석·판단 평가 |
| `metrics.py`, `data.py` | 지표 계산, 데이터 읽기 |

## 절차

1. 녹음 전용 서버 실행 (AI 가 키를 누르지 않도록 모의 해석기 사용)
   ```
   $env:INTERPRETER = "interpreter:MockInterpreter"
   $env:POST_KEY_CAPTION_S = "0"
   python app.py --port 8011
   ```
2. 시나리오 녹음 (다른 창)
   ```
   python -m bench.record K1 --base http://127.0.0.1:8011 --app-key <APP_KEY>
   python -m bench.record K1 --take 2 --base http://127.0.0.1:8011 --app-key <APP_KEY>
   ```
3. 단락 분리와 정답 초안
   ```
   python -m bench.make_clips bench/raw/K1-1.json --stt gemini --label
   ```
4. 정답 확정: `bench/data/K1-1/` 의 단락을 들으며 `clips.json` 의 `text`·`kind`·`options`·`digits`·`cut` 수정 후 `verified: true`
5. 평가
   ```
   python -m bench.stt_eval --device cuda --models kimthegarden/whisper-small-ko-low-qual-voice-ct2 large-v3 gemini rtzr gemini+interp --json bench/out/stt.json
   python -m bench.interp_eval --models gemini-3.5-flash-lite gemini-3.6-flash --json bench/out/interp.json
   python -m bench.interp_eval --hyp bench/out/stt.json --hyp-model kimthegarden/whisper-small-ko-low-qual-voice-ct2
   ```

## 지표

| 평가 | 지표 |
| --- | --- |
| 음성 인식 | 글자 오류율(전체·구역 A·구역 B), 메뉴 키 정확도, 숫자열 완전 일치·자리 오류율, 누락, 환각, 단락당 시간(중앙값·95%·최대) |
| 해석·판단 | 안내 종류 정확도, 선택지 추출 정밀도·재현율, 판단 정확도, 되묻기 정밀도·재현율, 위험 자동입력, 오류, 해석 시간·판단 지점 시간(중앙값·95%) |
| 결합 | `gemini+interp`: 받아쓰기+분류 1회 호출 시간, `interp_eval --hyp`: STT 인식 문장으로 낸 판단 정확도 |

## 기대 판단 표기 (`scenarios.json`)

`press:3` 키 3 누름, `ask` 질문(`ask:ambiguous` 처럼 사유까지 지정 가능), `nopress` 누르지 않음, `secure` 민감정보 입력 요청, `backtrack`·`end` 이전 계층·종료, `any` 채점 안 함, `A|B` 둘 중 하나.

## 안전 규칙

- 존재할 수 없는 가짜 번호만 입력 (주민등록번호 월 13 이상·일 32 이상, 계좌·팩스 0 연속)
- 되읽기 뒤 발급·발송이 진행되는 경로에서는 1번(맞음) 금지
- 문자 발송은 '지금 전화한 번호로 받기'(070)만
- 본인 번호로 녹음 금지
