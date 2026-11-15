# Cancellation inquiry approval guard: local verification

Base: `e51a4029fe61d8310024abd9334ed42eb79d946c` (main after PR #9).

The change narrows the inquiry exception; it does not infer transaction safety
from an institution name, caller goal, or a model's low critical score. A clear
inquiry label must be immediately assigned to a supported numeric/star/hash key
and end as a normal option/press instruction. Compound execution text and
processing/application confirmation questions retain the approval gate even in
another option or sentence. Recognizable adjacent inquiry choices are supported
without punctuation. New execution signals are scoped to cancellation context;
unrelated form/application menus retain their existing rule behavior.

## Historical records

- Screened 295 authorized archival call records, including the supplied Toss and CGV records.
- Only public STT before the first secure-input request/classification or user input was eligible.
- Excluded 451 later segments and 59 possible readback/long-digit segments.
- 622 public segments; exact repeated menu texts deduplicated per institution.
- Compared 136 distinct menus: 131 from the 10 identified institutions below,
  and 5 from unidentified sources (not attributed to a named institution).
- Old/new extraction retained the same option labels, keys, kinds and navigation fields.
- No critical=True menu became critical=False. The 131 identified menus retained
  their existing flags; compound approval regressions are separately exercised
  with explicit expected outcomes in `test_cancellation_approval_guard.py`.

| Institution | Distinct menus compared |
| --- | ---: |
| KB국민은행 | 37 |
| 하나은행 | 22 |
| 국민건강보험공단 | 17 |
| 근로복지공단 | 4 |
| 우체국 | 35 |
| 한국교통안전공단 | 2 |
| 한국장학재단 | 7 |
| 정부24 | 1 |
| 토스 | 2 |
| CGV | 4 |

Records were made with different historical code/model versions. These are
deterministic rule-regression checks, not real-call success rates or model/STT
accuracy measurements. The screened corpus contains no newly changed menu, so
it cannot by itself demonstrate the dangerous compound wording is fixed. New
rule, Navigator and mock Session tests reproduce that behavior independently,
including a model critical=False response and the requirement for explicit user
selection before any mock DTMF is sent. Existing model critical=True protection
and the alternate interpreter's stricter rules are preserved.

Public representative excerpts are in `testdata/approval_guard_ars_menus.json`.
No user goals, entered values, readbacks, API keys or raw audio are copied there.
The complete archival comparison was run locally; only the public representative
excerpts are included in the repository.

Verification used no real calls, external model requests or running-server
restarts. The separate CGV repetition-boundary work is excluded from this change.

## Review regressions

Independent read-only review reproduced execution in a later option/sentence
being auto-selected, and a punctuation-free inquiry menu becoming newly blocked.
Failing Navigator tests now cover both. Additional tests preserve unrelated
form-application menus after the public-corpus audit found overbroad wording.

Final verification: **856 passed, 1 skipped** with
`python -m pytest -q -o cache_dir=/tmp/ars-approval-pytest-cache`.
