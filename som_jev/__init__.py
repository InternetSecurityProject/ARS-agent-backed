"""som 방식 해석·탐색. 같은 저장소의 SOURCE_BRANCH 브랜치(SOURCE_REV) 모듈을 그대로 옮긴 패키지.

해석(UnifiedJevInterpreter)과 탐색 판단(Navigator)은 원본 코드와 동일. 원본과 다른 것은 모듈 가져오기 경로뿐임
(같은 패키지 안 모듈은 상대 가져오기, 자료 구조 Option·Prompt·RouteAssessment 와 _norm 은 서버의 interpreter 모듈).
서버의 자료 구조는 원본과 필드·메서드가 같고 받아쓰기 고침용 필드(shown·fixes·display_fixes, 기본값 있음)만 추가됨.
원본과 다른 점이 가져오기 경로뿐인지는 test_som_mode.py 가 확인.
"""
SOURCE_BRANCH = "origin/experiment/openai-stt-jev"
SOURCE_REV = "9f628ec8"
