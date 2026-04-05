# English Writing Coach

AI 전문가 3명이 동시에 영어 작문을 평가해주는 교육용 에이전트 시스템입니다.

## 주요 기능

- **과제 출제**: 다양한 유형의 영어 작문 과제를 자동 생성 (의견문, 이메일, 일기, 묘사 등)
- **병렬 평가**: 3개의 전문 에이전트가 동시에 작문을 분석
  - **문법 에이전트** - 문법 오류 검출 및 교정
  - **맥락/논리 에이전트** - 구조, 논리 흐름, 주제 적합성 평가
  - **원어민 표현 에이전트** - 자연스러운 표현, 어색한 직역 감지, 대안 제시
- **종합 평가**: 3개 평가를 종합하여 총점, 개선 포인트, 교정된 모범 답안 제공

## 기술 스택

- **LangGraph** - 에이전트 워크플로우 관리 (병렬 실행 + 순차 합산)
- **LangChain + GPT-4o-mini** - LLM 기반 에이전트
- **Streamlit** - 웹 UI

## 아키텍처

```
[과제 출제 그래프]
START -> AssignmentAgent -> END

[평가 그래프]
START ─┬─> GrammarAgent ──────────┐
       ├─> ContextAgent ──────────┼─> ComprehensiveAgent -> END
       └─> NativeExpressionAgent──┘
           (병렬 실행)                (종합)
```

`WritingState`의 `Annotated[list[dict], add]`를 활용하여 병렬 에이전트들의 결과가 자동으로 합산됩니다.

## 실행 방법

```bash
cd final

# 환경변수 설정
echo 'OPENAI_API_KEY=sk-...' > .env

# 실행
uv run streamlit run app.py
```

## 배포

Streamlit Cloud에 배포되어 있습니다. Secrets에 `OPENAI_API_KEY`를 설정하면 바로 동작합니다.
