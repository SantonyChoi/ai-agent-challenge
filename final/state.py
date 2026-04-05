from typing import TypedDict, Annotated
from operator import add


class WritingState(TypedDict):
    # 현재 단계: "idle" | "assignment_given" | "evaluating" | "done"
    phase: str
    # 과제 주제/지시문
    assignment: str
    # 유저가 제출한 영어 작문
    submission: str
    # 각 에이전트 평가 결과 (병렬 실행 후 합산)
    evaluations: Annotated[list[dict], add]
    # 종합 에이전트의 최종 피드백
    final_feedback: str
