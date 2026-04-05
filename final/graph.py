from langgraph.graph import StateGraph, START, END
from state import WritingState
from agents import (
    assignment_agent,
    grammar_agent,
    context_agent,
    native_expression_agent,
    comprehensive_agent,
)


def build_assignment_graph() -> StateGraph:
    """과제 출제 그래프: 과제 생성만 수행."""
    g = StateGraph(WritingState)
    g.add_node("assign", assignment_agent)
    g.add_edge(START, "assign")
    g.add_edge("assign", END)
    return g.compile()


def build_evaluation_graph() -> StateGraph:
    """평가 그래프: 3개 에이전트 병렬 실행 -> 종합 에이전트."""
    g = StateGraph(WritingState)

    # 평가 에이전트 노드 (병렬)
    g.add_node("grammar", grammar_agent)
    g.add_node("context", context_agent)
    g.add_node("native_expression", native_expression_agent)

    # 종합 에이전트 노드
    g.add_node("comprehensive", comprehensive_agent)

    # START -> 3개 병렬 실행
    g.add_edge(START, "grammar")
    g.add_edge(START, "context")
    g.add_edge(START, "native_expression")

    # 3개 모두 완료 -> 종합 에이전트
    g.add_edge("grammar", "comprehensive")
    g.add_edge("context", "comprehensive")
    g.add_edge("native_expression", "comprehensive")

    # 종합 에이전트 -> 끝
    g.add_edge("comprehensive", END)

    return g.compile()
