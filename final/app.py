import dotenv

dotenv.load_dotenv()

import streamlit as st
from graph import build_assignment_graph, build_evaluation_graph

st.set_page_config(page_title="English Writing Coach", layout="wide")
st.title("English Writing Coach")
st.caption("영어 작문 과제를 받고, AI 전문가 3명이 동시에 평가해드립니다.")

# --- 사이드바: 사용법 안내 ---
with st.sidebar:
    st.header("사용 방법")
    st.markdown(
        "1. **과제를 주세요** 버튼을 눌러 작문 과제를 받습니다.\n"
        "2. 영어로 작문을 작성한 뒤 **제출** 버튼을 누릅니다.\n"
        "3. 3명의 AI 전문가가 동시에 평가합니다:\n"
        "   - 문법 에이전트\n"
        "   - 맥락/논리 에이전트\n"
        "   - 원어민 표현 에이전트\n"
        "4. 종합 에이전트가 최종 피드백과 모범 답안을 제공합니다."
    )
    st.divider()
    st.markdown(
        "**Powered by** LangGraph + GPT-4o-mini\n\n"
        "3개 평가 에이전트가 병렬로 실행되어 빠르고 다각적인 피드백을 제공합니다."
    )

# --- Session State 초기화 ---
if "phase" not in st.session_state:
    st.session_state["phase"] = "idle"  # idle | assignment_given | evaluating | done
if "assignment" not in st.session_state:
    st.session_state["assignment"] = ""
if "submission" not in st.session_state:
    st.session_state["submission"] = ""
if "evaluations" not in st.session_state:
    st.session_state["evaluations"] = []
if "final_feedback" not in st.session_state:
    st.session_state["final_feedback"] = ""


# =============================================================================
# Phase 1: 과제 요청
# =============================================================================

if st.session_state["phase"] == "idle":
    st.markdown("**'과제를 주세요' 버튼을 눌러 영어 작문 과제를 받아보세요.**")
    if st.button("과제를 주세요", type="primary"):
        with st.spinner("과제를 생성하고 있습니다..."):
            try:
                graph = build_assignment_graph()
                result = graph.invoke(
                    {
                        "phase": "idle",
                        "assignment": "",
                        "submission": "",
                        "evaluations": [],
                        "final_feedback": "",
                    }
                )
                st.session_state["assignment"] = result["assignment"]
                st.session_state["phase"] = "assignment_given"
                st.rerun()
            except Exception as e:
                st.error(f"과제 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n`{e}`")


# =============================================================================
# Phase 2: 과제 표시 + 작문 제출
# =============================================================================

if st.session_state["phase"] == "assignment_given":
    st.subheader("과제")
    st.info(st.session_state["assignment"])

    submission = st.text_area(
        "여기에 영어로 작문을 작성하세요:",
        height=250,
        placeholder="Write your answer in English here...",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("제출", type="primary")
    with col2:
        new_assignment = st.button("다른 과제 받기")

    if submit and submission.strip():
        st.session_state["submission"] = submission
        st.session_state["phase"] = "evaluating"
        st.rerun()
    elif submit:
        st.warning("작문을 입력해주세요.")

    if new_assignment:
        st.session_state["phase"] = "idle"
        st.rerun()


# =============================================================================
# Phase 3: 평가 진행
# =============================================================================

if st.session_state["phase"] == "evaluating":
    st.subheader("과제")
    st.info(st.session_state["assignment"])

    st.subheader("제출한 작문")
    st.markdown(st.session_state["submission"])

    st.divider()

    try:
        with st.spinner("3명의 전문 평가 에이전트가 동시에 작문을 분석하고 있습니다..."):
            progress = st.progress(0, text="평가 시작...")

            progress.progress(10, text="문법 / 맥락 / 원어민 표현 에이전트 병렬 평가 중...")

            graph = build_evaluation_graph()
            result = graph.invoke(
                {
                    "phase": "evaluating",
                    "assignment": st.session_state["assignment"],
                    "submission": st.session_state["submission"],
                    "evaluations": [],
                    "final_feedback": "",
                }
            )

            progress.progress(90, text="종합 평가 작성 중...")

            st.session_state["evaluations"] = result["evaluations"]
            st.session_state["final_feedback"] = result["final_feedback"]
            st.session_state["phase"] = "done"

            progress.progress(100, text="평가 완료!")

        st.rerun()
    except Exception as e:
        st.error(f"평가 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.\n\n`{e}`")
        if st.button("처음으로 돌아가기"):
            st.session_state["phase"] = "idle"
            st.rerun()


# =============================================================================
# Phase 4: 결과 표시
# =============================================================================

if st.session_state["phase"] == "done":
    st.subheader("과제")
    st.info(st.session_state["assignment"])

    st.subheader("제출한 작문")
    st.markdown(st.session_state["submission"])

    st.divider()

    # 개별 에이전트 평가 (펼쳐볼 수 있게)
    st.subheader("전문가별 상세 평가")

    agent_labels = {
        "grammar": "문법 에이전트",
        "context": "맥락/논리 에이전트",
        "native_expression": "원어민 표현 에이전트",
    }

    cols = st.columns(3)
    for i, ev in enumerate(st.session_state["evaluations"]):
        label = agent_labels.get(ev["agent"], ev["agent"])
        with cols[i % 3]:
            with st.expander(label, expanded=True):
                st.markdown(ev["feedback"])

    st.divider()

    # 종합 평가
    st.subheader("종합 평가")
    st.markdown(st.session_state["final_feedback"])

    st.divider()

    if st.button("새로운 과제 받기", type="primary"):
        st.session_state["phase"] = "idle"
        st.session_state["assignment"] = ""
        st.session_state["submission"] = ""
        st.session_state["evaluations"] = []
        st.session_state["final_feedback"] = ""
        st.rerun()
