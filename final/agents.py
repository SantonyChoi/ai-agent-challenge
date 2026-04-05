from langchain_openai import ChatOpenAI
from state import WritingState

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
llm_strict = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================================
# 과제 출제 에이전트
# =============================================================================

def assignment_agent(state: WritingState) -> dict:
    """영어 작문 과제를 하나 생성한다."""
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an English writing teacher for Korean learners.\n"
                    "Generate ONE short English writing assignment.\n"
                    "The assignment should be specific and achievable in 3-8 sentences.\n"
                    "Include the topic, a brief context, and what the student should write about.\n"
                    "Vary the types: opinion essay, email, diary entry, story continuation, description, etc.\n"
                    "Respond in Korean for the instructions, but the writing task itself should require English output.\n"
                    "Do NOT include any sample answer."
                ),
            },
            {"role": "user", "content": "과제를 하나 주세요."},
        ]
    )
    return {
        "assignment": response.content,
        "phase": "assignment_given",
        "evaluations": [],
        "final_feedback": "",
    }


# =============================================================================
# 평가 에이전트들 (병렬 실행)
# =============================================================================

def grammar_agent(state: WritingState) -> dict:
    """문법 정확성을 평가한다."""
    response = llm_strict.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict English grammar evaluator.\n"
                    "Evaluate the student's writing for grammar accuracy.\n\n"
                    "Respond in Korean. Use this format:\n"
                    "## 문법 평가\n"
                    "**점수: X/10**\n\n"
                    "### 발견된 오류\n"
                    "- (each error with correction)\n\n"
                    "### 잘한 점\n"
                    "- (positive grammar usage)\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[과제]\n{state['assignment']}\n\n"
                    f"[학생 작문]\n{state['submission']}"
                ),
            },
        ]
    )
    return {
        "evaluations": [{"agent": "grammar", "feedback": response.content}],
    }


def context_agent(state: WritingState) -> dict:
    """맥락과 논리 흐름을 평가한다."""
    response = llm_strict.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are an English writing evaluator focused on context, coherence, and logical flow.\n"
                    "Evaluate whether the student's writing:\n"
                    "- Addresses the assignment properly\n"
                    "- Has a clear structure (intro, body, conclusion)\n"
                    "- Maintains logical flow between sentences\n"
                    "- Stays on topic\n\n"
                    "Respond in Korean. Use this format:\n"
                    "## 맥락/논리 평가\n"
                    "**점수: X/10**\n\n"
                    "### 구조 분석\n"
                    "- (structure observations)\n\n"
                    "### 개선 제안\n"
                    "- (suggestions for better flow)\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[과제]\n{state['assignment']}\n\n"
                    f"[학생 작문]\n{state['submission']}"
                ),
            },
        ]
    )
    return {
        "evaluations": [{"agent": "context", "feedback": response.content}],
    }


def native_expression_agent(state: WritingState) -> dict:
    """원어민스러운 표현을 평가한다."""
    response = llm_strict.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a native English speaker evaluating a Korean student's writing.\n"
                    "Focus on naturalness and native-like expression.\n"
                    "Evaluate:\n"
                    "- Word choice (natural vs. awkward/translated from Korean)\n"
                    "- Idiomatic expressions (or lack thereof)\n"
                    "- Sentence variety and rhythm\n"
                    "- Collocations\n\n"
                    "Respond in Korean. Use this format:\n"
                    "## 원어민 표현 평가\n"
                    "**점수: X/10**\n\n"
                    "### 어색한 표현 -> 자연스러운 대안\n"
                    "- \"student wrote\" -> \"better alternative\" (explanation)\n\n"
                    "### 잘 쓴 표현\n"
                    "- (natural expressions the student used well)\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[과제]\n{state['assignment']}\n\n"
                    f"[학생 작문]\n{state['submission']}"
                ),
            },
        ]
    )
    return {
        "evaluations": [{"agent": "native_expression", "feedback": response.content}],
    }


# =============================================================================
# 종합 에이전트
# =============================================================================

def comprehensive_agent(state: WritingState) -> dict:
    """모든 평가를 종합하여 최종 피드백을 생성한다."""
    eval_texts = ""
    for ev in state["evaluations"]:
        eval_texts += f"\n---\n[{ev['agent']}]\n{ev['feedback']}\n"

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a head English teacher who synthesizes feedback from three specialist evaluators:\n"
                    "1. Grammar evaluator\n"
                    "2. Context/coherence evaluator\n"
                    "3. Native expression evaluator\n\n"
                    "Your job:\n"
                    "- Summarize the overall assessment\n"
                    "- Give a total score (average of three scores, out of 10)\n"
                    "- Highlight the top 2-3 things to improve\n"
                    "- Highlight what the student did well\n"
                    "- Provide a rewritten 'model answer' that fixes all issues while keeping the student's intent\n\n"
                    "Respond in Korean. Use this format:\n"
                    "# 종합 평가\n\n"
                    "**종합 점수: X/10**\n\n"
                    "## 총평\n"
                    "(brief overall comment)\n\n"
                    "## 주요 개선 포인트\n"
                    "1. ...\n"
                    "2. ...\n\n"
                    "## 잘한 점\n"
                    "- ...\n\n"
                    "## 교정된 모범 답안\n"
                    "(rewritten version of the student's text)\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[과제]\n{state['assignment']}\n\n"
                    f"[학생 작문]\n{state['submission']}\n\n"
                    f"[전문가 평가들]\n{eval_texts}"
                ),
            },
        ]
    )
    return {
        "final_feedback": response.content,
        "phase": "done",
    }
