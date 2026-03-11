import dotenv

dotenv.load_dotenv()
import asyncio
import streamlit as st
from agents import (
    Agent,
    Runner,
    handoff,
    RunContextWrapper,
    input_guardrail,
    output_guardrail,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters
from pydantic import BaseModel


# --- Models ---


class HandoffData(BaseModel):
    reason: str


class InputGuardrailOutput(BaseModel):
    is_off_topic: bool
    is_inappropriate: bool
    reason: str


class OutputGuardrailOutput(BaseModel):
    is_unprofessional: bool
    leaks_internal_info: bool
    reason: str


# --- Input Guardrail ---

input_guardrail_agent = Agent(
    name="Input Guardrail Agent",
    model="gpt-4o-mini",
    instructions="""
    Analyze the user's message and determine:
    1. is_off_topic: True if the message is NOT related to a restaurant
       (e.g. menu, food, orders, reservations, complaints about restaurant service).
       Simple greetings and small talk are allowed (not off-topic).
    2. is_inappropriate: True if the message contains profanity, hate speech,
       threats, or sexually explicit content.
    Provide a brief reason for your decision.
    """,
    output_type=InputGuardrailOutput,
)


@input_guardrail
async def restaurant_input_guardrail(
    wrapper: RunContextWrapper,
    agent: Agent,
    input: str,
):
    result = await Runner.run(
        input_guardrail_agent,
        input,
        context=wrapper.context,
    )
    validation = result.final_output
    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=validation.is_off_topic or validation.is_inappropriate,
    )


# --- Output Guardrail ---

output_guardrail_agent = Agent(
    name="Output Guardrail Agent",
    model="gpt-4o-mini",
    instructions="""
    Analyze the assistant's response and determine:
    1. is_unprofessional: True if the response is rude, dismissive, sarcastic,
       or otherwise unprofessional for a restaurant service context.
    2. leaks_internal_info: True if the response reveals internal business details
       such as profit margins, supplier names, employee salaries, internal policies,
       cost prices, or any information that should not be shared with customers.
    Provide a brief reason for your decision.
    """,
    output_type=OutputGuardrailOutput,
)


@output_guardrail
async def restaurant_output_guardrail(
    wrapper: RunContextWrapper,
    agent: Agent,
    output: str,
):
    result = await Runner.run(
        output_guardrail_agent,
        output,
        context=wrapper.context,
    )
    validation = result.final_output
    return GuardrailFunctionOutput(
        output_info=validation,
        tripwire_triggered=validation.is_unprofessional or validation.leaks_internal_info,
    )


# --- Specialist Agents ---

menu_agent = Agent(
    name="Menu Agent",
    model="gpt-4o-mini",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n\n"
        "You are a menu specialist at a Korean-Italian fusion restaurant.\n"
        "- Answer questions about menu items, ingredients, and allergens.\n"
        "- Suggest dishes based on dietary preferences (vegetarian, vegan, gluten-free, etc.).\n"
        "- Describe dishes in an appetizing way.\n"
        "- Never reveal supplier names, cost prices, or profit margins.\n"
        "- Respond in the same language the user uses."
    ),
    output_guardrails=[restaurant_output_guardrail],
)

order_agent = Agent(
    name="Order Agent",
    model="gpt-4o-mini",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n\n"
        "You are an order specialist at a restaurant.\n"
        "- Take orders from customers, confirm items and quantities.\n"
        "- Suggest popular add-ons or drinks.\n"
        "- Summarize the order before confirming.\n"
        "- Never reveal internal pricing structures or supplier information.\n"
        "- Respond in the same language the user uses."
    ),
    output_guardrails=[restaurant_output_guardrail],
)

reservation_agent = Agent(
    name="Reservation Agent",
    model="gpt-4o-mini",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n\n"
        "You are a reservation specialist at a restaurant.\n"
        "- Help customers book tables by asking for date, time, and party size.\n"
        "- Confirm reservation details before finalizing.\n"
        "- Handle reservation changes and cancellations.\n"
        "- Never reveal internal capacity management or staffing details.\n"
        "- Respond in the same language the user uses."
    ),
    output_guardrails=[restaurant_output_guardrail],
)

complaints_agent = Agent(
    name="Complaints Agent",
    model="gpt-4o-mini",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n\n"
        "You are a customer complaints specialist at a Korean-Italian fusion restaurant.\n"
        "Your role is to handle unhappy customers with care and professionalism.\n\n"
        "PROCESS:\n"
        "1. Acknowledge and empathize with the customer's complaint sincerely.\n"
        "2. Apologize for the inconvenience.\n"
        "3. Offer appropriate solutions based on severity:\n"
        "   - Minor issues (wait time, small mistakes): Offer a complimentary dessert or drink.\n"
        "   - Moderate issues (wrong order, cold food): Offer a discount (10-20%) or a replacement.\n"
        "   - Serious issues (food safety, allergic reactions, rude staff): Offer a full refund "
        "and arrange a manager callback. Express that this will be escalated immediately.\n\n"
        "GUIDELINES:\n"
        "- Always remain calm, empathetic, and professional.\n"
        "- Never argue with the customer or make excuses.\n"
        "- Never reveal internal policies, employee names, or operational details.\n"
        "- If the customer remains unsatisfied after offering solutions, escalate by "
        "promising a manager will contact them within 24 hours.\n"
        "- Respond in the same language the user uses."
    ),
    output_guardrails=[restaurant_output_guardrail],
)


# --- Handoff Setup ---


def on_handoff(wrapper: RunContextWrapper, data: HandoffData):
    st.session_state.setdefault("handoff_logs", []).append(data.reason)


def make_handoff(agent):
    return handoff(
        agent=agent,
        on_handoff=on_handoff,
        input_type=HandoffData,
        input_filter=handoff_filters.remove_all_tools,
    )


# --- Triage Agent ---

triage_agent = Agent(
    name="Triage Agent",
    model="gpt-4o-mini",
    instructions=(
        f"{RECOMMENDED_PROMPT_PREFIX}\n\n"
        "You are the front desk of a restaurant.\n"
        "- Greet the customer warmly.\n"
        "- Determine what they need and hand off to the right specialist:\n"
        "  * Menu questions (ingredients, allergens, recommendations) -> Menu Agent\n"
        "  * Placing an order -> Order Agent\n"
        "  * Making/changing a reservation -> Reservation Agent\n"
        "  * Complaints, dissatisfaction, negative experiences -> Complaints Agent\n"
        "- Always hand off quickly. Do NOT answer specialist questions yourself.\n"
        "- Respond in the same language the user uses."
    ),
    input_guardrails=[restaurant_input_guardrail],
    handoffs=[
        make_handoff(menu_agent),
        make_handoff(order_agent),
        make_handoff(reservation_agent),
        make_handoff(complaints_agent),
    ],
)

# Allow specialists to hand off back to triage or to each other
all_agents = [menu_agent, order_agent, reservation_agent, complaints_agent]
for agent in all_agents:
    agent.handoffs = [
        make_handoff(a)
        for a in [triage_agent] + all_agents
        if a != agent
    ]


# --- Streamlit App ---

st.title("Restaurant Bot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "handoff_logs" not in st.session_state:
    st.session_state["handoff_logs"] = []


async def run_agent(message, placeholder, status_container):
    st.session_state["handoff_logs"] = []
    try:
        result = Runner.run_streamed(triage_agent, message)
        full_response = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                data = event.data
                if data.type == "response.output_text.delta":
                    full_response += data.delta
                    placeholder.markdown(full_response + "▌")
            elif event.type == "agent_updated_stream_event":
                agent_name = event.new_agent.name
                status_container.markdown(f"`[{agent_name}에게 연결 중...]`")
        placeholder.markdown(full_response)
        handoff_labels = "\n\n".join(
            f"`[{log}]`" for log in st.session_state["handoff_logs"]
        )
        return (handoff_labels + "\n\n" if handoff_labels else "") + full_response

    except InputGuardrailTripwireTriggered:
        msg = "죄송합니다. 레스토랑과 관련된 질문만 도와드릴 수 있습니다."
        placeholder.markdown(msg)
        return msg

    except OutputGuardrailTripwireTriggered:
        msg = "죄송합니다. 해당 응답을 제공할 수 없습니다. 다른 질문을 해주세요."
        placeholder.markdown(msg)
        return msg


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("무엇을 도와드릴까요?")

if prompt:
    st.session_state["messages"].append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.write(prompt)
    with st.chat_message("ai"):
        status_container = st.empty()
        placeholder = st.empty()
    display = asyncio.run(run_agent(prompt, placeholder, status_container))
    st.session_state["messages"].append({"role": "ai", "content": display})
    st.rerun()
