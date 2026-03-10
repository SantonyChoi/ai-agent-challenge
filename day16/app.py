import dotenv

dotenv.load_dotenv()
import asyncio
import streamlit as st
from agents import Agent, Runner, handoff, RunContextWrapper
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from agents.extensions import handoff_filters
from pydantic import BaseModel


class HandoffData(BaseModel):
    reason: str


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
        "- Respond in the same language the user uses."
    ),
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
        "- Respond in the same language the user uses."
    ),
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
        "- Respond in the same language the user uses."
    ),
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
        "- Always hand off quickly. Do NOT answer specialist questions yourself.\n"
        "- Respond in the same language the user uses."
    ),
    handoffs=[
        make_handoff(menu_agent),
        make_handoff(order_agent),
        make_handoff(reservation_agent),
    ],
)

# Allow specialists to hand off back to triage or to each other
for agent in [menu_agent, order_agent, reservation_agent]:
    agent.handoffs = [
        make_handoff(a) for a in [triage_agent, menu_agent, order_agent, reservation_agent] if a != agent
    ]


# --- Streamlit App ---

st.title("Restaurant Bot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "handoff_logs" not in st.session_state:
    st.session_state["handoff_logs"] = []


async def run_agent(message, placeholder, status_container):
    st.session_state["handoff_logs"] = []
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
    handoff_labels = "\n\n".join(f"`[{log}]`" for log in st.session_state["handoff_logs"])
    return (handoff_labels + "\n\n" if handoff_labels else "") + full_response


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
