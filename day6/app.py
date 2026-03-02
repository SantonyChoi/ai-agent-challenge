import dotenv

dotenv.load_dotenv()
import asyncio
import streamlit as st
from agents import Agent, Runner, WebSearchTool, SQLiteSession

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        instructions=(
            "You are an encouraging life coach.\n"
            "- Motivate the user and give practical self-improvement tips.\n"
            "- When the user asks about habits, motivation, or personal growth, "
            "use web search to find relevant advice.\n"
            "- Keep responses warm, supportive, and actionable.\n"
            "- Respond in the same language the user uses."
        ),
        tools=[WebSearchTool()],
    )
agent = st.session_state["agent"]

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "life-coach-history",
        "day6/life-coach-memory.db",
    )
session = st.session_state["session"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []


async def run_agent(message, placeholder, search_container):
    stream = Runner.run_streamed(
        agent,
        message,
        session=session,
    )
    searches = []
    full_response = ""
    async for event in stream.stream_events():
        if event.type == "raw_response_event":
            data = event.data
            if data.type == "response.output_text.delta":
                full_response += data.delta
                placeholder.markdown(build_display(searches, full_response + "▌"))
            elif data.type == "response.web_search_call.searching":
                search_container.info("웹 검색 중...")
            elif data.type == "response.web_search_call.completed":
                search_container.empty()
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                raw = item.raw_item
                if raw.type == "web_search_call" and getattr(raw, "status", "") == "completed":
                    action = getattr(raw, "action", None)
                    query = getattr(action, "query", "") if action else ""
                    if query and query not in searches:
                        searches.append(query)
    display = build_display(searches, full_response)
    placeholder.markdown(display)
    return display


def build_display(searches, text):
    parts = []
    for q in searches:
        parts.append(f'`[웹 검색: "{q}"]`')
    parts.append(text)
    return "\n\n".join(parts)


for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("무엇이든 물어보세요!")

if prompt:
    st.session_state["messages"].append({"role": "human", "content": prompt})
    with st.chat_message("human"):
        st.write(prompt)

    with st.chat_message("ai"):
        search_container = st.empty()
        placeholder = st.empty()
    display = asyncio.run(run_agent(prompt, placeholder, search_container))

    st.session_state["messages"].append({"role": "ai", "content": display})


with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())
        st.session_state["messages"] = []
        st.rerun()
    st.write(asyncio.run(session.get_items()))
