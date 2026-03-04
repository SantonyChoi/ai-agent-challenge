import dotenv

dotenv.load_dotenv()
import asyncio
import os
from openai import OpenAI
import streamlit as st
from agents import Agent, Runner, WebSearchTool, FileSearchTool, SQLiteSession

client = OpenAI()
vector_store_id = os.getenv("VECTOR_STORE_ID")

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        instructions=(
            "You are an encouraging life coach with access to the user's personal goals.\n"
            "- Search the user's goals document when they ask about goals or progress.\n"
            "- Search the web when you need current tips or advice.\n"
            "- You may use both tools in one turn if needed.\n"
            "- Respond in the same language the user uses."
        ),
        tools=[
            FileSearchTool(max_num_results=3, vector_store_ids=[vector_store_id]),
            WebSearchTool(),
        ],
    )

if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession("life-coach", "day8/life-coach-memory.db")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

agent = st.session_state["agent"]
session = st.session_state["session"]


async def run_agent(message, placeholder, status_container):
    stream = Runner.run_streamed(agent, message, session=session)
    labels = []
    full_response = ""
    async for event in stream.stream_events():
        if event.type == "raw_response_event":
            data = event.data
            if data.type == "response.output_text.delta":
                full_response += data.delta
                placeholder.markdown(full_response + "▌")
            elif data.type == "response.file_search_call.searching":
                labels.append("문서 검색 완료")
                status_container.markdown("\n\n".join(f"`[{l}]`" for l in labels))
            elif data.type == "response.web_search_call.searching":
                labels.append("웹 검색 완료")
                status_container.markdown("\n\n".join(f"`[{l}]`" for l in labels))
    placeholder.markdown(full_response)
    return "\n\n".join(f"`[{l}]`" for l in labels) + ("\n\n" if labels else "") + full_response


st.title("Life Coach Agent")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("무엇이든 물어보세요!", accept_file=True, file_type=["txt", "pdf"])

if prompt:
    for file in prompt.files:
        with st.chat_message("ai"):
            with st.status("파일 업로드 중...") as status:
                uploaded = client.files.create(file=(file.name, file.getvalue()), purpose="user_data")
                client.vector_stores.files.create(vector_store_id=vector_store_id, file_id=uploaded.id)
                status.update(label=f"'{file.name}' 업로드 완료", state="complete")
        st.session_state["messages"].append({"role": "ai", "content": f"`[파일 업로드: {file.name}]`"})

    if prompt.text:
        st.session_state["messages"].append({"role": "human", "content": prompt.text})
        with st.chat_message("human"):
            st.write(prompt.text)
        with st.chat_message("ai"):
            status_container = st.empty()
            placeholder = st.empty()
        display = asyncio.run(run_agent(prompt.text, placeholder, status_container))
        st.session_state["messages"].append({"role": "ai", "content": display})

with st.sidebar:
    if st.button("Reset memory"):
        asyncio.run(session.clear_session())
        st.session_state["messages"] = []
        st.rerun()
    st.write(asyncio.run(session.get_items()))
