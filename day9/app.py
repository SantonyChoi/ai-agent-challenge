import dotenv

dotenv.load_dotenv()
import asyncio
import base64
import os
from openai import OpenAI
import streamlit as st
from agents import Agent, Runner, WebSearchTool, FileSearchTool, ImageGenerationTool, SQLiteSession

client = OpenAI()
vector_store_id = os.getenv("VECTOR_STORE_ID")


class CleanSession:
    """Wraps SQLiteSession and strips 'action' fields that cause API errors."""

    def __init__(self, *args, **kwargs):
        self._session = SQLiteSession(*args, **kwargs)

    async def get_items(self):
        items = await self._session.get_items()
        cleaned = []
        for item in items:
            if isinstance(item, dict) and "action" in item:
                item = {k: v for k, v in item.items() if k != "action"}
            cleaned.append(item)
        return cleaned

    async def add_items(self, items):
        return await self._session.add_items(items)

    async def clear_session(self):
        return await self._session.clear_session()

if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="Life Coach",
        model="gpt-4o-mini",
        instructions=(
            "You are an encouraging life coach with access to the user's personal goals.\n"
            "- Search the user's goals document when they ask about goals or progress.\n"
            "- Search the web when you need current tips or advice.\n"
            "- When the user asks for an image, vision board, poster, or any visual content, "
            "IMMEDIATELY generate it using the image generation tool. Do NOT ask clarifying questions first. "
            "Use the user's goals from file search to create a meaningful prompt and generate right away.\n"
            "- You may use multiple tools in one turn if needed.\n"
            "- Respond in the same language the user uses."
        ),
        tools=[
            FileSearchTool(max_num_results=3, vector_store_ids=[vector_store_id]),
            WebSearchTool(),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "high",
                    "output_format": "jpeg",
                    "partial_images": 1,
                }
            ),
        ],
    )

if "session" not in st.session_state:
    st.session_state["session"] = CleanSession("life-coach", "day9/life-coach-memory.db")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

agent = st.session_state["agent"]
session = st.session_state["session"]


async def run_agent(message, placeholder, status_container, image_placeholder):
    stream = Runner.run_streamed(agent, message, session=session)
    labels = []
    full_response = ""
    generated_image = None
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
            elif data.type in ("response.image_generation_call.generating", "response.image_generation_call.in_progress"):
                if "이미지 생성 중..." not in labels:
                    labels.append("이미지 생성 중...")
                    status_container.markdown("\n\n".join(f"`[{l}]`" for l in labels))
            elif data.type == "response.image_generation_call.partial_image":
                generated_image = base64.b64decode(data.partial_image_b64)
                image_placeholder.image(generated_image)
    # Finalize labels
    labels = [l.replace("이미지 생성 중...", "이미지 생성 완료") for l in labels]
    status_container.markdown("\n\n".join(f"`[{l}]`" for l in labels))
    placeholder.markdown(full_response)
    label_text = "\n\n".join(f"`[{l}]`" for l in labels) + ("\n\n" if labels else "")
    return label_text + full_response, generated_image


st.title("Life Coach Agent")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image"):
            st.image(msg["image"])

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
            image_placeholder = st.empty()
            placeholder = st.empty()
        display, image = asyncio.run(run_agent(prompt.text, placeholder, status_container, image_placeholder))
        msg = {"role": "ai", "content": display}
        if image:
            msg["image"] = image
        st.session_state["messages"].append(msg)
        st.rerun()

with st.sidebar:
    if st.button("Reset memory"):
        asyncio.run(session.clear_session())
        st.session_state["messages"] = []
        st.rerun()
    st.write(asyncio.run(session.get_items()))
