import streamlit as st
import base64
from io import BytesIO
from PIL import Image
import asyncio
import json
from typing import List, Dict

# external utils
from llm_utils import get_llm, query_llm
from image_utils import generate_image

# helpers
def pil_to_base64(img: Image.Image, max_dim: int = 512) -> str:
    """Resize (if needed) → PNG → base64."""
    if max(img.size) > max_dim:
        img = img.copy()
        img.thumbnail((max_dim, max_dim))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_pil(data: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(data)))

# Session‑state helpers
def _init_state():
    if "sessions" not in st.session_state:
        st.session_state.sessions = {
            "default": {"llm": "openai/gpt-oss-120b", "history": []}
        }
    if "current_session" not in st.session_state:
        st.session_state.current_session = "default"

def add_msg(role: str, msg_type: str, content: str):
    """Append a message to the active session."""
    st.session_state.sessions[st.session_state.current_session]["history"].append(
        {"role": role, "type": msg_type, "content": content}
    )
    # optional trimming
    MAX_TURNS = 30
    hist = st.session_state.sessions[st.session_state.current_session]["history"]
    if len(hist) > MAX_TURNS:
        st.session_state.sessions[st.session_state.current_session]["history"] = hist[-MAX_TURNS:]

def get_history() -> List[Dict]:
    return st.session_state.sessions[st.session_state.current_session]["history"]

# Cached LLM client
@st.cache_resource(show_spinner=False)
def get_cached_llm(model_name: str):
    return get_llm(model_name)

# Async image generation wrapper
async def async_generate_image(prompt: str) -> Image.Image:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_image, prompt)

# UI – page config & sidebar
st.set_page_config(page_title="AI Multi‑LLM Chatbot", layout="wide")
st.title(" Multi‑LLM Chatbot + Image Generation")

_init_state()   # ensure session_state exists

# Sidebar 
st.sidebar.header("⚙️ Settings")

# Session selector
session_names = list(st.session_state.sessions.keys())
selected = st.sidebar.selectbox("Choose session:", session_names,
                               index=session_names.index(st.session_state.current_session))
st.session_state.current_session = selected

# LLM selector
llm_options = ["gemma2-9b-it", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"]
current_llm = st.sidebar.selectbox(
    "LLM for this session:",
    llm_options,
    index=llm_options.index(st.session_state.sessions[st.session_state.current_session]["llm"])
    if st.session_state.sessions[st.session_state.current_session]["llm"] in llm_options
    else 0,
)
st.session_state.sessions[st.session_state.current_session]["llm"] = current_llm

# Image mode & size
image_mode = st.sidebar.checkbox("Enable Image Mode", value=False)
image_width = st.sidebar.slider("Image Width (px)", 128, 1024, 512, 16)
use_container_width = st.sidebar.checkbox("Use Full Container Width", value=False)

# New session UI
new_name = st.sidebar.text_input("Create new session:", key="new_session_input").strip()
if st.sidebar.button("➕ Add Session"):
    if not new_name:
        base, i = "Session", 1
        while f"{base} {i}" in st.session_state.sessions:
            i += 1
        new_name = f"{base} {i}"
    if new_name not in st.session_state.sessions:
        st.session_state.sessions[new_name] = {"llm": current_llm, "history": []}
        st.session_state.current_session = new_name
        st.sidebar.success(f"Created session: {new_name}")
        st.rerun()
    else:
        st.sidebar.warning("Session name already exists.")

# Delete / clear UI
if st.sidebar.button("🗑️ Delete Current Session"):
    if st.session_state.current_session != "default":
        del st.session_state.sessions[st.session_state.current_session]
        st.session_state.current_session = "default"
        st.sidebar.success("Deleted – switched to default.")
        st.rerun()
    else:
        st.sidebar.warning("Default session cannot be deleted.")

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.sessions[st.session_state.current_session]["history"] = []
    st.sidebar.success("Chat cleared.")
    st.rerun()

# Chat rendering
st.subheader(f"💬 {st.session_state.current_session}")

for msg in get_history():
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        else:  # image
            st.image(
                base64_to_pil(msg["content"]),
                caption="Generated image",
                width=None if use_container_width else image_width,
                use_container_width=use_container_width,
            )

# Input handling
def is_image_request(txt: str, mode_enabled: bool) -> bool:
    if mode_enabled:
        return True
    txt = txt.strip().lower()
    return txt.startswith(("!img ", "/img ", "image:", "generate image:"))

placeholder = "Type your message…" if not image_mode else "Describe the image you want…"
if prompt := st.chat_input(placeholder):
    # Image branch
    if is_image_request(prompt, image_mode):
        # clean up the command prefix if we’re not in forced image mode
        clean_prompt = (
            prompt.replace("generate image", "", 1)
                   .replace("image", "", 1)
                   .strip()
            if not image_mode else prompt
        )

        # user bubble
        with st.chat_message("user"):
            st.markdown(f" {clean_prompt}")
        add_msg("user", "text", f"[Image request] {clean_prompt}")

        # assistant bubble (async generation)
        with st.chat_message("assistant"):
            with st.spinner("Generating image…"):
                try:
                    img = asyncio.run(async_generate_image(clean_prompt))
                    st.image(
                        img,
                        caption=clean_prompt,
                        width=None if use_container_width else image_width,
                        use_container_width=use_container_width,
                    )
                    add_msg("assistant", "image", pil_to_base64(img))
                except Exception as e:
                    st.error(f"❌ Image generation failed: {e}")

    # Text chat branch
    else:
        # user bubble
        add_msg("user", "text", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # LLM response
        llm = get_cached_llm(current_llm)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = query_llm(llm, prompt)
                except Exception as e:
                    response = f"⚠️ Error: {e}"
                st.markdown(response)
        add_msg("assistant", "text", response)

# Export / import util
def export_current_session() -> str:
    """Return a JSON string of the active session (history + LLM)."""
    sess = st.session_state.sessions[st.session_state.current_session]
    return json.dumps(sess, indent=2)

def import_session_from_json(json_str: str, name: str):
    """Create a new session from a JSON dump."""
    data = json.loads(json_str)
    if name in st.session_state.sessions:
        raise ValueError(f"Session '{name}' already exists")
    st.session_state.sessions[name] = data

with st.expander("💾 Export / Import"):
    st.download_button(
        label="Download current session",
        data=export_current_session(),
        file_name=f"{st.session_state.current_session}.json",
        mime="application/json",
    )
    uploaded = st.file_uploader("Import a session JSON", type="json")
    if uploaded:
        try:
            import_session_from_json(uploaded.read().decode(), uploaded.name.replace(".json", ""))
            st.success("Session imported!")
            st.rerun()
        except Exception as e:
            st.error(f"session loaded")