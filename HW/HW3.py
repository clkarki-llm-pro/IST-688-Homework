import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from google import genai
from google.genai import types

@st.cache_data(show_spinner=False)
def read_url_content(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator=" ")
        return " ".join(text.split())
    except requests.RequestException as e:
        st.warning(f"Could not read {url}: {e}")
        return None

# The system prompt shapes how the bot behaves on every turn
MAX_CHARS_PER_DOC = 20000


def build_system_prompt(documents):
    if not documents:
        return (
            "You are a helpful assistant. The user has not supplied any documents yet. "
            "Ask them to paste a URL in the sidebar before answering questions."
        )

    parts = [
        "You are a helpful assistant who answers questions using ONLY the reference "
        "documents provided below.",
        "",
        "Rules:",
        "- Base every answer on the documents. Do not use outside knowledge.",
        "- If the documents do not contain the answer, say so plainly instead of guessing.",
        "- When the documents disagree, point out the disagreement.",
        "- Say which document your answer came from.",
        "- Keep answers clear and reasonably short.",
        "",
    ]
    for i, (url, text) in enumerate(documents, start=1):
        parts.append(f"--- DOCUMENT {i} (source: {url}) ---")
        parts.append(text[:MAX_CHARS_PER_DOC])
        parts.append("")
    return "\n".join(parts)

#Show title
st.title("HW3 - Chat about two URLs")

st.write(
    "Paste up to two URLs in the sidebar and pick a model, then ask questions below. "
    "Both pages are downloaded and placed in a system prompt that is re-sent with every "
    "message, so the documents are never forgotten. For conversation memory this app uses "
    "a **buffer of the last 6 messages (3 user/assistant exchanges)** - anything older is "
    "dropped, so the bot will not remember the start of a long chat."
)

MODEL_CHOICES = {
    "OpenAI - gpt-5.5": ("openai", "gpt-5.5"),
    "Google - gemini-3.6-flash": ("google", "gemini-3.6-flash"),
}

st.sidebar.header("Options")

url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com")
url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com")

model_label = st.sidebar.selectbox("Which model?", list(MODEL_CHOICES.keys()))
vendor, model_name = MODEL_CHOICES[model_label]

if st.sidebar.button("Clear conversation"):
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
    st.rerun()

documents = []
for u in (url1, url2):
    if u.strip():
        text = read_url_content(u.strip())
        if text:
            documents.append((u.strip(), text))

if documents:
    st.sidebar.success(f"Loaded {len(documents)} document(s).")
else:
    st.sidebar.info("Add at least one URL to begin.")

SYSTEM_PROMPT = build_system_prompt(documents)


def stream_openai(model, system_prompt, buffer):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}] + buffer,
        stream=True,
    )


def stream_gemini(model, system_prompt, buffer):
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    contents = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in buffer
    ]

    def generator():
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        ):
            if chunk.text:
                yield chunk.text

    return generator()

if "messages" not in st.session_state:
    st.session_state.messages = \
    [{"role": "assistant", "content": "How can I help you?"}]

#Display chat messages from history on app rerun
for msg in st.session_state.messages:
    chat_msg = st.chat_message(msg["role"])
    chat_msg.write(msg["content"])

# React to user input
if prompt := st.chat_input("What is up?"):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    buffer = st.session_state.messages[-6:]
    while buffer and buffer[0]["role"] == "assistant":
        buffer = buffer[1:]

    with st.chat_message("assistant"):
        try:
            if vendor == "openai":
                stream = stream_openai(model_name, SYSTEM_PROMPT, buffer)
            else:
                stream = stream_gemini(model_name, SYSTEM_PROMPT, buffer)
            response = st.write_stream(stream)
        except Exception as e:
            response = f"Sorry, something went wrong: {e}"
            st.error(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})