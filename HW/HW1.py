import streamlit as st
import pymupdf
from openai import OpenAI


def read_pdf(uploaded_file):
    pdf_bytes = uploaded_file.read()
    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    )
    text = ""
    for page in doc:
        text = text + page.get_text()
    doc.close()
    return text


st.title("My Document Question Answering")
st.write(
    "Upload a document below and ask a question "
    "about it – GPT will answer! To use this app, "
    "you need to provide an OpenAI API key, which "
    "you can get [here]"
    "(https://platform.openai.com/account/api-keys)."
)

openai_api_key = st.text_input(
    "OpenAI API Key",
    type="password",
)

if not openai_api_key:
    st.info(
        "Please add your OpenAI API key to continue.",
        icon="🗝️",
    )
else:

    client = OpenAI(api_key=openai_api_key)

    try:
        client.models.list()
    except Exception:
        st.error(
            "That API key doesn't seem to work. "
            "Please check it and try again.",
            icon="🚫",
        )
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)",
        type=("txt", "pdf"),
    )

    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'txt':
            document = uploaded_file.read().decode()
        elif file_extension == 'pdf':
            document = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")
            st.stop()

        prompt = (
            f"Here's a document: {document} "
            f"\n\n---\n\n {question}"
        )
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        stream = client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            stream=True,
        )

        st.write_stream(stream)