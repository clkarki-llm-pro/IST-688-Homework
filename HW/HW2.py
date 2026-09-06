import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None

# Show title and description.
st.title("My Document question answering")
st.write(
    "Paste a URL below and ask a question about it – GPT will answer! "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    summary_type = st.sidebar.selectbox(
        "Type of summary",
        (
            "Summarize in 100 words",
            "Summarize in 2 connecting paragraphs",
            "Summarize in 5 bullet points"
        ),
    )

    instruction_map = {
        "Summarize in 100 words": "Summarize the document in about 100 words.",
        "Summarize in 2 connecting paragraphs": "Summarize the document in exactly 2 connecting paragraphs.",
        "Summarize in 5 bullet points": "Summarize the document in exactly 5 bullet points."
    }
    instruction = instruction_map[summary_type]

    language = st.sidebar.selectbox(
            "Output language",
            ("English", "Spanish", "Nepali"),
        )

    use_advanced_model = st.sidebar.checkbox("Use advanced model")
    model = "gpt-5-nano" if use_advanced_model else "gpt-3.5-turbo"

    # Validate the key right away by making a test call.
    # This catches a bad/fake key immediately, instead of waiting
    # until the user has uploaded a document and asked a question.
    try:
        client.models.list()
    except Exception:
        st.error("That API key doesn't seem to work. Please check it and try again.", icon="🚫")
        st.stop()
    # Let the user enter a URL to summarize.
    url = st.text_input(
        "Enter a URL to summarize",
        placeholder="https://example.com",
    )

    if url:

        # Process the URL.
        document = read_url_content(url)
        if document is None:
            st.error("Couldn't read that URL. Check the address and try again.")
            st.stop()

        messages = [
            {
                "role": "user",
                "content": (
                    f"Here's a document: {document}\n\n---\n\n{instruction}\n\n"
                    f"Write your entire response in {language}. "
                    f"Do not include any text in any other language."
                ),
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)
