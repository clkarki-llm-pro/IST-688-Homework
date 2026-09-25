# HW/HW5.py

#### 1. TOOLS WE NEED ####
import streamlit as st
from openai import OpenAI
import sys
import json
from pathlib import Path
from bs4 import BeautifulSoup

# A fix for working with ChromaDB on Streamlit Community Cloud
# This MUST run before chromadb is imported
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb


#### 2. SETTINGS ####
# Where the club pages live, where the database is saved, and which model to use
BASE_DIR = Path(__file__).parent.parent
HTML_FOLDER = BASE_DIR / 'HW-04-Data'
CHROMA_PATH = str(BASE_DIR / 'ChromaDB_for_HW4')
MODEL = 'gpt-5-mini'

# Create OpenAI client (only once, then reuse it)
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


#### 3. CUTTING A PAGE IN HALF ####
# Same semantic chunking as HW 4: two chunks per page, split at the
# paragraph boundary nearest the midpoint.
def split_into_two_chunks(text):
    paragraphs = [p for p in text.split('\n\n') if p.strip()]

    # If it's too short to split meaningfully, keep as one chunk
    if len(paragraphs) < 2:
        return [text]

    midpoint = len(text) // 2

    # Walk the paragraphs and cut at the boundary closest to the midpoint
    best_index = 1
    best_distance = None
    running = 0
    for i, para in enumerate(paragraphs[:-1]):
        running += len(para) + 2
        distance = abs(running - midpoint)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = i + 1

    first = '\n\n'.join(paragraphs[:best_index]).strip()
    second = '\n\n'.join(paragraphs[best_index:]).strip()
    return [c for c in (first, second) if c]


#### 4. READING THE WORDS OFF A WEB PAGE ####
def extract_text_from_html(html_path):
    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Drop the parts of the page that are navigation or code, not content
    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
        tag.decompose()

    text = soup.get_text(separator='\n')

    # Collapse the blank-line noise that get_text leaves behind
    lines = [line.strip() for line in text.splitlines()]
    return '\n\n'.join(line for line in lines if line)


#### 5. SAVING ONE PAGE'S CHUNKS INTO THE DATABASE ####
def add_documents_to_collection(collection, chunks, file_name):
    client = st.session_state.openai_client

    # One embeddings call for both chunks
    response = client.embeddings.create(
        input=chunks,
        model='text-embedding-3-small'
    )
    embeddings = [item.embedding for item in response.data]

    ids = [f'{file_name}_chunk{i + 1}' for i in range(len(chunks))]
    metadatas = [{'source': file_name, 'chunk': i + 1} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )


#### 6. FILLING THE DATABASE WITH ALL THE PAGES ####
def load_html_to_collection(folder_path, collection):
    folder = Path(folder_path)

    if not folder.is_dir():
        st.error(f'Could not find the folder {folder_path}')
        return 0

    html_files = sorted(folder.glob('*.html')) + sorted(folder.glob('*.htm'))
    loaded = 0
    for html_file in html_files:
        text = extract_text_from_html(html_file)
        if not text:
            st.warning(f'No text extracted from {html_file.name} - skipping')
            continue
        chunks = split_into_two_chunks(text)

        # Put the club's name (the first line of the page) at the start
        # of every chunk, so the second chunk can be matched to the club too
        club_name = text.split('\n\n')[0]
        chunks = [f'{club_name}\n\n{chunk}' for chunk in chunks]

        add_documents_to_collection(collection, chunks, html_file.name)
        loaded += 1
    return loaded


#### 7. OPENING THE DATABASE (AND BUILDING IT THE FIRST TIME) ####
if 'HW5_VectorDB' not in st.session_state:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # HW 5 has its own collection, so HW 4's database stays untouched
    collection = chroma_client.get_or_create_collection(name='HW5Collection')

    # Only build the DB the first time - on later runs the saved
    # collection already has the documents and this block is skipped
    if collection.count() == 0:
        with st.spinner('Building the vector database (first run only)...'):
            loaded = load_html_to_collection(HTML_FOLDER, collection)
        st.success(f'Loaded {loaded} pages into the vector database.')

    st.session_state.HW5_VectorDB = collection

collection = st.session_state.HW5_VectorDB


#### 8. PAGE TITLE ####
st.title('HW 5: Smarter Student Organizations Chatbot')
st.caption(
    'This bot decides for itself when to search the student organization pages '
    'and what to search for, so follow-up questions like "how do I join it?" work.'
)


#### 9. THE SEARCH TOOL ####
# The LLM calls this with a query it writes itself. We embed that query,
# search ChromaDB, and return the matching page text.
def relevant_club_info(query):
    client = st.session_state.openai_client
    response = client.embeddings.create(
        input=query,
        model='text-embedding-3-small'
    )
    query_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=4
    )

    documents = results['documents'][0]
    ids = results['ids'][0]

    extra_info = ''
    for doc_id, doc in zip(ids, documents):
        extra_info += f'--- PAGE: {doc_id} ---\n'
        extra_info += doc[:6000] + '\n\n'

    return extra_info, ids


#### 10. THE TOOL'S INSTRUCTION CARD ####
# The LLM reads this to decide when and how to call the tool
TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'relevant_club_info',
            'description': (
                'Search the Syracuse iSchool student organization pages and return '
                'the most relevant text. Use for any question about iSchool clubs: '
                'what they do, how to join, meetings, officers, contacts, or events.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': (
                            'A complete, standalone search phrase. Replace words like '
                            '"it" or "that club" with the actual organization name '
                            'from the conversation.'
                        ),
                    },
                },
                'required': ['query'],
            },
        },
    }
]


#### 11. THE BOT'S RULES ####
# 11a. Rules for the first call: when to search and how to write the query
SYSTEM_PROMPT = (
    'You are an assistant that answers questions about student '
    'organizations at the Syracuse iSchool.\n\n'
    'You have a tool, relevant_club_info, that searches the student organization pages.\n\n'
    'Rules:\n'
    '- Call relevant_club_info for any question about iSchool student organizations.\n'
    '- Write the query as a standalone search phrase. If the user refers back to '
    'something earlier ("it", "that club"), use the conversation to fill in the name.\n'
    '- Do not call the tool for greetings, thanks, or small talk. Just reply normally.\n'
    '- Do not make up organization names, officers, meeting times, or contacts.'
)

# 11b. Rules for the second call: answer from the pages the search found
# (same rules as HW 4 - the pages get added to the end)
ANSWER_PROMPT = (
    'You are an assistant that answers questions about student '
    'organizations at the Syracuse iSchool. '
    'Use the pages below to answer the question.\n\n'
    'Rules:\n'
    '- When you use the pages, start your answer with '
    '"Based on the student organization pages:" and name the page(s) you used.\n'
    '- If the pages do not answer the question, say so, then start with '
    '"Answering from general knowledge:" before continuing.\n'
    '- Do not make up organization names, officers, meeting times, or contacts.\n\n'
    'Pages:\n'
)


#### 12. THE "HOW I FOUND THIS" BOX ####
# Shows what the bot searched for and which pages came back
def show_searches(searches):
    with st.expander('How I found this'):
        for search in searches:
            st.markdown(f'**Searched for:** {search["query"]}')
            st.markdown('**Pages:** ' + ', '.join(search['sources']))


#### 13. SHOWING THE CHAT SO FAR ####
if 'HW5_messages' not in st.session_state:
    st.session_state.HW5_messages = [
        {'role': 'assistant',
         'content': 'Ask me anything about the iSchool student organizations.'}
    ]

for msg in st.session_state.HW5_messages:
    with st.chat_message(msg['role']):
        st.write(msg['content'])
        if msg.get('searches'):
            show_searches(msg['searches'])


#### 14. WHEN THE USER ASKS SOMETHING ####
if prompt := st.chat_input('What would you like to know?'):

    # 14a. Show the user's question and remember it
    st.session_state.HW5_messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('user'):
        st.markdown(prompt)

    # 14b. Short-term memory: the last 5 interactions (10 messages).
    # Only role and content are sent - the 'searches' key is just for
    # display, and the API rejects keys it doesn't recognize
    buffer = [
        {'role': m['role'], 'content': m['content']}
        for m in st.session_state.HW5_messages[-10:]
    ]
    while buffer and buffer[0]['role'] == 'assistant':
        buffer = buffer[1:]

    searches = []
    client = st.session_state.openai_client

    with st.chat_message('assistant'):
        try:
            # 14c. First call: the LLM decides whether to search and writes the query.
            # Not streamed, because we need the whole reply to see if it's a tool call
            with st.spinner('Thinking...'):
                first = client.chat.completions.create(
                    model=MODEL,
                    messages=[{'role': 'system', 'content': SYSTEM_PROMPT}] + buffer,
                    tools=TOOLS,
                    tool_choice='auto',
                )
            reply = first.choices[0].message

            if not reply.tool_calls:
                # No search needed (e.g. a greeting) - show the answer directly
                response = reply.content or "Sorry, I didn't get a response. Please try again."
                st.markdown(response)
            else:
                # 14d. Run the search with the query the LLM wrote
                pages = ''
                for tool_call in reply.tool_calls:
                    try:
                        query = json.loads(tool_call.function.arguments).get('query') or prompt
                    except json.JSONDecodeError:
                        query = prompt

                    with st.spinner(f'Searching club pages for "{query}"...'):
                        extra_info, source_ids = relevant_club_info(query)

                    searches.append({'query': query, 'sources': source_ids})
                    pages += extra_info

                # 14e. Second call: put the pages in the system prompt, like HW 4.
                # No tools are passed, so the LLM cannot call the function again
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=[{'role': 'system', 'content': ANSWER_PROMPT + pages}] + buffer,
                    stream=True,
                )
                response = st.write_stream(stream)
                show_searches(searches)

        except Exception as e:
            response = f'Sorry, something went wrong: {e}'
            st.error(response)

    # 14f. Remember the bot's answer for next time
    st.session_state.HW5_messages.append(
        {'role': 'assistant', 'content': response, 'searches': searches}
    )