import streamlit as st
from openai import OpenAI
import sys
from pathlib import Path
from bs4 import BeautifulSoup

# A fix for working with ChromaDB on Streamlit Community Cloud
# This MUST run before chromadb is imported
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb

### Using Chroma DB with OpenAI embeddings ###

# Folder holding the student-organization HTML pages, resolved relative to
# this file so the app works no matter where Streamlit was launched from
BASE_DIR = Path(__file__).parent.parent
HTML_FOLDER = BASE_DIR / 'HW-04-Data'
CHROMA_PATH = str(BASE_DIR / 'ChromaDB_for_HW4')

# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

#### CHUNKING ####
# Chunking method: semantic chunking, two chunks per page.
# Each page is split at the paragraph boundary nearest the midpoint.
#
# Why semantic chunking: fixed-size chunking cuts sentences off
# mid-thought and ignores natural breaks, so a fragment's embedding
# captures little meaning. Splitting on a paragraph keeps ideas intact
# and tends to separate what the club is from how to join it. Adaptive chunking's
# main advantage is in letting a model decide where and how many times to cut the chunks.
# Since, we have a fixed cut point, we can use the midpoint rule to split the page into 
# 2 chunks. Therefore, semantic chunking is a good choice for this assignment.

# the assignment asks for two, and halving the page brings the
# chunk closer in length to a typical short question, which improves the
# similarity score. Variable chunk size is the tradeoff, but these pages
# are short and the midpoint rule keeps the halves close in size.

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

 
#### Extract text from HTML files ####
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
 
 
# A function that will add one document's chunks to the collection
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

 
#### POPULATE COLLECTION WITH HTML PAGES ####
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
        add_documents_to_collection(collection, chunks, html_file.name)
        loaded += 1
    return loaded
 

#### Store the vector database collection in st.session_state.HW4_VectorDB
if 'HW4_VectorDB' not in st.session_state:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name='HW4Collection')
 
    # Only build the DB the first time - on later runs the persisted
    # collection already has the documents and this block is skipped
    if collection.count() == 0:
        with st.spinner('Building the vector database (first run only)...'):
            loaded = load_html_to_collection(HTML_FOLDER, collection)
        st.success(f'Loaded {loaded} pages into the vector database.')
 
    st.session_state.HW4_VectorDB = collection
 
collection = st.session_state.HW4_VectorDB
 

#### MAIN APP ####
#### MAIN APP ####
st.title('HW 4: iSchool Student Organizations Chatbot')
 
 
#### GET RELEVANT INFO FROM THE VECTOR DB ####
def get_info_from_vectordb(collection, query):
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
 
 
#### THE CHATBOT ####
if 'HW4_messages' not in st.session_state:
    st.session_state.HW4_messages = [
        {'role': 'assistant',
         'content': 'Ask me anything about the iSchool student organizations.'}
    ]
 
for msg in st.session_state.HW4_messages:
    chat_msg = st.chat_message(msg['role'])
    chat_msg.write(msg['content'])
 
if prompt := st.chat_input('What would you like to know?'):
 
    st.session_state.HW4_messages.append({'role': 'user', 'content': prompt})
 
    with st.chat_message('user'):
        st.markdown(prompt)
 
    extra_info, source_ids = get_info_from_vectordb(collection, prompt)
 
    system_prompt = (
        'You are an assistant that answers questions about student '
        'organizations at the Syracuse iSchool. '
        'Use the pages below to answer the question.\n\n'
        'Rules:\n'
        '- When you use the pages, start your answer with '
        '"Based on the student organization pages:" and name the page(s) you used.\n'
        '- If the pages do not answer the question, say so, then start with '
        '"Answering from general knowledge:" before continuing.\n'
        '- Do not make up organization names, officers, meeting times, or contacts.\n\n'
        'Pages:\n' + extra_info
    )
 
    # Conversation memory buffer: the last 5 interactions, where one
    # interaction is a user message plus the assistant's reply (10 messages)
    buffer = st.session_state.HW4_messages[-10:]
    while buffer and buffer[0]['role'] == 'assistant':
        buffer = buffer[1:]
 
    with st.chat_message('assistant'):
        client = st.session_state.openai_client
        stream = client.chat.completions.create(
            model='gpt-5-mini',
            messages=[{'role': 'system', 'content': system_prompt}] + buffer,
            stream=True,
        )
        response = st.write_stream(stream)
 
    st.session_state.HW4_messages.append({'role': 'assistant', 'content': response})
 