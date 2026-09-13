import streamlit as st
st.set_page_config(page_title="HW Manager", page_icon="📚", layout="centered")

hw1 = st.Page("HW/HW1.py", title="HW 1: Document QA", icon = "1️⃣")
hw2 = st.Page("HW/HW2.py", title="HW 2: Document Summarization", icon = "2️⃣")
hw3 = st.Page("HW/HW3.py", title="HW 3: URL Chatbot", icon = "3️⃣", default=True)

pg = st.navigation([hw1, hw2, hw3])
pg.run()