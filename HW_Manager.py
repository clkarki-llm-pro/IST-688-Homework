import streamlit as st
st.set_page_config(page_title="HW Manager", page_icon="📚", layout="centered")

hw1 = st.Page("HW/HW1.py", title="HW 1: Document QA", icon = "1️⃣", default=True)
hw2 = st.Page("HW/HW2.py", title="HW 2: Document Summarization", icon = "2️⃣")

pg = st.navigation([hw1, hw2])
pg.run()