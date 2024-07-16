import streamlit as st

def home_page():
    st.title("AI Tool Suite: Knowledge Assistant, Text-to-SQL Agent, and LLM Comparison")

def main():
    st.set_page_config(layout='wide')

    home_pg =st.Page(home_page, title="Home", icon=":material/home:")
    compare_pg = st.Page("llm-comparison.py", title="Compare LLMs", icon=":material/compare:")
    qa_pg = st.Page("qabot.py", title="Knowledge Assistant", icon=":material/contact_support:")
    t2sql_pg = st.Page("text2sql.py", title="T2SQL", icon=":material/robot_2:")
    vector_pg = st.Page("vector_visual.py", title="To Embedding", icon=":material/database:")
    
    pg = st.navigation([home_pg, vector_pg, compare_pg, qa_pg, t2sql_pg])
    pg.run()

if __name__ == "__main__":
    main()