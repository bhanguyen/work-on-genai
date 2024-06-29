import streamlit as st

def home_page():
    st.title("HELLO WORD")

def main():
    st.set_page_config(layout='wide')

    home_pg =st.Page(home_page, title="Home", icon=":material/home:")
    compare_pg = st.Page("llm-comparison.py", title="Compare LLMs", icon=":material/compare:")
    qa_pg = st.Page("qabot.py", title="QA Chatbot", icon=":material/contact_support:")
    t2sql_pg = st.Page("text2sql.py", title="T2SQL", icon=":material/robot_2:")
    
    pg = st.navigation([home_pg, compare_pg, qa_pg, t2sql_pg])
    pg.run()

if __name__ == "__main__":
    main()