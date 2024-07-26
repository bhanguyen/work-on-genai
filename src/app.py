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

    model_categories = {
"Ollama": ["phi3", "llama3", "mistral", "gemma2"],
"Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-2.1"],
"OpenAI": ["gpt-4", "gpt-3.5-turbo"]
}
    with st.sidebar:
        # Initialize session state for selections
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = {category: [] for category in model_categories}

        # Category selection using checkboxes and model selection
        for category, models in model_categories.items():
            category_selected = st.checkbox(f"Include {category} models")
            if category_selected:
                st.session_state.selected_models[category] = st.multiselect(
                    f"Select {category} models:",
                    models,
                    default=st.session_state.selected_models[category]
                )
            else:
                st.session_state.selected_models[category] = []

    # Flatten the selected models list
    selected_models = [model for models in st.session_state.selected_models.values() for model in models]

    tabs = st.tabs(selected_models)
    for i, tab in enumerate(tabs):
        with tab:
            st.header(f"{selected_models[i]} Model")
            
if __name__ == "__main__":
    main()