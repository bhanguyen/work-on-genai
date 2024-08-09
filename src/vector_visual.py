import json
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv


# Set page configuration
# st.set_page_config(layout="centered")

# Initialize the sentence transformer model for generating query embedding if not in the database
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

# Function to query the PostgreSQL database
@st.cache_data
def query_database(query):
    load_dotenv()

    conn = psycopg2.connect(
        dbname=os.environ['PGVECTOR_DATABASE'],
        user=os.environ['PGVECTOR_USER'],
        password=os.environ['PGVECTOR_PASSWORD'],
        host=os.environ['PGVECTOR_HOST'],
        port=os.environ['PGVECTOR_PORT']
    )
    cursor = conn.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

# Function to get embeddings from the database
@st.cache_data
def get_embeddings_from_db(node_ids):
    embeddings = []
    for node_id in node_ids:
        result = query_database(
            f"SELECT embedding FROM public.data_llamaindex WHERE node_id = '{node_id}'")
        if result:
            embedding = json.loads(result[0][0])  # Assuming the embedding is stored as a JSON string
            print(embedding)
            embeddings.append(embedding)
    return np.array(embeddings)

# Function to get the embedding for a query
@st.cache_data
def get_query_embedding(query):
    result = query_database(f"SELECT embedding FROM public.data_llamaindex WHERE node_id = '{query}'")
    if result:
        embedding = np.array(result[0][0])
    else:
        embedding = model.encode(query)  # Use the model to get the embedding
    return embedding

# Function to reduce embeddings using PCA
@st.cache_data
def reduce_dimensionality(embeddings, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings, pca

# Function to plot 2D embeddings
def plot_embeddings_2d(embeddings, query_embedding):
    plt.figure(figsize=(10, 7))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c='blue', label='Context Embeddings')
    plt.scatter(query_embedding[0], query_embedding[1], c='red', label='Query Embedding', marker='x', s=200)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.title('Embedding Vectors Visualization (2D)')
    st.pyplot(plt.gcf())

def plot_embeddings_3d(embeddings, query_embedding):

    fig = go.Figure()

    # Plot context embeddings
    fig.add_trace(go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers',
        marker=dict(color='blue', size=6),
        name='Context Embeddings'
    ))

    # Plot query embedding
    fig.add_trace(go.Scatter3d(
        x=[query_embedding[0]],
        y=[query_embedding[1]],
        z=[query_embedding[2]],
        mode='markers',
        marker=dict(color='red', size=4, symbol='x'),
        name='Query Embedding'
    ))

    fig.update_layout(
        title='Embedding Vectors Visualization (3D)',
        scene=dict(
            xaxis_title='PCA1',
            yaxis_title='PCA2',
            zaxis_title='PCA3',
            xaxis=dict(showgrid=True, gridcolor='gray'),
            yaxis=dict(showgrid=True, gridcolor='gray'),
            zaxis=dict(showgrid=True, gridcolor='gray')
        ),
        legend_title='Legend',
        showlegend=False,
        width=800,
        height=700,
    )

    st.plotly_chart(fig)

def main(context_node_ids, question):
    st.title("Embedding Vectors Visualization")
    # Input query
    query = question

    # Get the embedding for the query
    query_embedding = get_query_embedding(query)

    # Display the actual vector values
    st.subheader("Query Embedding (Original)")
    st.write(str(query_embedding))

    # Example node_ids for context
    context_node_ids = context_node_ids

    # Get embeddings from database
    context_embeddings = get_embeddings_from_db(context_node_ids)

    if context_embeddings.size == 0:
        st.error("No context embeddings found in the database.")
        return

    if query_embedding.shape[0] != context_embeddings.shape[1]:
        st.error(f"The dimension of the query embedding {query_embedding.shape[0]} does not match the context embeddings {context_embeddings.shape[1]}.")
        return

    # # Option to select visualization type
    # visualization_type = st.selectbox("Select visualization type", ("2D", "3D"))

    # # Set number of components based on visualization type
    # n_components = 2 if visualization_type == "2D" else 3
    n_components = 3

    # Reduce dimensionality of context embeddings
    reduced_context_embeddings, pca = reduce_dimensionality(context_embeddings, n_components=n_components)

    # Transform the query embedding using the same PCA model
    reduced_query_embedding = pca.transform([query_embedding])[0]

    # Display the reduced vector values
    st.subheader("Query Embedding (Reduced)")
    st.write(reduced_query_embedding)

    # Plot embeddings based on selected visualization type
    # if visualization_type == "2D":
    #     plot_embeddings_2d(reduced_context_embeddings, reduced_query_embedding)
    # else:
    #     plot_embeddings_3d(reduced_context_embeddings, reduced_query_embedding)
    plot_embeddings_3d(reduced_context_embeddings, reduced_query_embedding)

if __name__ == "__main__":

    query = st.text_input("Enter your query:")
    main(context_node_ids, query)