import os
import logging
import torch
import utils
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from langchain.llms import Ollama
import subprocess
import streamlit as st
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
from utils import *
global count
count =0


if "first_click_done" not in st.session_state:
    st.session_state.first_click_done = False

from prompt_template_utils import get_prompt_template
from langchain.vectorstores import Chroma

from constants import (
    PERSIST_DIRECTORY,
    # MAX_NEW_TOKENS,
    # MODELS_PATH,
    CHROMA_SETTINGS,
)

global x
global query_file_data
x=''
# prompt =''
def get_ollama_models():
    try:
        # Run the Ollama command to list available models
        result = subprocess.run(['ollama', 'ls'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if the command was successful
        if result.returncode == 0:
            # Split the output into lines
            output_lines = result.stdout.strip().split('\n')
            
            # Extract the model names (first column)
            model_names = [line.split()[0] for line in output_lines[1:]]  # Skip the header line
            
            return model_names
        else:
            return f"Error: {result.stderr}"
    
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def get_embeddings(device_type="cuda"):
    if "instructor" in EMBEDDING_MODEL_NAME:
        return HuggingFaceInstructEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            embed_instruction="Represent the document for retrieval:",
            query_instruction="Represent the question for retrieving supporting documents:",
        )

    elif "bge" in EMBEDDING_MODEL_NAME:
        return HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            query_instruction="Represent this sentence for searching relevant passages:",
        )

    else:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
        )
    
# st.set_page_config(page_title='EvidenceBot')
# EMBEDDING_MODEL_NAME = 'hkunlp/instructor-large'

# Display custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Navbar
st.markdown(nav_bar_generate_page, unsafe_allow_html=True)

# Generate Response section
st.markdown('<div id="generateResponse" class="section active">', unsafe_allow_html=True)

model_names = get_ollama_models()

col1, col2 = st.columns(2)
with col1:
    # m_name = st.selectbox('Model Name', ['llama2:latest', 'LLAMA2-70B', 'LLAMA3-8B', 'MIXTRAL-7B', 'MIXTRAL-8x7B'])
    m_name = st.selectbox("**LLM Model**", model_names, help=helper_model)
with col2:
    EMBEDDING_MODEL_NAME = st.selectbox(
    'Embedding Model', 
    [
        'hkunlp/instructor-large', 
        'hkunlp/instructor-base', 
        'sentence-transformers/all-MiniLM-L6-v2', 
        'sentence-transformers/paraphrase-mpnet-base-v2', 
        'sentence-transformers/distiluse-base-multilingual-cased-v2', 
        'sentence-transformers/all-distilroberta-v1'
    ], help=helper_embedding_model
)


col14, col15 = st.columns(2)

with col14:
    Chunk_Size = st.number_input(
        "Chunk_Size", 
        min_value=10, 
        max_value=10000, 
        step=10, 
        value=100, 
        help=helper_chunk_size
    )

with col15:
    size_of_k = st.number_input(
        "K", 
        min_value=1, 
        max_value=100, 
        step=1, 
        value=4, 
        help=helper_k
    )



col3, col4, col5 = st.columns(3)
with col3:
    temp = st.number_input("**Temperature**", min_value=0.0, max_value=1.0, step=0.01, value=0.1, help=helper_temp)
with col4:
    top_p = st.number_input("**Top_P**", min_value=0.0, max_value=1.0, step=0.01, value=0.1, help=helper_top_p)
with col5:
    top_k = st.number_input("Top_K", min_value=0, max_value=100, step=10, value=10, help=helper_top_k)



col6, col7, col8 = st.columns(3)
with col6:
    tfs_z = st.number_input("tfs_z", min_value=0.0, max_value=5.0, step=0.01, value=2.0, help=helper_tfs_z)
with col7:
    num_ctx = st.number_input("num_ctx", min_value=0, max_value=100000, step=10, value=2048, help=helper_num_ctx)
with col8:
    repeat_penalty = st.number_input("repeat_penalty", min_value=0.0, max_value=5.0, step=0.1, value=1.1, help=helper_repeat_penalty)


col9, col10, col11 = st.columns(3)
with col9:
    mirostat = st.number_input("mirostat", min_value=0, max_value=100, step=1, value=0, help=helper_mirostat)
with col10:
    mirostat_eta = st.number_input("mirostat_eta", min_value=0.0, max_value=5.0, step=0.1, value=0.01, help=helper_mirostat_eta)
with col11:
    mirostat_tau = st.number_input("mirostat_tau", min_value=0.0, max_value=5.0, step=0.1, value=5.0, help=helper_mirostat_tau)

col12, col13 = st.columns(2)
global selected_option
selected_option= st.radio(
    "Choose an option:",
    ["Individual Question Mode", "Batch Question Mode"],
    horizontal=True,  # Enables horizontal layout
    key="generate_horizontal_radio"
)

if selected_option=="Individual Question Mode":
  chat_input = st.text_area('Enter your prompt', key='instruction_input')
else:
    query_file = st.file_uploader("Upload Query CSV File", type=["csv"], key='reference_file')
    if query_file is not None:
      query_file_data = pd.read_csv(query_file)
      st.write("Here is your uploaded Query file:")
      st.dataframe(query_file_data)
    
st.markdown('</div>', unsafe_allow_html=True)




def retrieval_qa_pipline(device_type, use_history, promptTemplate_type):
    embeddings = get_embeddings(device_type)
    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": size_of_k})
    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)
    # llm= Ollama(model=m_name, temperature= temp)
    llm= Ollama(model=m_name, temperature= temp,top_p=top_p, top_k =top_k, 
            mirostat=mirostat, mirostat_eta=mirostat_eta, mirostat_tau=mirostat_tau, num_ctx=num_ctx,
            repeat_penalty=repeat_penalty, tfs_z=tfs_z)
    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )
    return qa


def get_llm_response(query = "Query"):
    global m_name
    global LLM_ANSWER
    global LLM_SOURCE_DOC
    print(m_name, query)
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    show_sources = True
    use_history = True
    model_type = "llama"
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")
    cnt =0
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]
    LLM_ANSWER = answer
    LLM_SOURCE_DOC = docs
    utils.log_to_csv(query, answer, m_name)
    return 1
    
if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    if st.button("Send"):
        if not st.session_state.first_click_done:
          print("sending value")
          subprocess.run(['python', 'ingest.py', EMBEDDING_MODEL_NAME, str(Chunk_Size)])
          st.session_state.first_click_done = True


        if selected_option == "Individual Question Mode":    
            if chat_input == "":
                st.warning("Enter your instruction")
            else:
                get_llm_response(chat_input)
                x += '<b><h5>LLM Response:</h5></b>'
                x += LLM_ANSWER

                x += '<b><h5>SOURCE CONTEXT:</h5></b>'
                x += str(LLM_SOURCE_DOC)

                st.markdown(
                    """
                    <style>
                    .scrollable-div {
                        max-height: 800px; /* Adjust the height as needed */
                        overflow-y: scroll;
                        padding: 10px;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                        background-color: #f9f9f9;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # Display LLM Response
                st.markdown(f'<div class="scrollable-div">{x}</div>', unsafe_allow_html=True)

        elif selected_option == "Batch Question Mode":
            progress_text = st.empty()  # Create an empty placeholder
            for index, row in query_file_data.iterrows():
                # if index == 0:
                #   continue  # Skip the first row
                get_llm_response(row[query_file_data.columns[0]])
                progress_text.text(f"Processed {index + 1} out of {len(query_file_data)}")
            st.write("Processing Finished...")
        