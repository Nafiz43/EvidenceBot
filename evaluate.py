import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from nltk.translate import meteor_score
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rouge_score import rouge_scorer
import streamlit as st
import plotly.express as px
import math
from collections import Counter
import evaluate
import numpy as np
# st.set_page_config(page_title='EvidenceBot - Evaluate')


def cosine_similarity_score(sentences1, sentences2):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings1 = model.encode(sentences1, convert_to_tensor=True)
    embeddings2 = model.encode(sentences2, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Move tensors from GPU to CPU
    cos_scores_cpu = cos_scores.cpu()

    return cos_scores_cpu.numpy().diagonal()

def rouge_l_score(sentence1, sentence2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = scorer.score(sentence1, sentence2)['rougeL'].fmeasure
    return rouge_score



def bert_score(sentences1, sentences2):
    scorer = BERTScorer(model_type='bert-base-uncased')
    results = scorer.score([sentences1], [sentences2])[2]
    return results


def calculate_bleu_4(candidate, reference):
    """
    Calculate BLEU-4 score between a candidate translation and a reference translation.

    Args:
        candidate (str): Candidate translation string.
        reference (str): Reference translation string.

    Returns:
        float: BLEU-4 score.
    """
    def get_ngram_counts(sentence, n):
        """
        Calculate n-gram counts for a given sentence.
        """
        ngram_counts = Counter()
        words = sentence.split()
        for i in range(len(words)):
            for j in range(1, n + 1):
                if i + j <= len(words):
                    ngram = ' '.join(words[i:i + j])
                    ngram_counts[ngram] += 1
        return ngram_counts

    def compute_bleu_4(candidate, reference):
        """
        Compute BLEU-4 score between a candidate translation and a reference translation.
        """
        candidate_len = len(candidate.split())
        reference_len = len(reference.split())
        precision = 0.0
        n = 4  # BLEU-4 specific
        for i in range(1, n + 1):
            candidate_ngrams = get_ngram_counts(candidate, i)
            reference_ngrams = get_ngram_counts(reference, i)
            common_ngrams = sum(min(candidate_ngrams[ngram], reference_ngrams[ngram]) for ngram in candidate_ngrams)
            precision += common_ngrams / max(1, candidate_len - i + 1)
        precision /= n
        if precision == 0:
            return 0.0  # Avoiding math domain error when precision is zero
        brevity_penalty = min(1, math.exp(1 - reference_len / candidate_len))
        bleu = brevity_penalty * math.exp(math.log(precision) / n)
        return bleu

    return compute_bleu_4(candidate, reference)


# Define custom CSS styles
custom_css = """
<style>
    body {
        background-color: #000000;
    }
    .container {
        max-width: 800px;
        margin: 40px auto;
        padding: 20px;
        background-color: #fff;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .form-group label {
        font-weight: bold;
    }
    .navbar {
        background-color: #8dd5e3;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .navbar-brand {
        font-weight: bold;
        font-size: 36px;
        color: #343a40;
        text-decoration: none;
    }
    .navbar-brand:hover,
    .navbar-brand:focus {
        text-decoration: none;
        color: #343a40;
    }
    .navbar-nav {
        display: flex;
        list-style-type: none;
    }
    .navbar-nav .nav-item {
        margin-left: 10px;
    }
    .navbar-nav .nav-link {
        color: #343a40;
        padding: 5px 15px;
        border-radius: 5px;
        transition: background-color 0.3s;
        margin-left: auto;
        text-decoration: none;
        font-size: 20px;
    }
    .navbar-nav .nav-link:hover,
    .navbar-nav .nav-link.active {
        background-color: #343a40;
        color: #fff;
    }
</style>
"""

# Display custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# Navbar
st.markdown("""
<nav class="navbar">
    <a class="navbar-brand" href="#">
        <img src="https://raw.githubusercontent.com/Nafiz43/portfolio/main/img/EvidenceBotLogo.webp" alt="Logo" width="60" height="60" class="d-inline-block align-top">
        EvidenceBot
    </a>
  <ul class="navbar-nav flex-row">
    <li class="nav-item">
      <a class="nav-link" href="http://localhost:8501">Generate</a>
    </li>
    <li class="nav-item">
      <a class="nav-link active" href="http://localhost:8502">Evaluate</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" href="http://localhost:8503" data-target="about">About</a>
    </li>
  </ul>
</nav>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
selected_option = st.radio(
    "Choose an option:",
    ["Individual Response Evaluation Mode", "Batch Response Evaluation Mode"],
    horizontal=True,  # Enables horizontal layout
    key="horizontal_radio_evaluation"
)

# st.markdown('<b><h5>Evaluation Metrics:</h5></b>', unsafe_allow_html=True)

# metric_options = st.columns(4)
# with metric_options[0]:
#     bert_checkbox = st.checkbox('BERT')
# with metric_options[1]:
#     bleu4_checkbox = st.checkbox('Bleu-4')
# with metric_options[2]:
#     rouge_l_checkbox = st.checkbox('Rouge-L')
# with metric_options[3]:
#     cosine_similarity_checkbox = st.checkbox('Cosine Similarity')


if selected_option=="Batch Response Evaluation Mode":
    file_uploader_ = st.columns(2)
    with file_uploader_[0]:
        reference_file = st.file_uploader("Upload Reference CSV File", type=["csv"], key='reference_file')
        if reference_file is not None:
            reference_file_data = pd.read_csv(reference_file)
            st.write("Here is your uploaded Reference file:")
            st.dataframe(reference_file_data)
        # else:
        #     st.write("Please upload a valid CSV file to proceed.")

    with file_uploader_[1]:
        candidate_file = st.file_uploader("Upload Candidate CSV File", type=["csv"], key='candidate_file')
        if candidate_file is not None:
            candidate_file_data = pd.read_csv(candidate_file)
            st.write("Here is your uploaded Candidate file:")
            st.dataframe(candidate_file_data)
        # else:
        #     st.write("Please upload a valid CSV file to proceed.")

    

else:
    text_boxes = st.columns(2)
    with text_boxes[0]:
        reference_text = st.text_area('Reference Text', 'Reference text goes here...')
    with text_boxes[1]:
        candidate_text = st.text_area('Candidate Text', 'Candidate text goes here...')



# Submit Button
if st.button('Submit'):
    if selected_option=="Batch Response Evaluation Mode":
        reference_data_array = []
        candidate_data_array = []

        if (len(reference_data_array)!=len(candidate_data_array)):
            raise ValueError("Length of both of the files are not same!")

        st.write(reference_file_data.columns[0])
        for index, row in reference_file_data.iterrows():
            if index == 0:
                continue
            reference_data_array.append(row[reference_file_data.columns[0]])
            # st.write(row[reference_file_data.columns[0]])
        for index, row in candidate_file_data.iterrows():
            if index==0:
                continue
            candidate_data_array.append(row[candidate_file_data.columns[0]])
            # st.write(row[candidate_file_data.columns[0]])
        bleu_4_score_v = 0
        rouge_l_score_v = 0
        cosine_similarity_score_v = 0
        bert_score_v = 0

        for i in range(0, len(reference_data_array)):
            bleu_4_score_v+= calculate_bleu_4(reference_data_array[i], candidate_data_array[i])
            rouge_l_score_v+= rouge_l_score(reference_data_array[i], candidate_data_array[i])
            cosine_similarity_score_v+= cosine_similarity_score(reference_data_array[i], candidate_data_array[i])
            bert_score_v+= bert_score(reference_data_array[i], candidate_data_array[i])

        bleu_4_score_v = np.round(bleu_4_score_v/len(reference_data_array),2)
        rouge_l_score_v = np.round(rouge_l_score_v/len(reference_data_array), 2)
        cosine_similarity_score_v = np.round(cosine_similarity_score_v/len(reference_data_array), 2)
        bert_score_v = np.round(bert_score_v/len(reference_data_array), 2)

        # st.write(bleu_4_score_v, rouge_l_score_v, cosine_similarity_score_v, bert_score_v)



    else:
        bleu_4_score_v = calculate_bleu_4(reference_text, candidate_text)
        rouge_l_score_v = rouge_l_score(reference_text, candidate_text)
        cosine_similarity_score_v = cosine_similarity_score(reference_text, candidate_text)
        bert_score_v = bert_score(reference_text, candidate_text)

    st.markdown('<b><h5>Results:</h5></b>', unsafe_allow_html=True)


    df = {
        'Category': ['Cosine Similarity', 'Rouge-L', 'Bleu-4', 'BERT'],
        'Value': [cosine_similarity_score_v.item()*100, rouge_l_score_v*100, bleu_4_score_v*100, bert_score_v.item()*100]
    }
    
    fig = px.bar(df, x='Category', y='Value', color='Category')
    
    # Customize the graph layout
    fig.update_layout(
        xaxis_title='Category',
        yaxis_title='Value',
        # title='Comparison sh',
        yaxis_range=[0, 100],  # Set the y-axis range from 0 to 100
    )
    
    # Display the graph using Streamlit
    st.plotly_chart(fig)