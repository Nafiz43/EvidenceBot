import pandas as pd
# from sentence_transformers import SentenceTransformer
# from bert_score import BERTScorer
# from rouge_score import rouge_scorer
# from sentence_transformers import SentenceTransformer, util
# from rouge_score import rouge_scorer
import streamlit as st
import plotly.express as px
# import math
# from collections import Counter
import numpy as np
from utils import custom_css, nav_bar_evaluate_page, calculate_bleu_4, rouge_l_score, cosine_similarity_score, bert_score
# st.set_page_config(page_title='EvidenceBot - Evaluate')


st.markdown(custom_css, unsafe_allow_html=True)

# Navbar
st.markdown(nav_bar_evaluate_page, unsafe_allow_html=True)

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
        reference_text = st.text_area('Reference Text', 'Reference text goes here...', help='It is the ground truth text.')
    with text_boxes[1]:
        candidate_text = st.text_area('Candidate Text', 'Candidate text goes here...', help='It is the generated text that needs to be evaluated.')



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