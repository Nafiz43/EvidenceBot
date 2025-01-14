import pandas as pd
import torch
import streamlit as st
import math
from collections import Counter
import evaluate


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
      <a class="nav-link" href="http://localhost:8502">Evaluate</a>
    </li>
    <li class="nav-item">
      <a class="nav-link active" href="http://localhost:8503" data-target="about">About</a>
    </li>
  </ul>
</nav>
""", unsafe_allow_html=True)

# Evaluation Metrics section

st.markdown("""
# 🛠️ **About EvidenceBot**
**EvidenceBot** is a cutting-edge, privacy-preserving tool designed to empower users with advanced interactions using Large Language Models (LLMs). By leveraging a Retrieval-Augmented Generation (RAG)-based pipeline, this app ensures efficient and secure processing of large document sets while maintaining data privacy.

---

## ✨ **Key Features**
1. **Privacy-Preserving RAG Pipeline**:
   - Breaks documents into manageable chunks and retrieves only the most relevant context for queries.
   - Ensures sensitive data remains secure within a local environment.
2. **Customizable Parameters**:
   - Experiment with RAG-related parameters like chunk size and embedding models.
   - Adjust LLM parameters (e.g., temperature) to fine-tune outputs for specific tasks.
3. **Response Modes**:
   - **Chat Mode**: Engage with the tool as a chatbot.
   - **Batch Mode**: Submit multiple queries simultaneously and store results in a structured format.
4. **Comprehensive Evaluation Module**:
   - Compare LLM-generated responses against ground truth using metrics such as BLEU-4, ROUGE-L, METEOR, and BERTScore.
   - Supports graphical insights and batch processing for large-scale evaluations.

---

## 🖼️ **Interface and Accessibility**
Built with **Streamlit**, the app combines:
- **Intuitive UI**: Streamlined design for ease of use.
- **Dynamic Interactivity**: Seamless file uploads, configurable settings, and responsive outputs.

---

## 👩‍💻 **Developed By**
- **Nafiz Imtiaz Khan**  
  *University of California, Davis*  
  *[nikhan@ucdavis.edu](mailto:nikhan@ucdavis.edu)*  

- **Vladimir Filkov**  
  *University of California, Davis*  
  *[vfilkov@ucdavis.edu](mailto:vfilkov@ucdavis.edu)*  

---

## 📚 **Use Cases**
- **Efficient Document Analysis**: Process large-scale datasets with enhanced relevance and accuracy.
- **Parameter Experimentation**: Optimize tool performance for diverse scenarios.
- **LLM Response Evaluation**: Measure and refine model outputs for improved results.

---

## 🚀 **Future Prospects**
While currently focused on text-based data, future enhancements aim to include support for multimedia formats (audio, images, and videos) and compatibility with a broader range of open-source models.

---
""")
