import streamlit as st
import streamlit.components.v1 as components
from utils import custom_css, nav_bar_about_page


st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(nav_bar_about_page, unsafe_allow_html=True)


# pdf_path = "figs/rag_pipeline.pdf"  # Make sure this path exists relative to your Streamlit app
# components.html(f"""
#     <iframe src="{pdf_path}" width="100%" height="600px" type="application/pdf"></iframe>
# """, height=600)


st.markdown("""
# 🛠️ **About EvidenceBot**
**EvidenceBot** is a cutting-edge, privacy-preserving tool designed to empower users with advanced interactions using Large Language Models (LLMs). By leveraging a Retrieval-Augmented Generation (RAG)-based pipeline, this app ensures efficient and secure processing of large document sets while maintaining data privacy.

---
## 📖 **System Architecture**
""")

st.image(
    "https://raw.githubusercontent.com/Nafiz43/portfolio/refs/heads/main/img/EvidenceBotArchitecture.png",
    caption="EvidenceBot System Architecture",
    use_container_width=True
)

st.markdown("""
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
