"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
import os
import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
import math
from collections import Counter

def log_to_csv(question, answer, m_name):
    log_dir, log_file = "History", "log.csv"
    # Ensure log directory exists, create if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Construct the full file path
    log_path = os.path.join(log_dir, log_file)

    # Check if file exists, if not create and write headers
    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # include the model name to match the logged values
            writer.writerow(["timestamp", "question", "answer", "model"])

    # Append the log entry
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer, m_name])

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


custom_css = """
<style>
    body {
      background-color:  #000000;
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
    .chat-container {
      margin-top: 20px;
      border: 1px solid #ccc;
      border-radius: 5px;
      padding: 10px;
      background-color: #f8f9fa;
    }
    .chat-messages {
      max-height: 300px;
      overflow-y: auto;
      margin-bottom: 10px;
    }
    .chat-message {
      background-color: #e9ecef;
      padding: 10px;
      margin-bottom: 10px;
      border-radius: 5px;
    }
    .chat-input {
      display: flex;
      align-items: center;
    }
    .chat-input input {
      flex: 1;
      margin-right: 10px;
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
      text-decoration: none; /* Add this line to remove the underline on hover and focus */
      color: #343a40; /* Add this line to maintain the text color on hover and focus */
    }


    .navbar-nav {
      display: flex;
      list-style-type: none; /* Add this line */
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

nav_bar_about_page = """
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
"""



nav_bar_generate_page = """
<nav class="navbar">
    <a class="navbar-brand" href="#">
        <img src="https://raw.githubusercontent.com/Nafiz43/portfolio/main/img/EvidenceBotLogo.webp" alt="Logo" width="60" height="60" class="d-inline-block align-top">
        EvidenceBot
    </a>
 <ul class="navbar-nav flex-row">
    <li class="nav-item">
      <a class="nav-link active" href="http://localhost:8501">Generate</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" href="http://localhost:8502">Evaluate</a>
    </li>
    <li class="nav-item">
      <a class="nav-link" href="http://localhost:8503" data-target="about">About</a>
    </li>
  </ul>
</nav>
"""

nav_bar_evaluate_page = """
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
"""

helper_model = """    
    Refers to the name or identifier of the local model used for generating responses. This model is responsible for processing input data, understanding context, and generating relevant outputs. Examples include `llama2:latest` or any model available in the local `ollama` environment.
    """

helper_embedding_model = """    
    Refers to the name or identifier of the embedding model employed to generate vector representations of the input data. This model is critical in tasks involving similarity searches, semantic understanding, or clustering, as it transforms raw data (e.g., text) into numerical embeddings that capture meaningful relationships and contextual semantics. Examples include `openai/text-embedding-ada-002` or custom-trained models
    """

helper_chunk_size = """
    Defines the size of each segment or chunk of text that is processed or embedded individually. This parameter determines how input data is divided into smaller, manageable pieces, typically measured in terms of characters, words, or tokens. Choosing an appropriate chunk size is essential to balance between capturing enough context in each chunk and avoiding excessive data redundancy or model limitations.
    """
helper_k = """
    Specifies the number of top results or nearest neighbors to return during similarity search or retrieval operations. This parameter determines how many entries from the database are considered most relevant to the input query, based on their similarity scores. Higher values of `K` provide broader results, while lower values focus on the most relevant matches.
    """
helper_temp = """
    Controls the randomness of the model's output. A higher value (e.g., `1.0`) results in more diverse outputs, while a lower value (e.g., `0.2`) makes the output more deterministic and focused.
    """
helper_top_p = """
    Regulates the probability mass of the tokens considered for sampling. The model selects from the smallest possible set of tokens whose cumulative probability is greater than or equal to `top_p`. A value of `0.9` typically balances diversity and relevance.
    """
helper_top_k = """
    Limits the model to sampling from the top `K` most likely tokens at each step. For example, setting `top_k = 50` considers only the top 50 tokens for generating the next word.
    """
helper_tfs_z = """
    Adjusts sampling by filtering tokens based on their tail free distribution. This helps to maintain a balance between creative and coherent output. A value of `1.0` retains all tokens, while lower values filter out less probable tokens.
    """
helper_num_ctx = """
    Defines the maximum number of tokens the model considers as context for generating predictions. A higher value allows the model to consider more history but increases computational cost.
    """
helper_repeat_penalty = """
    Discourages the model from repeating the same tokens by applying a penalty to their probability during sampling. A value greater than `1.0` (e.g., `1.1`) reduces repetition, while `1.0` disables the penalty.
    """
helper_mirostat = """
    A dynamic sampling technique aimed at maintaining a target perplexity during text generation. This helps to produce outputs with consistent quality and coherence.
    """
helper_mirostat_eta = """
    The learning rate for updating the perplexity in Mirostat. It determines how quickly the algorithm adjusts to achieve the target perplexity.
    """
helper_mirostat_tau = """
    The target perplexity value for Mirostat. This sets the desired balance between diversity and coherence in the generated text.
    """