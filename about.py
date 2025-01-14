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
st.markdown('<b><h5>About the App:</h5></b>', unsafe_allow_html=True)
st.image('archi.png', caption="My Image", use_column_width=True)