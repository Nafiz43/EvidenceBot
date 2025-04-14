# Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
# Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.


python3 kill_processes.py
pkill -f streamlit

streamlit run 01_generate.py --server.port=8501 &
streamlit run 02_evaluate.py --server.headless=true --server.port=8502 &
streamlit run 00_about.py --server.headless=true --server.port=8503