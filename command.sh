

python3 kill_processes.py
pkill -f streamlit

streamlit run 01_generate.py --server.port=8501 &
streamlit run 02_evaluate.py --server.headless=true --server.port=8502 &
streamlit run 00_about.py --server.headless=true --server.port=8503