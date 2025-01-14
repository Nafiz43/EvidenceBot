

python3 kill_processes.py
pkill -f streamlit

streamlit run generate.py --server.port=8501 &
streamlit run evaluate.py --server.headless=true --server.port=8502 &
streamlit run about.py --server.headless=true --server.port=8503