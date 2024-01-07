# How to run
- Run Docker
- cd to repo
- Run:
  + docker build --pull --rm -f "dockerfile" -t my-airflow:latest "."
  + docker compose up -d
- Open localhost:8080
- Login using password in \airflow\standalone_admin_password.txt

- Run Streamlit: 
  + pip install streamlit
  + streamlit run test-streamlit.py