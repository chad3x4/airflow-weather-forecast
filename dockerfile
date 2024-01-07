FROM apache/airflow:latest

USER root
RUN apt-get update && \
    apt-get -y install git && \
    apt-get clean

USER airflow
RUN pip install tensorflow scikit-learn==1.2.2 joblib streamlit && pip install typing-extensions --upgrade