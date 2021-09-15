FROM python:3.7
WORKDIR /app
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD streamlit run Dashboard/dashboard.py