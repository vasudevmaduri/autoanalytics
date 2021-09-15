FROM python:3.7
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . . 
CMD streamlit run Dashboard/dashboard.py --server.port 8501 --server.enableCORS