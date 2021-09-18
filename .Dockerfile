FROM python:3.7
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run Dashboard/dashboard.py --server.port 8501 --server.enableCORS=false
EXPOSE 8501