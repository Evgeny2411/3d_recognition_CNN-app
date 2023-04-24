FROM python:3.9

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY App.py /app/
COPY model.py /app/

RUN mkdir /app/models
copy models /app/models

WORKDIR /app

EXPOSE 8501

CMD ["streamlit", "run", "App.py"]
