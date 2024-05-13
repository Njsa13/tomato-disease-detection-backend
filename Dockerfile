FROM python:3.12.2

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD uvicorn main:app --port=8000 --host=0.0.0.0