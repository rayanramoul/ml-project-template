FROM python:3.9-slim-buster

# Install dependencies
RUN apt-get update

COPY requirements.txt /requirements.txt
COPY . /opt/app

WORKDIR /opt/app

RUN pip install -r requirements.txt

EXPOSE 8501
CMD python -m streamlit run app.py