# streamlit-dockerized
Example of a basic dockerized streamlit app
<div align="center">
  <img src="https://github.com/QUANT-AI-Lab/streamlit-dockerized/blob/master/resources/screenshot.png?raw=true" alt="Findr"></img>

# How to run ?
## Without docker 
```bash
pip install -r requirements.txt
streamlit run app.py
```
## With docker 
```bash
docker build --tag streamlit-dockerized .
docker run --port 8501:8501 streamlit-dockerized
```
