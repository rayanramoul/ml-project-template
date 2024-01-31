import streamlit as st
import pandas as pd


st.title('Streamlit Application')

file = st.file_uploader('Upload CSV file', type='csv')
if file:
    df = pd.read_csv(file)
    st.dataframe(df)
    
    option = st.selectbox('Select a column', df.columns)
    
    button = st.button('Plot')
    if button:
        st.bar_chart(df[option])