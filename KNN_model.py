import pandas as pandas
import numpy as numpy
import pickle
import streamlit as st

pickle_in=open("classification_KNN.pkl","rb")
df=pickle.load(pickle_in)

def predict_status(sepal_length,sepal_width,petal_length,petal_width):
    pred=df.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    return pred

st.title("Iris flower classification")
html_temp=  """"
                <h1>Iris flower classification</h1>
            """

st.markdown(html_temp,unsafe_allow_html=True)
sepal_length=st.number_input("enter the sepal length: ")
sepal_width=st.number_input("enter the sepal width : ") 
petal_length=st.number_input("enter the petal length : ")
petal_width=st.number_input("enter the petal width : ")
result=""

if st.button("Predict"):
    result=predict_status(sepal_length,sepal_width,petal_length,petal_width)
st.success("The output is{}".format(result))    


