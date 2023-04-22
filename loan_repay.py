import streamlit as st
import pandas as pd
import numpy as np
import pickle

clf_pickle = open('dt.pickle', 'rb')
map_pickle = open('output.pickle', 'rb')

rfc = pickle.load(clf_pickle)
result_mapping = pickle.load(map_pickle)
clf_pickle.close()
map_pickle.close()

st.header("Loan Repayment using Decision Tree")
st.subheader("Input the following details")

initial_payment = st.number_input('Initial Payment', min_value=0)
last_payment = st.number_input('Last Payment ', min_value=0)
credit_score = st.number_input('Credit Score', min_value=0)
house_number = st.number_input('House Number', min_value=0)

results = [initial_payment, last_payment, credit_score,

           house_number]

if st.button("Calculate Result"):
    new_prediction = rfc.predict([results])

    if new_prediction[0] == 'yes':
        st.error('The client will  repay the loan')
    elif new_prediction[0] == 'No':
        st.success('The client will no repay the loan')
    else:
        st.text('The result is indecisive. Check other parameters')
