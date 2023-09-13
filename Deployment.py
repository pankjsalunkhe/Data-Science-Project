#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import streamlit as st
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
st.title('Model Development \n Telecommunication Churning')
st.sidebar.header('Input Features')
def input_features():
    Voice_Plan=st.sidebar.selectbox('Voice Plan',('1','0'))
    International_Plan=st.sidebar.selectbox('International Plan',('1','0'))
    International_Calls=st.sidebar.number_input('Insert Number Of Calls')
    International_Charges=st.sidebar.number_input('Insert International Charge')
    Day_Charges=st.sidebar.number_input('Insert Day Charge')
    Evening_Mins=st.sidebar.number_input('Insert Evening Minutes')
    Night_Mins=st.sidebar.number_input('Insert Night Minutes')

    data={'Voice_Plan':Voice_Plan,
          'International_Plan':International_Plan,
          'International_Calls':International_Calls,
          'International_Charges':International_Charges,
          'Day_Charges':Day_Charges,
          'Evening_Mins':Evening_Mins,
          'Night_Mins':Night_Mins}
    features=pd.DataFrame(data,index=[0])
    return features
df=input_features()
st.subheader('User Input Features')
st.write(df)

churn=pd.read_csv('Churn_Without_Outliers',encoding='utf_8')
x=churn.iloc[:,1:8]
y=churn.iloc[:,8]
sm=SMOTEENN()
X,Y = sm.fit_resample(x,y)
RF=RandomForestClassifier()
RF.fit(X,Y)
predict=RF.predict(df)
prediction_probability=RF.predict_proba(df)

st.subheader('Prediction Result')
st.write('Yes' if prediction_probability[0][1]>0.5 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_probability)

