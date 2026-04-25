
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(layout= 'wide', page_title='Car Insurance Deployment')

st.image('https://thumbs.dreamstime.com/b/car-insurance-logo-auto-protection-safety-secure-shield-stock-photo-generative-ai-illustration-representing-vehicle-security-369131383.jpg?w=992')

df = pd.read_csv('cleaned_df.csv', index_col= 0)
# st.dataframe(df)

kidsdriv = st.selectbox('PLease provid number of driving kids', df.kidsdriv.unique())
age = st.sidebar.slider('Enter your Age', 16, 81)
homekids = st.selectbox('PLease provid total number of kids', df.homekids.unique())
yoj = st.sidebar.slider('Please provide number of years on job', min_value= int(df.yoj.min()), max_value= int(df.yoj.max()), step= 1)
income = st.number_input('Please provide your income', min_value= df.income.min(), max_value= df.income.max())
parent1 = st.selectbox('PLease select whether you are single parent or not', df.parent1.unique())
home_val = st.number_input('Please provide your home value', min_value= df.home_val.min(), max_value= df.home_val.max())
mstatus = st.sidebar.radio('Marital Status', df.mstatus.unique())
gender = st.sidebar.radio('Gender', df.gender.unique())
education = st.selectbox('Please enter your educational background', df.education.unique())
occupation = st.selectbox('Please enter your Occupation', df.occupation.unique())
travtime = st.sidebar.slider('Please enter your travel time in minutes', min_value= df.travtime.min(), max_value= df.travtime.max(), step= 1)
car_use = st.selectbox('PLease provid your car usage', df.car_use.unique())
bluebook = st.number_input('Please provide your car value', min_value= df.bluebook.min(), max_value= df.bluebook.max())
tif = st.sidebar.slider('Loyalty years', min_value= df.tif.min(), max_value= df.tif.max(), step= 1)
car_type = st.selectbox('PLease provid your car type', df.car_type.unique())
red_car = st.selectbox('Red car or Not', df.red_car.unique())
oldclaim = st.number_input('Please provide your old claim amount', min_value= df.oldclaim.min(), max_value= df.oldclaim.max())
clm_freq = st.sidebar.slider('Please provide number of previous claims', min_value= df.clm_freq.min(), max_value= df.clm_freq.max(), step= 1)
revoked = st.selectbox('License Revoked within 7 years', df.revoked.unique())
mvr_pts = st.sidebar.slider('Please provide vechile record points', min_value= df.mvr_pts.min(), max_value= df.mvr_pts.max(), step= 1)
clm_amt = st.number_input('Please provide your total claims amount', min_value= df.clm_amt.min(), max_value= df.clm_amt.max())
car_age = st.sidebar.slider('Please enter your car age', min_value= int(df.car_age.min()), max_value= int(df.car_age.max()), step= 1)
urbanicity = st.selectbox('Urbanicity', df.urbanicity.unique())
customer_loyalty = st.selectbox('Customer Loyalty', df.customer_loyalty.unique())

# Import Model pkl file
model = joblib.load('knn_model.pkl')

new_data = pd.DataFrame(columns= df.columns.drop('claim_flag'), 
                        data= [[kidsdriv, age, homekids, yoj, income, parent1, home_val,
                            mstatus, gender, education, occupation, travtime, car_use,
                            bluebook, tif, car_type, red_car, oldclaim, clm_freq,
                            revoked, mvr_pts, clm_amt, car_age, urbanicity,
                            customer_loyalty]])

if st.button('Predict'):

    result = model.predict(new_data)[0]

    if result == 0:
        st.write('Desired Customer')

    else:
        st.write('Undesired Customer')
