#import packages
import numpy as np
import pandas as pd
import matplotlib as mp
import altair as alt
import streamlit as st
import datetime
import dateutil
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Create cache with data objects 
@st.cache(allow_output_mutation=True)
def make_model(dataLoc):
    
    data = pd.read_csv(dataLoc)
    
    def clean_sm(x):
        x = np.where(x == 1, 1, 0)
        return x

    ss = pd.DataFrame({'sm_li': clean_sm(data['web1h']),
          'income': np.where(data['income'] > 9, 0, data['income']),
          'education': np.where(data['educ2'] > 8, 0, data['educ2']),
          'parent': clean_sm(data['par']),
          'married': clean_sm(data['marital']),
          'female': np.where(data['gender'] == 2, 1, 0),
          'age': np.where(data['age'] > 98, 0, data['age'])})

    y = ss['sm_li']
    X = ss.drop(columns='sm_li')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    log_model = LogisticRegression(random_state = 42,class_weight = 'balanced').fit(X_train, y_train)
    user_input = pd.DataFrame([[np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]],columns = X_train.columns.values, index = ['User Input'])

    education_matrix = {'ed_levels': ['Less than high school',
    'High school incomplete',
    'High school graduate',
    'Some college, no degree',
    'Two-year associate degree from a college or university',
    'Four-year college or university degree',
    'Some postgraduate or professional schooling',
    'Postgraduate or professional degree'], 'ed_index': range(1,9)}
    
    education_df = pd.DataFrame(data = education_matrix)

    salary_matrix = {'salary_levels': ['Less than $10,000',
    '$10K to $20K',
    '$20K to $30K',
    '$30K to $40K',
    '$40K to $50K',
    '$50K to $75K',
    '$75K to $100K',
    '$100K to $150K',
    '$150K or more'], 'salary_index': range(1,10)}

    salary_df = pd.DataFrame(data = salary_matrix)

    return log_model, user_input, education_df, salary_df

log_model, user_input, education_df, salary_df = make_model('./social_media_usage.csv')

with st.form(key='my_form_to_submit', clear_on_submit = False):
    st.title('Let\'s see if i can predict whether you\'re a LinkedIn user')
    st.header('A fun little app by Logan Suba')

    st.text('\nTo start, please tell us a bit about yourself')

    user_name = st.text_input('First name') + ' ' + st.text_input('Last name')

    dob = st.date_input('When is your birthday?',min_value = datetime.date(1900, 1, 1))

    age = abs((datetime.date.today() - dob).days) / 365

    education = st.selectbox('What is your eduation level?', education_df['ed_levels'])

    education_index = pd.to_numeric(education_df['ed_index'][education_df['ed_levels'] == education].iloc[0])

    salary = st.selectbox('What is your yearly salary?', salary_df['salary_levels'])

    salary_index = pd.to_numeric(salary_df['salary_index'][salary_df['salary_levels'] == salary].iloc[0])

    parent = st.radio('Are you a parent?', ['Yes', 'No'])

    if parent == 'Yes':
        parent_index = 1
    else:
        parent_index = 0 

    married = st.radio('Are you Married?', ['Yes', 'No'])

    if married == 'Yes':
        married_index = 1
    else:
        married_index = 0 

    gender = st.radio('Whats your Gender?', ['Female', 'Male'])

    if gender == 'Female':
        gender_index = 1
    else:
        gender_index = 0 

    user_input.iloc[0] = {'income':salary_index, 'education': education_index, 'parent': parent_index, 'married': married_index, 'female': gender_index, 'age': age}

    submitted = st.form_submit_button("Submit")
    
    if submitted:
        y_user = log_model.predict(user_input)
        y_user_perc = log_model.predict_proba(user_input)[0,1]
        if y_user == 0:
            st.write(f'{user_name}, The model predicts that you have a {y_user_perc:.0%} chance of being a LinkedIn member and that you are NOT a LinkedIn User')
        else:
            st.write(f'{user_name}, The model predicts that you have a {y_user_perc:.0%} chance of being a LinkedIn member and that you ARE a LinkedIn User')