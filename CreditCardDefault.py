import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import hiplot as hip
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import plotly.io as pio
from PIL import Image
import streamlit.components.v1 as components

#add navigation sidebar
st.sidebar.title("🔎Explore")
page = st.sidebar.selectbox("Select a page:", ["🏡Home", "👪Demographic Data", "💳Credit Limit & Balance", "📊Macroeconomic Factors","🤑Sep Defaulters"], index=0)
for _ in range(15):  # Change 10 to the number of empty lines you want
    st.sidebar.write("")
st.sidebar.write("Find code at: https://github.com/mahnoorsheikh16/FDS-Project-CMSE830-Credit-Card-Default-Prediction")

#import data
data = pd.read_csv(r"C:\Users\manos\OneDrive\Desktop\MSU\Fall 2024\CMSE 830 Foundations of Data Science\Project\Midterm\UCI_Credit_Card.csv")
data_macro = pd.read_excel(r"C:\Users\manos\OneDrive\Desktop\MSU\Fall 2024\CMSE 830 Foundations of Data Science\Project\Midterm\data_macro.xlsx")
data_income = pd.read_excel(r"C:\Users\manos\OneDrive\Desktop\MSU\Fall 2024\CMSE 830 Foundations of Data Science\Project\Midterm\data_income.xlsx")

data.rename(columns={'default.payment.next.month': 'Default'}, inplace=True)
data.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)

#set page content
if page == "🏡Home":
    st.title('Beyond the Client')
    st.header("Enhancing Credit Card Default Prediction with Behavioral Analysis and Macroeconomic Data")
    st.write("")
    st.write("")
    st.write("The Bank of Taiwan is facing losses due to increasing credit card defaults. This dashboard provides a means to identify factors associated with credit defaults and help identify potential faulty clients to curb losses.")
    st.write("")
    st.write("*PS. Imagine you are an employee of the Bank of Taiwan in Sep 2005. Profits are low, customers are defaulting, your boss is angry, and he wants the answer to 'WHY' asap.*")

elif page == "👪Demographic Data":
    st.subheader("Defaults in relation to Gender, Relationship Status, Age & Education Level")
    st.write("")
    st.write("")
    sex_data = filtered_data = data_income[data_income['Demographic'].isin(['Male', 'Female'])]
    fig2 = px.pie(sex_data, values='defaults', names='Demographic', title='Defaults by Sex')
    st.plotly_chart(fig2)
    edu_data = filtered_data = data_income[data_income['Demographic'].isin(['High school', 'University', 'Graduate school', 'others'])]
    fig3 = px.pie(edu_data, values='defaults', names='Demographic', title='Defaults by Education Level')
    st.plotly_chart(fig3)
    age_data = filtered_data = data_income[data_income['Demographic'].isin(['Under 30 yrs','30-34 yrs','35-39 yrs','40-44 yrs','45-54 yrs','65 yrs and above'])]
    fig4 = px.pie(age_data, values='defaults', names='Demographic', title='Defaults by Age')
    st.plotly_chart(fig4)
    marriage_data = data.groupby(['MARRIAGE', 'Default']).size().reset_index(name='Count')
    fig5 = px.pie(marriage_data, values='Count', names='MARRIAGE', title='Defaults by Relationship Status')
    st.plotly_chart(fig5)
    
elif page == "💳Credit Limit & Balance":
    st.subheader("Defaults in relation to Credit Limit and Monthly Repayment History")
    st.write("")
    st.write("")
    st.write("**HiPlot Visualization of Payment History**")
    st.write("KEY: -2:No payment due, -1=Paid duly, 0=Payment delay <1 month, 1=Payment delay of 1 month, 2=Payment delay of 2 months, … , 9=Payment delay of nine months and above")
    st.write("PAY_1:Repayment status in Sept, PAY_2:Repayment status in August, ... so on")
    with open('hiplot.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    components.html(html_content, height=800, scrolling=True)
    st.write("")
    st.write("")
    with open("correlation_heatmap.json", "r") as f:
        fig1_json = f.read()
        fig1 = pio.from_json(fig1_json)
    st.plotly_chart(fig1, use_container_width=True)
    st.write("")
    st.write("")
    image = Image.open("density_plot.png")
    st.image(image, width=500)


elif page == "📊Macroeconomic Factors":
    st.subheader("Defaults in relation to Unemployment Rate and Inflation")
    st.write("")
    st.write("")
    st.write("Choose a factor")
    checkbox_unemp = st.checkbox("Unemployment Rate", value=False)
    checkbox_inf = st.checkbox("Inflation Rate", value=False)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=data_macro["Month"], y=data_macro["defaults"], mode='lines', name='Defaults',line=dict(color='orange')),secondary_y=True)
    if checkbox_unemp:
        fig.add_trace(go.Scatter(x=data_macro["Month"], y=data_macro["Unemployment Rate"], mode='lines', name='Unemployment Rate',line=dict(color='green')),secondary_y=False)
    elif checkbox_inf:
        fig.add_trace(go.Scatter(x=data_macro["Month"], y=data_macro["CPI"], mode='lines', name='Inflation Rate',line=dict(color='green')),secondary_y=False)
    fig.update_layout(xaxis_title='Month',yaxis_title='Rate',yaxis2_title='Defaults')
    st.plotly_chart(fig, use_container_width=True)

elif page == "🤑Sep Defaulters":
    #show raw table of those who will default
    st.subheader("Customers Predicted to Default Next Month")
    st.write("")
    st.write("")
    data1 = data[data['Default'] == 'yes']
    rows_per_page = 10
    total_pages = (len(data1) // rows_per_page) + (1 if len(data1) % rows_per_page > 0 else 0)
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0
    start_row = st.session_state.current_page * rows_per_page
    end_row = start_row + rows_per_page
    if start_row < len(data1):
        st.dataframe(data1.iloc[start_row:end_row])
    st.write("Page:", st.session_state.current_page + 1, "of", total_pages)
    if st.session_state.current_page > 0:
        if st.button("Previous"):
            st.session_state.current_page -= 1
    if st.session_state.current_page < total_pages - 1:
        if st.button("Next"):
            st.session_state.current_page += 1

