import streamlit as st

st.set_page_config(page_title="Welcome", page_icon="👋")

st.title("Welcome!")

st.write("This is a recommendation application for people who are looking for DS and ML related jobs. \n")

st.write("You can firstly look for trend and statistical tendencies over the past three years, find some trend over time and statistical tendencies.")

st.write("Then we prepare a job role recommendation for you. Fill in your related personal/professional profiles accurately in our survey, our system will recommend the right job for you based on the dataset!")

st.write("The visualization and recommendation are based on the data from [Kaggle’s Annual Machine Learning and Data Science Survey](https://www.kaggle.com/c/kaggle-survey-2020/) (2020-2022).")