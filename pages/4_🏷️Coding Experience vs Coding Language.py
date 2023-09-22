import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data files
survey2020 = pd.read_csv("../data/kaggle_survey_2020_responses.csv", skiprows=0, low_memory=False)
survey2021 = pd.read_csv("../data/kaggle_survey_2021_responses.csv", skiprows=0, low_memory=False)
survey2022 = pd.read_csv("../data/kaggle_survey_2022_responses.csv", skiprows=0, low_memory=False)

# 1.2.2 Coding language vs coding experience

# select features of 1.2.2 coding languages vs. coding experience

# for 2020 data
# get data
data2020 = survey2020.loc[:, ["Q6"]]
data2020 = pd.concat([data2020, survey2020.filter(like="Q7_")], axis=1)
data2020.drop(0, inplace=True)
# Q6
data2020.rename(columns={"Q6": "Coding Experience"}, inplace=True)
# Q7
def preprocess_coding_languages(obj, language_name):
    if obj == language_name:
        return 1
    return 0
languages_name_table = {
    "Q7_Part_1" : "Python",
    "Q7_Part_2" : "R",
    "Q7_Part_3" : "SQL",
    "Q7_Part_4" : "C",
    "Q7_Part_5" : "C++",
    "Q7_Part_6" : "Java",
    "Q7_Part_7" : "Javascript",
    "Q7_Part_8" : "Julia",
    "Q7_Part_9" : "Swift",
    "Q7_Part_10" : "Bash",
    "Q7_Part_11" : "MATLAB",
    "Q7_Part_12" : "None",
    "Q7_OTHER" : "Other"
}
for title, language_name in languages_name_table.items():
    data2020[title] = data2020[title].apply(preprocess_coding_languages, args=(language_name, ))
    data2020.rename(columns={title: language_name}, inplace=True)
# drop
data2020 = data2020[data2020["Coding Experience"] != "I have never written code"]
data2020.dropna(inplace=True)

# prepare for heatmap
recording_num2020 = data2020.shape[0]
heatmap_data2020 = data2020.groupby("Coding Experience").sum().reset_index()
heatmap_data2020["Coding Experience"] = pd.Categorical(heatmap_data2020["Coding Experience"], categories=["I have never written code", "< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"], ordered=True)
heatmap_data2020.sort_values(by="Coding Experience", inplace=True)
heatmap_data2020.set_index("Coding Experience", inplace=True)
heatmap_data2020 = heatmap_data2020 / recording_num2020

heatmap_data2020.drop(columns=["Julia", "Swift", "None"], inplace=True)

plt.figure(figsize=(10,8))
heatmap_2020 = sns.heatmap(data= heatmap_data2020, fmt=".0%", cmap="crest", annot=True).figure

# same to 2021 and 2022
# 2021
data2021 = survey2021.loc[:, ["Q6"]]
data2021 = pd.concat([data2021, survey2021.filter(like="Q7_")], axis=1)
data2021.drop(0, inplace=True)
# Q6
data2021.rename(columns={"Q6": "Coding Experience"}, inplace=True)
# Q7
for title, language_name in languages_name_table.items():
    data2021[title] = data2021[title].apply(preprocess_coding_languages, args=(language_name, ))
    data2021.rename(columns={title: language_name}, inplace=True)
# drop
data2021 = data2021[data2021["Coding Experience"] != "I have never written code"]
data2021.dropna(inplace=True)

recording_num2021 = data2021.shape[0]
heatmap_data2021 = data2021.groupby("Coding Experience").sum().reset_index()
heatmap_data2021["Coding Experience"] = pd.Categorical(heatmap_data2021["Coding Experience"], categories=["< 1 years", "1-3 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"], ordered=True)
heatmap_data2021.sort_values(by="Coding Experience", inplace=True)
heatmap_data2021.set_index("Coding Experience", inplace=True)
heatmap_data2021 = heatmap_data2021 / recording_num2021

heatmap_data2021.drop(columns=["Julia", "Swift", "None"], inplace=True)

plt.figure(figsize=(10,8))
heatmap_2021 = sns.heatmap(data= heatmap_data2021, fmt=".0%", cmap="crest", annot=True).figure

# 2022
# Notice: the coding languages are different in 2022's survey
data2022 = survey2022.loc[:, ["Q11"]]
data2022 = pd.concat([data2022, survey2022.filter(like="Q12_")], axis=1)
data2022.drop(0, inplace=True)
# Q11
data2022.rename(columns={"Q11": "Coding Experience"}, inplace=True)
# Q12
languages_name_table = {
    "Q12_1" : "Python",
    "Q12_2" : "R",
    "Q12_3" : "SQL",
    "Q12_4" : "C",
    "Q12_5": "C#",
    "Q12_6" : "C++",
    "Q12_7" : "Java",
    "Q12_8" : "Javascript",
    "Q12_9" : "Bash",
    "Q12_10" : "PHP",
    "Q12_11" : "MATLAB",
    "Q12_12" : "Julia",
    "Q12_13" : "Go",
    "Q12_14" : "None",
    "Q12_15" : "Other"
}
for title, language_name in languages_name_table.items():
    data2022[title] = data2022[title].apply(preprocess_coding_languages, args=(language_name, ))
    data2022.rename(columns={title: language_name}, inplace=True)
# drop
data2022 = data2022[data2022["Coding Experience"] != "I have never written code"]
data2022.dropna(inplace=True)

recording_num2022 = data2022.shape[0]
heatmap_data2022 = data2022.groupby("Coding Experience").sum().reset_index()
heatmap_data2022["Coding Experience"] = pd.Categorical(heatmap_data2022["Coding Experience"], categories=["< 1 years", "1-3 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"], ordered=True)
heatmap_data2022.sort_values(by="Coding Experience", inplace=True)
heatmap_data2022.set_index("Coding Experience", inplace=True)
heatmap_data2022 = heatmap_data2022 / recording_num2022

heatmap_data2022.drop(columns=["Julia", "Go", "None"], inplace=True)

plt.figure(figsize=(10,8))
heatmap_2022 = sns.heatmap(data= heatmap_data2022, fmt=".0%", cmap="crest", annot=True).figure






st.header("1.2.2 Coding Experience vs Coding Language")

with st.container():
    option = st.selectbox(
        "Please select a year:",
        ("2020", "2021", "2022")
    )
    if option == "2020":
        st.pyplot(heatmap_2020)
    elif option == "2021":
        st.pyplot(heatmap_2021)
    elif option == "2022":
        st.pyplot(heatmap_2022)