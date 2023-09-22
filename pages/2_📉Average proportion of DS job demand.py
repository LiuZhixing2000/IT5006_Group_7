import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
def get_median(numbers):
    if len(numbers) > 2 or len(numbers) <= 0:
        return None
    numbers[0] = numbers[0].replace(',', '')
    if len(numbers) == 1:
        return float(numbers[0])
    numbers[1] = numbers[1].replace(',', '')
    return (float(numbers[0]) + float(numbers[1])) / 2
def get_interval_median(row):
    row = row.apply(lambda x: re.findall(r'\d+\,\d+|\d+', str(x)))
    ans = row.apply(get_median)
    return ans

# read data files
survey2020 = pd.read_csv("../data/kaggle_survey_2020_responses.csv", skiprows=0, low_memory=False)
survey2021 = pd.read_csv("../data/kaggle_survey_2021_responses.csv", skiprows=0, low_memory=False)
survey2022 = pd.read_csv("../data/kaggle_survey_2022_responses.csv", skiprows=0, low_memory=False)

# 1.1.2 The need for data roles over 2020-2023
df2020 = survey2020.loc[1:,['Q3','Q20', 'Q21']]
df2021 = survey2021.loc[1:,['Q3', 'Q21', 'Q22']]
df2022 = survey2022.loc[1:,['Q4', 'Q25', 'Q26']]
df2020 = df2020.dropna()
df2021 = df2021.dropna()
df2022 = df2022.dropna()
# filter 国家（“Brazil”, "India", "US", "Japan", "China"）Maria
countries_to_filter = ['Brazil', 'India', 'United States of America', 'Japan', 'China']

# Filter the rows
df2020 = df2020[df2020['Q3'].isin(countries_to_filter)]
df2021 = df2021[df2021['Q3'].isin(countries_to_filter)]
df2022 = df2022[df2022['Q4'].isin(countries_to_filter)]

# 公司人口 type transformation into numeric 范围取中点 航宝
df2020["Q20"] = get_interval_median(df2020["Q20"])
df2021["Q21"] = get_interval_median(df2021["Q21"])
df2022["Q25"] = get_interval_median(df2022["Q25"])

# 岗位需求 type transformation into numeric 范围取最小值 Eugenia
df2020["Q21"] = pd.to_numeric(df2020["Q21"].str.extract(r'(\d*)')[0])
df2021["Q22"] = pd.to_numeric(df2021["Q22"].str.extract(r'(\d*)')[0])
df2022["Q26"] = pd.to_numeric(df2022["Q26"].str.extract(r'(\d*)')[0])

# Compute Ratio = #employee/#公司规模 （新建一列）Eugenia
df2020["2020 Ratio"] = df2020["Q21"] / df2020["Q20"]
df2021["2021 Ratio"] = df2021["Q22"] / df2021["Q21"]
df2022["2022 Ratio"] = df2022["Q26"] / df2022["Q25"]

# rename columns for readability
df2020.rename(columns={"Q3": "Country", "Q20": "Employees", "Q21": "Needs"}, inplace=True)
df2021.rename(columns={"Q3": "Country", "Q21": "Employees", "Q22": "Needs"}, inplace=True)
df2022.rename(columns={"Q4": "Country", "Q25": "Employees", "Q26": "Needs"}, inplace=True)

plot_df2020 = df2020.groupby("Country").mean().T
plot_df2021 = df2021.groupby("Country").mean().T
plot_df2022 = df2022.groupby("Country").mean().T

# plot
df = pd.concat([plot_df2020, plot_df2021, plot_df2022])
df = df.iloc[[2, 5, 8], :]
#plt.figure(figsize=(14,6))
#plt.title('The average proportion of DS job demand in each country',fontsize=20)
st.subheader('The average proportion of DS job demand in each country', divider='rainbow')
#plt.xlabel(u'Country',fontsize=14)
#plt.ylabel(u'Need Ratio',fontsize=14)
#sns.set_theme()
st.line_chart(data=df.T)
#plt.legend(bbox_to_anchor=(1.15, 1))