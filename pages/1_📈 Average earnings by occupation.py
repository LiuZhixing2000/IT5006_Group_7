import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data files
survey2020 = pd.read_csv("../data/kaggle_survey_2020_responses.csv", skiprows=0, low_memory=False)
survey2021 = pd.read_csv("../data/kaggle_survey_2021_responses.csv", skiprows=0, low_memory=False)
survey2022 = pd.read_csv("../data/kaggle_survey_2022_responses.csv", skiprows=0, low_memory=False)


# 1.1.1 Average earning by occupation in 2020-2022
# get data
# keep the occupation & salary
col2020 = survey2020.loc[:,['Q1','Q5', 'Q6', 'Q24']]
col2021 = survey2021.loc[:,['Q1', 'Q5', 'Q6', 'Q25']]
col2022 = survey2022.loc[:,['Q2', 'Q11', 'Q23', 'Q29']]

# 先对年龄离散处理 => filter 年龄  (18-35) Eugenia
low_year = 18
high_year = 35
col2020["Q1"] = pd.to_numeric(col2020["Q1"].str.extract(r'(\d*)[\s-]*(\d*)')[0])
col2021["Q1"] = pd.to_numeric(col2021["Q1"].str.extract(r'(\d*)[\s-]*(\d*)')[0])
col2022["Q2"] = pd.to_numeric(col2022["Q2"].str.extract(r'(\d*)[\s-]*(\d*)')[0])
col2020 = col2020[col2020.Q1.between(low_year, high_year)]
col2021 = col2021[col2021.Q1.between(low_year, high_year)]
col2022 = col2022[col2022.Q2.between(low_year, high_year)]

# salary discretisation （取中点）航宝
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
col2020["Q24"] = get_interval_median(col2020["Q24"])
col2021["Q25"] = get_interval_median(col2021["Q25"])
col2022["Q29"] = get_interval_median(col2022["Q29"])

# missing values删掉, outliers(salary)=> boxplot => stats Maria
col2020 = col2020.dropna()
col2021 = col2021.dropna()
col2022 = col2022.dropna()

# rename columns for readability
col2020.rename(columns={"Q1": "Age", "Q5": "Occupation", "Q6": "Coding Experience", "Q24": "Salary"}, inplace=True)
col2021.rename(columns={"Q1": "Age", "Q5": "Occupation", "Q6": "Coding Experience", "Q25": "Salary"}, inplace=True)
col2022.rename(columns={"Q2": "Age", "Q23": "Occupation", "Q11": "Coding Experience", "Q29": "Salary"}, inplace=True)

#Set yearly compensation max at $150,000
col2020['Salary'] = col2020['Salary'].where(col2020['Salary'] <= 150000, 150000)
col2021['Salary'] = col2021['Salary'].where(col2021['Salary'] <= 150000, 150000)
col2022['Salary'] = col2022['Salary'].where(col2022['Salary'] <= 150000, 150000)

#Occupartion 合并
##筛选2020的occupation
#1. delete rows with occupation  'DBA/Database Engineer', and 'Other'
col2020 = col2020[~col2020['Occupation'].isin(['DBA/Database Engineer', 'Other'])]

#2. 重新给Business Analyst 和Data Analyst 命名为‘Data/Business Analyst’
col2020['Occupation'] = col2020['Occupation'].replace('Business Analyst', 'Data/Business Analyst')
col2020['Occupation'] = col2020['Occupation'].replace('Data Analyst', 'Data/Business Analyst')

#3. 给每个occupation编号
# List of unique occupations after removal
occupations = col2020['Occupation'].unique()

# Create a dictionary that maps each occupation to a unique number
occupation_map = {occupation: i+1 for i, occupation in enumerate(occupations)}

# Add a new column to the DataFrame
col2020['Occupation_Number'] = col2020['Occupation'].map(occupation_map)


##筛选2021的occupation
#1. delete rows with occupation  'DBA/Database Engineer', 'Others', 'Developer Relations/Advocacy'
col2021 = col2021[~col2021['Occupation'].isin(['DBA/Database Engineer', 'Other','Developer Relations/Advocacy',])]

#2. 重新给Program/Project Manager和Product Manager 命名为‘Product/Project Manager’
col2021['Occupation'] = col2021['Occupation'].replace('Program/Project Manager', 'Product/Project Manager')
col2021['Occupation'] = col2021['Occupation'].replace('Product Manager', 'Product/Project Manager')
#2. 重新给Business Analyst 和Data Analyst 命名为‘Data/Business Analyst’
col2021['Occupation'] = col2021['Occupation'].replace('Business Analyst', 'Data/Business Analyst')
col2021['Occupation'] = col2021['Occupation'].replace('Data Analyst', 'Data/Business Analyst')


# 3. Using the 2020 created occupation_map, 使occupation编号一样
col2021['Occupation_Number'] = col2021['Occupation'].map(occupation_map)

##筛选2022的occupation
#1. delete rows with occupation'Teacher / professor','Data Architect','Developer Advocate','Data Administrator‘,'others','Engineer (non-software)',
col2022 = col2022[~col2022['Occupation'].isin(['Data Architect', 'Other','Developer Advocate','Teacher / professor','Data Administrator', 'Engineer (non-software)'])]

#2. 重新给Manager命名为‘Product/Project Manager’
col2022['Occupation'] = col2022['Occupation'].replace('Manager (Program, Project, Operations, Executive-level, etc)', 'Product/Project Manager')

#2. 重新给'Machine Learning/ MLops Engineer'命名为'Machine Learning Engineer'
col2022['Occupation'] = col2022['Occupation'].replace('Machine Learning/ MLops Engineer', 'Machine Learning Engineer')


#2. 重新给Data Analyst (Business, Marketing, Financial, Quantitative, etc)命名为‘Data/Business Analyst’
col2022['Occupation'] = col2022['Occupation'].replace('Data Analyst (Business, Marketing, Financial, Quantitative, etc)', 'Data/Business Analyst')


# 3. Using the 2020 created occupation_map, 使occupation编号一样
col2022['Occupation_Number'] = col2022['Occupation'].map(occupation_map)

# Eugenia
# calculate the average salary by each year
col2020_1_1_1 = col2020.groupby("Occupation").mean(numeric_only=True)
col2021_1_1_1 = col2021.groupby("Occupation").mean(numeric_only=True)
col2022_1_1_1 = col2022.groupby("Occupation").mean(numeric_only=True)

# merge data by three years
df_inner = pd.merge(col2020_1_1_1, col2021_1_1_1, how='inner', left_index=True, right_index=True)
df_inner = pd.merge(df_inner, col2022_1_1_1, how='inner', left_index=True, right_index=True)
df_inner.rename(columns={"Salary_x" : "2020 Salary", "Salary_y" : "2021 Salary", "Salary": "2022 Salary"}, inplace=True)

df_inner = df_inner[["2020 Salary", "2021 Salary", "2022 Salary"]]
#plt.figure(figsize=(14,6))
#plt.title('Average earnings by occupation in 2020-2022',fontsize=20)
st.subheader('Average earnings by occupation in 2020-2022', divider='rainbow')

#plt.xlabel(u'year',fontsize=14)
#plt.ylabel(u'average income',fontsize=14)
#sns.set_theme()
#sns.lineplot(data=df_inner.T)
st.line_chart(data=df_inner.T)
#plt.legend(bbox_to_anchor=(1.3, 1))