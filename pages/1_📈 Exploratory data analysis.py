import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read data files
survey2020 = pd.read_csv("../data/kaggle_survey_2020_responses.csv", skiprows=0, low_memory=False)
survey2021 = pd.read_csv("../data/kaggle_survey_2021_responses.csv", skiprows=0, low_memory=False)
survey2022 = pd.read_csv("../data/kaggle_survey_2022_responses.csv", skiprows=0, low_memory=False)

# 
def show_1():
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
    st.markdown("## 1.1 Trend over Time")
    st.markdown("### 1.1.1 Trend of Average Earnings")
    st.subheader('Average earnings by occupation in 2020-2022', divider='rainbow')

    #plt.xlabel(u'year',fontsize=14)
    #plt.ylabel(u'average income',fontsize=14)
    #sns.set_theme()
    #sns.lineplot(data=df_inner.T)
    st.line_chart(data=df_inner.T)
    #plt.legend(bbox_to_anchor=(1.3, 1))
    df_inner.T
    
    st.markdown("#### Data Preprocessing")
    st.write("In this visualization, we show the trend of salary in 2020-2022. Scope here is occupation.")
    st.write("We only retained data for those aged 18-35 to focus on the younger demographic.")
    st.write("We found that the occupation choices in 2022 survey are different from those in 2020 and 2021, so made the following integrations:")
    st.markdown("- Merge Data Analyst and Business Analyst to Business/Data Analyst")
    st.markdown("- Merge Program/Project Manager and Product Manager to Product/Project Manager")
    st.markdown("- Rename Machine Learning/ MLops Engineer to Machine Learning Engineer")
    st.markdown("- Rename Data Analyst (Business, Marketing, Financial, Quantitative, etc) to Business/Data Analyst")
    st.markdown("#### Observations")
    st.markdown("- The impacts of macroeconomic depression and recovery have been clearly reflected on a majority of the DS related job roles over the past two years, with decreased average incomes followed by surges. ")
    st.markdown("- The only anomaly was statisticians, which had experienced its negative impact in the year 2022.")
    st.markdown("- Most of the jobs’ average wages have recovered and surplused their pre-covid levels, except ‘Data engineer’, ‘Data/Business Analyst’ and ‘Statistician’.")
    st.markdown("- ‘Product/Project manager’ happened to be the most lucrative career path in 2020, and has also experienced the fastest growth over the past two years. ")



def show_2():
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
    st.markdown("## 1.1 Trend over Time")
    st.markdown("### 1.1.2 Trend of DS Job Demand")
    st.subheader('The average proportion of DS job demand in each country', divider='rainbow')
    #plt.xlabel(u'Country',fontsize=14)
    #plt.ylabel(u'Need Ratio',fontsize=14)
    #sns.set_theme()
    st.line_chart(data=df.T)
    df.T
    #plt.legend(bbox_to_anchor=(1.15, 1))
    st.markdown("#### Data Preprocessing")
    st.write("In this visualization, we show the trend of data science job demand in 2020-2022. Scope here is country.")
    st.write("There are two question in the surveys: *What is the size of the company where you are employed?* and *Approximately how many individuals are responsible for data science workloads at your place of business?* We use the ratio of these two features to represent the DS job demand of a company.")
    st.markdown("#### Observations")
    st.markdown("- DS related job demand has contracted in Brazil, China and India, especially in the year 2022.")
    st.markdown("- In Japan, large expansion from the bottom level, recovered and surplused its demand in 2020. ")
    st.markdown("- China has the largest job market among those five countries, however, it has also experienced a largest contraction over the past two years. ")



def show_3():
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
    
    # 1.2.1 Salary of occupations by coding experience
    # 2020
    col2020_1_2_1 = col2020.drop(columns=["Age", "Occupation_Number"])
    col2022_1_2_1 = col2022.drop(columns=["Age", "Occupation_Number"])

    col2020_1_2_1 = pd.merge(left= col2020_1_2_1, right=col2020_1_2_1.pivot(columns=["Coding Experience"], values="Salary"), how="inner", left_index=True, right_index=True)
    col2020_1_2_1.drop(columns=["Coding Experience", "Salary"], inplace=True)
    barplot_data2020 = col2020_1_2_1.groupby("Occupation").mean()
    barplot_data2020["< 3 years"] = (barplot_data2020["I have never written code"] + barplot_data2020["1-2 years"])/2
    barplot_data2020["3-10 years"] = (barplot_data2020["3-5 years"] + barplot_data2020["5-10 years"])/2
    barplot_data2020["10+ years"] = (barplot_data2020["10-20 years"] + barplot_data2020["20+ years"])/2
    barplot_data2020.drop(columns=["I have never written code", "< 1 years", "1-2 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"], inplace=True)


    # 2021
    col2021_1_2_1 = col2021.drop(columns=["Age", "Occupation_Number"])
    col2021_1_2_1 = pd.merge(left= col2021_1_2_1, right=col2021_1_2_1.pivot(columns=["Coding Experience"], values="Salary"), how="inner", left_index=True, right_index=True)
    col2021_1_2_1.drop(columns=["Coding Experience", "Salary"], inplace=True)
    barplot_data2021 = col2021_1_2_1.groupby("Occupation").mean()
    barplot_data2021["< 3 years"] = (barplot_data2021["I have never written code"] + barplot_data2021["1-3 years"])/2
    barplot_data2021["3-10 years"] = (barplot_data2021["3-5 years"] + barplot_data2021["5-10 years"])/2
    barplot_data2021["10+ years"] = (barplot_data2021["10-20 years"] + barplot_data2021["20+ years"])/2
    barplot_data2021.drop(columns=["I have never written code", "< 1 years", "1-3 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"], inplace=True)

    # 2022
    col2022_1_2_1 = col2022.drop(columns=["Age", "Occupation_Number"])
    col2022_1_2_1 = pd.merge(left= col2022_1_2_1, right=col2022_1_2_1.pivot(columns=["Coding Experience"], values="Salary"), how="inner", left_index=True, right_index=True)
    col2022_1_2_1.drop(columns=["Coding Experience", "Salary"], inplace=True)
    barplot_data2022 = col2022_1_2_1.groupby("Occupation").mean()
    barplot_data2022["< 3 years"] = (barplot_data2022["I have never written code"] + barplot_data2022["1-3 years"])/2
    barplot_data2022["3-10 years"] = (barplot_data2022["3-5 years"] + barplot_data2022["5-10 years"])/2
    barplot_data2022["10+ years"] = (barplot_data2022["10-20 years"] + barplot_data2022["20+ years"])/2
    barplot_data2022.drop(columns=["I have never written code", "< 1 years", "1-3 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"], inplace=True)
    
    col2020_1_2_1.rename(columns={"1-2 years": "1-3 years"}, inplace=True)


    st.markdown("## 1.2 Statistical Tendencies")
    st.markdown("### 1.2.1 Salary Tendency")
    st.subheader("Salary of Occupations by Coding Experience", divider='rainbow')

    # select by user
    from collections import Counter
    options = st.multiselect(
        'Please select your interested year(s)',
        ['2020', '2021', '2022'],
        default="2020")

    if options==["2020"]:
        st.bar_chart(barplot_data2020)
        barplot_data2020

    if options==["2021"]:
        st.bar_chart(barplot_data2021)
        barplot_data2021

    if options==["2022"]:
        st.bar_chart(barplot_data2022)
        barplot_data2022

    if (options==["2022", "2021"] or options==["2021", "2022"]):
        table12 = pd.concat([col2021_1_2_1, col2022_1_2_1])
        barplot_data12 = table12.groupby("Occupation").mean()
        barplot_data12["< 3 years"] = (barplot_data12["I have never written code"] + barplot_data12["1-3 years"])/2
        barplot_data12["3-10 years"] = (barplot_data12["3-5 years"] + barplot_data12["5-10 years"])/2
        barplot_data12["10+ years"] = (barplot_data12["10-20 years"] + barplot_data12["20+ years"])/2
        barplot_data12.drop(columns=["I have never written code", "< 1 years", "1-3 years", "3-5 years", "5-10 years",\
                                    "10-20 years", "20+ years"], inplace=True)
        st.bar_chart(barplot_data12)    
        barplot_data12
        
    if (options==["2020", "2021"] or options==["2021", "2020"]):
        table01 = pd.concat([col2020_1_2_1, col2021_1_2_1])
        barplot_data01 = table01.groupby("Occupation").mean()
        barplot_data01["< 3 years"] = (barplot_data01["I have never written code"] + barplot_data01["1-3 years"])/2
        barplot_data01["3-10 years"] = (barplot_data01["3-5 years"] + barplot_data01["5-10 years"])/2
        barplot_data01["10+ years"] = (barplot_data01["10-20 years"] + barplot_data01["20+ years"])/2
        barplot_data01.drop(columns=["I have never written code", "< 1 years", "1-3 years", "3-5 years", "5-10 years",\
                                    "10-20 years", "20+ years"], inplace=True)
        st.bar_chart(barplot_data01)    
        barplot_data01
        
    if (options==["2022", "2020"] or options==["2020", "2022"]):
        table02 = pd.concat([col2020_1_2_1, col2022_1_2_1])
        barplot_data02 = table02.groupby("Occupation").mean()
        barplot_data02["< 3 years"] = (barplot_data02["I have never written code"] + barplot_data02["1-3 years"])/2
        barplot_data02["3-10 years"] = (barplot_data02["3-5 years"] + barplot_data02["5-10 years"])/2
        barplot_data02["10+ years"] = (barplot_data02["10-20 years"] + barplot_data02["20+ years"])/2
        barplot_data02.drop(columns=["I have never written code", "< 1 years", "1-3 years", "3-5 years", "5-10 years",\
                                    "10-20 years", "20+ years"], inplace=True)
        st.bar_chart(barplot_data02)    
        barplot_data02
        
    if Counter(options)==Counter(["2020", "2021", "2022"]):
        table = pd.concat([col2020_1_2_1, col2022_1_2_1])
        table = pd.concat([table, col2021_1_2_1])
        barplot_data = table.groupby("Occupation").mean()
        barplot_data["< 3 years"] = (barplot_data["I have never written code"] + barplot_data["1-3 years"])/2
        barplot_data["3-10 years"] = (barplot_data["3-5 years"] + barplot_data["5-10 years"])/2
        barplot_data["10+ years"] = (barplot_data["10-20 years"] + barplot_data["20+ years"])/2
        barplot_data.drop(columns=["I have never written code", "< 1 years", "1-3 years", "3-5 years", "5-10 years",\
                                    "10-20 years", "20+ years"], inplace=True)
        st.bar_chart(barplot_data)     
        barplot_data
        
    st.markdown("#### Data Preprocessing")
    st.write("In this visualization, we want to show the salary levels for different positions and how coding experience affects the salary.")
    st.write("We applied the same integration to occupation as in 1.1.1, in order to maintain consistency of occupation choices over the three years.")
    st.markdown("#### Observations")
    st.markdown("- Overall, workers with longer coding experience earn higher wages for all job roles.")
    st.markdown("- ‘Machine Learning engineers’ typically require longer coding experiences in order to earn higher income, it has the lowest initial salaries. ")
    st.markdown("- In contrast, ‘Product/Project Manager’ requires less experience in coding for earning a decent amount of initial salary. ")



def show_4():
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
    data2020["Coding Experience"].replace({"1-2 years": "1-3 years"}, inplace=True)

    # prepare for heatmap
    recording_num2020 = data2020.shape[0]
    heatmap_data2020 = data2020.groupby("Coding Experience").sum().reset_index()
    heatmap_data2020["Coding Experience"] = pd.Categorical(heatmap_data2020["Coding Experience"], categories=["I have never written code", "< 1 years", "1-3 years", "3-5 years", "5-10 years", "10-20 years", "20+ years"], ordered=True)
    heatmap_data2020.sort_values(by="Coding Experience", inplace=True)
    heatmap_data2020.set_index("Coding Experience", inplace=True)
    heatmap_data2020 = heatmap_data2020 / recording_num2020

    heatmap_data2020.drop(columns=["Julia", "Swift", "None"], inplace=True)









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


    # for multi-year visualization
    heatmap_data2020["C#"] = 0.0
    heatmap_data2020["PHP"] = 0.0
    heatmap_data2021["C#"] = 0.0
    heatmap_data2021["PHP"] = 0.0

    heatmap_data01 = (heatmap_data2020 * recording_num2020 + heatmap_data2021 * recording_num2021) / (recording_num2020 + recording_num2021)
    heatmap_data02 = (heatmap_data2020 * recording_num2020 + heatmap_data2022 * recording_num2022) / (recording_num2020 + recording_num2022)
    heatmap_data12 = (heatmap_data2021 * recording_num2021 + heatmap_data2022 * recording_num2022) / (recording_num2021 + recording_num2022)
    heatmap_data012 = (heatmap_data2020 * recording_num2020 + heatmap_data2021 * recording_num2021 + heatmap_data2022 * recording_num2022) / (recording_num2020 + recording_num2021 + recording_num2022)

    heatmap_data01.drop(columns=["C#", "PHP"], inplace=True)
    heatmap_data2020.drop(columns=["C#", "PHP"], inplace=True)
    heatmap_data2021.drop(columns=["C#", "PHP"], inplace=True)
    order = heatmap_data2022.columns
    heatmap_data02 = heatmap_data02[order]
    heatmap_data12 = heatmap_data12[order]
    heatmap_data012 = heatmap_data12[order]




    f2020, ax = plt.subplots(figsize = (14, 10))
    #plt.title('The relationship between the code languages and coding experience',fontsize=20)
    cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    #plt.xlabel(u'code languages',fontsize=14)
    sns.heatmap(data= heatmap_data2020, fmt=".0%", cmap="RdBu", annot=True, linewidths = 0.05, ax = ax)
    ax.set_xlabel("code languages",fontsize=15)
    ax.set_ylabel("coding experience",fontsize=15)
    ax.tick_params(axis="x",labelsize=12)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_title('YEAR 2020', fontsize=18, position=(0.5,1.05))



    f2021, ax = plt.subplots(figsize = (14, 10))
    #plt.title('The relationship between the code languages and coding experience',fontsize=20)
    cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    #plt.xlabel(u'code languages',fontsize=14)
    sns.heatmap(data= heatmap_data2021, fmt=".0%", cmap="RdBu", annot=True, linewidths = 0.05, ax = ax)
    ax.set_xlabel("code languages",fontsize=15)
    ax.set_ylabel("coding experience",fontsize=15)
    ax.tick_params(axis="x",labelsize=12)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_title('YEAR 2021', fontsize=18, position=(0.5,1.05))




    f2022, ax = plt.subplots(figsize = (14, 10))
    #plt.title('The relationship between the code languages and coding experience',fontsize=20)
    cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    #plt.xlabel(u'code languages',fontsize=14)
    sns.heatmap(data= heatmap_data2022, fmt=".0%", cmap="RdBu", annot=True, linewidths = 0.05, ax = ax)
    ax.set_xlabel("code languages",fontsize=15)
    ax.set_ylabel("coding experience",fontsize=15)
    ax.tick_params(axis="x",labelsize=12)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_title('YEAR 2022', fontsize=18, position=(0.5,1.05))


    f01, ax = plt.subplots(figsize = (14, 10))
    #plt.title('The relationship between the code languages and coding experience',fontsize=20)
    cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    #plt.xlabel(u'code languages',fontsize=14)
    sns.heatmap(data= heatmap_data01, fmt=".0%", cmap="RdBu", annot=True, linewidths = 0.05, ax = ax)
    ax.set_xlabel("code languages",fontsize=15)
    ax.set_ylabel("coding experience",fontsize=15)
    ax.tick_params(axis="x",labelsize=12)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_title('YEAR 2022', fontsize=18, position=(0.5,1.05))


    f02, ax = plt.subplots(figsize = (14, 10))
    #plt.title('The relationship between the code languages and coding experience',fontsize=20)
    cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    #plt.xlabel(u'code languages',fontsize=14)
    sns.heatmap(data= heatmap_data02, fmt=".0%", cmap="RdBu", annot=True, linewidths = 0.05, ax = ax)
    ax.set_xlabel("code languages",fontsize=15)
    ax.set_ylabel("coding experience",fontsize=15)
    ax.tick_params(axis="x",labelsize=12)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_title('YEAR 2022', fontsize=18, position=(0.5,1.05))



    f12, ax = plt.subplots(figsize = (14, 10))
    #plt.title('The relationship between the code languages and coding experience',fontsize=20)
    cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    #plt.xlabel(u'code languages',fontsize=14)
    sns.heatmap(data= heatmap_data12, fmt=".0%", cmap="RdBu", annot=True, linewidths = 0.05, ax = ax)
    ax.set_xlabel("code languages",fontsize=15)
    ax.set_ylabel("coding experience",fontsize=15)
    ax.tick_params(axis="x",labelsize=12)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_title('YEAR 2022', fontsize=18, position=(0.5,1.05))



    f012, ax = plt.subplots(figsize = (14, 10))
    #plt.title('The relationship between the code languages and coding experience',fontsize=20)
    cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=0.8, as_cmap = True)
    #plt.xlabel(u'code languages',fontsize=14)
    sns.heatmap(data= heatmap_data012, fmt=".0%", cmap="RdBu", annot=True, linewidths = 0.05, ax = ax)
    ax.set_xlabel("code languages",fontsize=15)
    ax.set_ylabel("coding experience",fontsize=15)
    ax.tick_params(axis="x",labelsize=12)
    ax.tick_params(axis="y",labelsize=12)
    ax.set_title('YEAR 2022', fontsize=18, position=(0.5,1.05))

    st.markdown("## 1.2 Statistical Tendencies")
    st.markdown("### 1.2.2 Coding Lauunguages and Coding Experience Tendency")
    st.subheader('The relationship between the code languages and coding experience', divider="rainbow")

    with st.container():
        start, end = st.select_slider(
            'Select a range of time',
            options=['2020', '2021', '2022'],
            value=('2020', '2021'))
        if start == "2020" and end == "2020":
            st.pyplot(f2020)
            heatmap_data2020
        elif start == "2021" and end == "2021":
            st.pyplot(f2021)
            heatmap_data2021
        elif start == "2022" and end == "2022":
            st.pyplot(f2022)
            heatmap_data2022
        elif start == "2020" and end == "2021":
            st.pyplot(f01)
            heatmap_data01
        elif start == "2021" and end == "2022":
            st.pyplot(f12)
            heatmap_data12
        elif start == "2020" and end == "2022":
            st.pyplot(f012)
            heatmap_data012
            
    st.markdown("#### Observations:")
    st.markdown("- Python is the most employable tool across all experience levels, followed by SQL.")
    st.markdown("- Python is mostly used  by newbies, while SQL is mostly adopted for employees with a certain level of experience. ")


st.set_page_config(page_title="Exploratory Data Analysis", page_icon="📈")

st.sidebar.header("Exploratory Data Analysis")

user_choose = st.sidebar.radio(
    "We made four visualizations, the former two show some trend over time, the former two show statistical tendencies. Please choose your interest.",
    ["***Average Earnings by Occupation***", 
     "***Average Proportion of DS Job Demand***",
     "***Salary of Occupations by Coding Experience***",
     "***Coding Experience vs Coding Language***"],
    captions = ["Line Plot", "Line Plot", "Bar Plot", "Heat Map"]
)

if(user_choose == "***Average Earnings by Occupation***"):
    show_1()
elif(user_choose == "***Average Proportion of DS Job Demand***"):
    show_2()
elif(user_choose == "***Salary of Occupations by Coding Experience***"):
    show_3()
elif(user_choose == "***Coding Experience vs Coding Language***"):
    show_4()