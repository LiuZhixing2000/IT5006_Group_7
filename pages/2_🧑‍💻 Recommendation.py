import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing

st.set_page_config(page_title="Recommendation", page_icon="🧑‍💻")
st.sidebar.header("Recommendation")
st.markdown("## Recommendation")

job_icon_map = {
    "Data Engineer": "./pictures/de_icon.png",
    "Software Engineer": "./pictures/se_icon.png",
    "Data Scientist": "./pictures/ds_icon.png",
    "Research Scientist": "./pictures/rs_icon.jpg",
    "Statistician": "./pictures/st_icon.png",
    "Data/Business Analyst": "./pictures/ba_icon.png",
    "Machine Learning Engineer": "./pictures/mle_icon.png",
    "Product/Project Manager": "./pictures/pm_icon.png"
}

job_description_map = {
    "Data Engineer": "- **Job Description**: A data engineer is responsible for collecting, managing, and converting raw data into information that can be interpreted by data scientists and business analysts.\
                    It is a broad field with applications in just about every industry. \
                    Click [here](https://www.coursera.org/articles/what-does-a-data-engineer-do-and-how-do-i-become-one) to know more.",
    "Software Engineer": "- **Job Description**: Software engineers, sometimes called software developers, create software for computers and applications. \
                    In addition to building their own systems, software engineers also test, improve, and maintain software built by other engineers. \
                    Click [here](https://www.coursera.org/articles/software-engineer) to know more.",
    "Data Scientist": "- **Job Description**: A data scientist uses data to understand and explain the phenomena around them, and help organizations make better decisions.\
                    Data scientists have become more common and in demand, as big data continues to be increasingly important to the way organizations make decisions. \
                    Click [here](https://www.coursera.org/articles/what-is-a-data-scientist) to know more.",
    "Research Scientist": "- **Job Description**: A research scientist is someone who asks research questions, develops hypotheses and designs research studies and experiments to find answers to scientific questions. \
                    They often hold Master's or doctorate degrees in their subject, providing them with the knowledge and skills they need to design effective studies. \
                    Click [here](https://www.srgtalent.com/blog/what-does-a-research-scientist-do-and-how-do-i-become-one) to know more.",
    "Statistician": "- **Job Description**: Statisticians are experts who compile and analyze statistical data in order to solve problems for businesses, government organizations, and other institutions.\
                    In their day to day work, statisticians determine the data a company will require to solve a problem and then apply mathematical theories to use that data to construct a solution. \
                    In many cases, they also source the data for companies by designing surveys, questionnaires, experiments, and polls. \
                    Click [here](https://www.coursera.org/articles/statistician) to know more.",
    "Data/Business Analyst": "- **Job Description**: Data analysts collect, clean, and interpret data sets in order to answer a question or solve a problem. \
                    Business analysts use data to form business insights and recommend changes in businesses and other organizations. \
                    Click [here](https://www.coursera.org/articles/data-analyst-vs-business-analyst) to know more.",
    "Machine Learning Engineer": "- **Job Description**: Machine learning engineers work with algorithms, data, and artificial intelligence. \
                    Machine learning is a fascinating branch of artificial intelligence that involves predicting and adapting outcomes as more data is received. \
                    The demand for machine learning professionals has also grown exponentially in recent years. \
                    Click [here](https://www.coursera.org/articles/what-is-machine-learning-engineer) to know more.",
    "Product/Project Manager": "- **Job Description**: A product manager (PM) is a professional role that is responsible for the development of products for an organization, known as the practice of product management. \
                    A project manager has the responsibility of the planning, procurement and execution of a project. \
                    Product/project managers coordinate work done by many other functions (like software engineers, data scientists, and product designers), and are ultimately responsible for product outcomes. \
                    Click [here](https://www.coursera.org/articles/product-manager-vs-project-manager) to know more."
}

job_activities_map = {
    "Data Engineer": "- **Daily Working Avtivities**: \
                    Data ingestion, processing, storage, and modeling; \
                    build and maintain ETL(Extract, Transform, Load) processes to move data between systems, ensuring data quality and consistency; \
                    integrate data from different sources to create a unified and comprehensive view, ensure interoperability between various data systems; \
                    data quality and security assurance.",
    "Software Engineer": "- **Daily Working Avtivities**: \
                    create software designs and architectures that meet project specifications, \
                    write and implement code based on the design specifications, use programming languages such as Python, Java, JavaScript, C++, or others; \
                    participate in and conduct code reviews to ensure code quality, adherence to coding standards, and knowledge sharing within the team; \
                    use version control systems (e.g., Git) to manage and track changes to the codebase, collaborate with team members through branching, merging, and resolving merge conflicts.",
    "Data Scientist": "- **Daily Working Avtivities**: \
                    Conduct exploratory data analysis to understand the patterns, relationships, and distributions within the data; \
                    create new features or transform existing ones to improve the performance of machine learning models; \
                    develop and implement machine learning models to solve specific business problems; \
                    iterate on the model-building process, adjusting parameters, trying different algorithms, and refining the approach to improve model performance.",
    "Research Scientist": "- **Daily Working Avtivities**: \
                    Review relevant scientific literature to stay informed about the latest advancements in the field and understand the existing knowledge base; \
                    design experiments or research studies to address specific research questions or hypotheses, conduct experiments, gather data, and do some research analysis based on the data; \
                    use statistical methods to test hypotheses and determine the statistical significance of research findings, develop and use mathematical models or simulations to explore and understand complex phenomena in the field of study.",
    "Statistician": "- **Daily Working Avtivities**: \
                    Determine the data a company will require to solve a problem and then apply mathematical theories to use that data to construct a solution; \
                    conduct statistical analyses using appropriate methods and tools, this may involve descriptive statistics, inferential statistics, regression analysis, or other advanced statistical techniques; \
                    develop statistical models to predict outcomes, understand relationships between variables, or identify patterns in data; \
                    create visualizations, such as charts, graphs, or dashboards, to communicate statistical findings effectively to non-technical stakeholders.",
    "Data/Business Analyst": "- **Daily Working Avtivities**: \
                    Data collection, cleaning, and preprocessing; \
                    perform exploratory data analysis to uncover patterns, trends, and insights in the data; \
                    develop visualizations, such as charts and graphs, to communicate complex data in an understandable format; \
                    use business intelligence tools to query databases, create reports, and analyze data.",
    "Machine Learning Engineer": "- **Daily Working Avtivities**: \
                    Feature engineering, model selection, model training and evaluation, and hyperparameter tuning; \
                    deploy machine learning models to production or staging environments, integrate models into existing systems or applications; \
                    implement monitoring solutions to track the performance of deployed models, address issues related to model degradation or changes in data distribution.",
    "Product/Project Manager": "- **Daily Working Avtivities**: \
                    Conduct market research to understand customer needs, industry trends, and competitor offerings; \
                    create user stories, product requirements, and specifications, clearly communicate product features and functionality to development teams; \
                    plan product releases and coordinate with development teams to ensure timely and successful launches."
}

job_skill_map = {
    "Data Engineer": "- **Skill Required**: \
                    Proficiency in programming languages commonly used; \
                    familiar with databases; \
                    ability of designing and implementing data models.",
    "Software Engineer": "- **Skill Required**: \
                    Strong coding ability; \
                    strong understanding of software development principles and methodologies; \
                    strong problem-solving skills.",
    "Data Scientist": "- **Skill Required**: \
                    Strong understanding of statistical concepts and methodologies for data analysis; \
                    expertise in machine learning techniques and algorithms.",
    "Research Scientist": "- **Skill Required**: \
                    Strong understanding of research methodologies; \
                    ability to design and execute experiments; \
                    strong critical thinking skills.",
    "Statistician": "- **Skill Required**: \
                    Proficiency in a wide range of statistical methods; \
                    strong mathematical background; \
                    familiarity with statistical tools.",
    "Data/Business Analyst": "- **Skill Required**: \
                    Knowledge of statistical methods and techniques to analyze data trends and patterns; \
                    familiarity with BI tools such as Tableau, Power BI, QlikView, or Looker.",
    "Machine Learning Engineer": "- **Skill Required**: \
                    Expertise in using machine learning libraries and frameworks; \
                    knowledge of various machine learning models and techniques; \
                    ability to evaluate and improve model performance.",
    "Product/Project Manager": "- **Skill Required**: \
                    ability to analyze data, identify trends, make informed decisions, and address challenges and obstacles; \
                    stay informed about industry trends."
}

def throw_default_roles(name):
    role = "Data Scientist"
    st.markdown("### " + name +", here are your recommended job roles!")
    st.markdown("💁‍♀️ Our most recommended job role for you is ***" + role + "*** .")
    st.markdown("##### " + role)
    
    from PIL import Image
    col_icon, col_text = st.columns([0.3, 0.7])
    with col_icon:
        image = Image.open(job_icon_map[role])
        st.image(image)
    with col_text:
        st.markdown(job_description_map[role])
        st.markdown(job_activities_map[role])
        st.markdown(job_skill_map[role])
    
    st.markdown("🙋‍♂️ We also recommend **Data/Business Analyst**.")
    role = "Data/Business Analyst"
    col_icon, col_text = st.columns([0.3, 0.7])
    with col_icon:
        image = Image.open(job_icon_map[role])
        st.image(image)
    with col_text:
        st.markdown(job_description_map[role])
        st.markdown(job_activities_map[role])
        st.markdown(job_skill_map[role])
    
    


def show_recommendation():
    ########################################
    # train data read and preprocess #######
    ########################################
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    data = pd.read_csv('merged.csv')

    data = data[~data['job_title'].isin(['DBA/Database Engineer', 'Other','Developer Relations/Advocacy',\
                                        'Data Architect','Developer Advocate','Teacher / professor','Data Administrator', 'Engineer (non-software)'])]




    #Occupation 合并

    #重新给Business Analyst 和Data Analyst 命名为‘Data/Business Analyst’
    data['job_title'] = data['job_title'].replace('Business Analyst', 'Data/Business Analyst')
    data['job_title'] = data['job_title'].replace('Data Analyst', 'Data/Business Analyst')

    #重新给Program/Project Manager和Product Manager 命名为‘Product/Project Manager’
    data['job_title'] = data['job_title'].replace('Program/Project Manager', 'Product/Project Manager')
    data['job_title'] = data['job_title'].replace('Product Manager', 'Product/Project Manager')

    #重新给Manager命名为‘Product/Project Manager’
    data['job_title'] = data['job_title'].replace('Manager (Program, Project, Operations, Executive-level, etc)', 'Product/Project Manager')

    #重新给'Machine Learning/ MLops Engineer'命名为'Machine Learning Engineer'
    data['job_title'] = data['job_title'].replace('Machine Learning/ MLops Engineer', 'Machine Learning Engineer')

    #重新给Data Analyst (Business, Marketing, Financial, Quantitative, etc)命名为‘Data/Business Analyst’
    data['job_title'] = data['job_title'].replace('Data Analyst (Business, Marketing, Financial, Quantitative, etc)', 'Data/Business Analyst')

    # List of unique occupations after removal
    occupations = data['job_title'].unique()

    # Create a dictionary that maps each occupation to a unique number
    occupation_map = {occupation: i+1 for i, occupation in enumerate(occupations)}

    data['Occupation_Number'] = data['job_title'].map(occupation_map)
    
    
    
    

    # filter education
    educations = ['Master’s degree', 'Doctoral degree', 'Bachelor’s degree',
        'Some college/university study without earning a bachelor’s degree', 'Professional degree',
        'No formal education past high school', 'Professional doctorate']
    # Filter the rows
    data = data[data['highest_education'].isin(educations)]
    
    
    
    

    data = data.reset_index(drop=True)
    data = data.drop('Occupation_Number', axis=1)







    # model fitting
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    # Encode categorical variables
    label_encoders = {}
    for column in data.columns:
        if data[column].dtype == object:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            label_encoders[column] = le

    # Split the data into features and target variable
    X = data.drop('job_title', axis=1)  
    y = data['job_title']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)

    # Fit the model
    rf.fit(X_train, y_train)

    # Get feature importances
    importances = rf.feature_importances_

    # Convert the importances into a DataFrame
    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})

    # Sort the DataFrame by importance
    feature_importances = feature_importances.sort_values('importance', ascending=False)

    # Display the feature importances
    # print(feature_importances)

    usefull = ['job_title',
        'code_writing_experience',
        'hosted_notebook_used', 'data_vis_used', 'work_activities',
        'program_language_used', 'ide_used', 'ml_frameworks_used',
        'ml_algorithms', 'platforms', 'media_sources']

    df = data
    ###################################






    ###################################
    # job title map ###################
    ###################################
    job_title_map = {
        1: "Data Engineer", 
        2: "Software Engineer", 
        3: "Data Scientist", 
        4: "Research Scientist",
        5: "Statistician",
        6: "Data/Business Analyst",
        7: "Machine Learning Engineer",
        8: "Product/Project Manager"
    }
    ###################################








    ###################################
    # web page ########################
    ###################################
    survey = st.empty()

    name = None
    age = None
    gender = None
    degree = None
    TPU = None
    ml_experience = None
    coding_experience = None
    notebooks = None
    visualization_tools = None
    activities = None
    programming_languages = None
    IDEs = None
    ml_framework = None
    ml_algorithm = None
    cv_methods = None
    NLP = None
    course_platfroms = None
    media_sources = None

    submitted = False

    if(submitted == False):    
        with survey.form("survey"):
            st.markdown("Please fill in the following survey. Please be aware that our survey and recommendation model are aimed at people aged 18-40 years.")
            name = st.text_input("**Name**", placeholder="Type in your name...")
            
            age = st.selectbox(
                "**Age**: What's your age (# years)?", 
                ("18-21", "22-24", "30-34", "35-39")
            )
            
            gender = st.selectbox("**Gender**: Your biological gender.", ("Woman", "Man"))
            
            degree = st.selectbox(
                "**Highest Degree**: The highest level of formal education that you have attained or plan to attain within the next 2 years.", 
                ("No formal education past high school", "Some college/university study without earning a bachelor’s degree", "Bachelor’s degree", "Master’s degree", "Doctoral degree", "Professional degree")
            )
            
            coding_experience = st.selectbox(
                "**Coding Experience**: For how many years have you been writing code and/or programming?",
                (
                    "< 1 years", 
                    "1-3 years", 
                    "3-5 years", 
                    "5-10 years", 
                    "10-20 years", 
                    "20+ years"
                )
            )
                
            programming_languages = st.multiselect(
                "**Programming Languages**: What programming languages do you use on a regular basis? (Select all that apply)",
                (
                    "Python", 
                    "R", 
                    "SQL", 
                    "C", 
                    "C++", 
                    "Java", 
                    "Javascript", 
                    "Julia", 
                    "Swift", 
                    "Bash", 
                    "MATLAB", 
                    "C#", 
                    "PHP", 
                    "Go", 
                    "None", 
                    "Other"
                )
            )
            
            IDEs = st.multiselect(
                "**IDEs**: Which of the following integrated development environments (IDE's) do you use on a regular basis? (Select all that apply)",
                (
                    "JupyterLab (or products based off of Jupyter)", 
                    "RStudio/Posit", 
                    "Visual Studio", 
                    "Visual Studio Code (VSCode)", 
                    "PyCharm", 
                    "Spyder", 
                    "Notepad++", 
                    "Sublime Text", 
                    "Vim, Emacs, or similar", 
                    "MATLAB", 
                    "IntelliJ",
                    "None", 
                    "Other"
                )
            )
            
            notebooks = st.multiselect(
                "**Notebook products**: Which of the following hosted notebook products do you use on a regular basis? (Select all that apply)",
                (
                    "Kaggle Notebooks", 
                    "Colab Notebooks", 
                    "Azure Notebooks", 
                    "Paperspace / Gradient", 
                    "Binder / JupyterHub", 
                    "Code Ocean", 
                    "IBM Watson Studio", 
                    "Amazon Sagemaker Studio", 
                    "Amazon Sagemaker Studio Lab", 
                    "Amazon EMR Notebooks", 
                    "Google Cloud AI Platform Notebooks ", 
                    "Google Cloud Datalab Notebooks", 
                    "Google Cloud Vertex AI Workbench", 
                    "Databricks Collaborative Notebooks", 
                    "Hex Workspaces", 
                    "Noteable Notebooks", 
                    "Deepnote Notebooks", 
                    "Gradient Notebooks",
                    "None", 
                    "Other"
                )
            )
            
            visualization_tools = st.multiselect(
                "**Data Visualization Tools**: What data visualization libraries or tools do you use on a regular basis? (Select all that apply)",
                (
                    "Matplotlib", 
                    "Seaborn", 
                    "Plotly / Plotly Express", 
                    "Ggplot / ggplot2", 
                    "Shiny", 
                    "D3 js", 
                    "Altair", 
                    "Bokeh", 
                    "Geoplotlib", 
                    "Leaflet / Folium", 
                    "Pygal", 
                    "Dygraphs", 
                    "Highcharter",
                    "None", 
                    "Other"
                )
            )
            
            ml_experience = st.selectbox(
                "**Machine Learning Experience**: For how many years have you used machine learning methods?",
                (
                    'I do not use machine learning methods',
                    'Under 1 year',
                    '1-3 years',
                    '3-5 years',
                    '5-10 years',
                    '10 or more years'
                )
            )
            
            ml_framework = st.multiselect(
                "**Machine Learning Framework**: Which of the following machine earning frameworks do you use on a regular basis? (Select all that apply)",
                (
                    "Scikit-learn", 
                    "TensorFlow", 
                    "Keras", 
                    "PyTorch", 
                    "Fast.ai", 
                    "MXNet", 
                    "Xgboost", 
                    "LightGBM", 
                    "CatBoost", 
                    "Prophet", 
                    "H2O 3", 
                    "Caret", 
                    "Tidymodels", 
                    "JAX", 
                    "PyTorch Lightning", 
                    "Huggingface", 
                    "None", 
                    "Other"
                )
            )
            
            ml_algorithm = st.multiselect(
                "**Machine Learning Algorithm**: Which of the following ML algorithms do you use on a regular basis? (Select all that apply)",
                (
                    "Linear or Logistic Regression", 
                    "Decision Trees or Random Forests", 
                    "Gradient Boosting Machines (xgboost, lightgbm, etc)", 
                    "Bayesian Approaches", "Evolutionary Approaches", 
                    "Dense Neural Networks (MLPs, etc)", 
                    "Convolutional Neural Networks", 
                    "Generative Adversarial Networks", 
                    "Recurrent Neural Networks", 
                    "Transformer Networks (BERT, gpt-3, etc)", 
                    "Autoencoder Networks (DAE, VAE, etc)", 
                    "Graph Neural Networks", 
                    "None", 
                    "Other"
                )
            )

            # expected_salary = st.selectbox(
            #     "**Expected Salary**: What is your expected yearly compensation? (Approximate **$USD**)",
            #     ("$0-999", "1,000-1,999", "2,000-2,999", "3,000-3,999", "4,000-4,999", "5,000-7,499", "7,500-9,999", 
            #     "10,000-14,999", "15,000-19,999", "20,000-24,999", "25,000-29,999", "30,000-39,999", "40,000-49,999", "50,000-59,999", "60,000-69,999", "70,000-79,999", "80,000-89,999", "90,000-99,999",
            #     "100,000-124,999", "125,000-149,999", "150,000-199,999", "200,000-249,999", "250,000-299,999", "300,000-500,000", "> $500,000")
            # )


            
            cv_methods = st.multiselect(
                "**Computer Vision Methods**: Which categories of computer vision methods do you use on a regular basis? (Select all that apply)",
                (
                    "General purpose image/video tools (PIL, cv2, skimage, etc)",
                    "Image segmentation methods (U-Net, Mask R-CNN, etc)",
                    "Object detection methods (YOLOv3, RetinaNet, etc)",
                    "Image classification and other general purpose networks (VGG, Inception, ResNet, ResNeXt, NASNet, EfficientNet, etc)",
                    "Vision transformer networks (ViT, DeiT, BiT, BEiT, Swin, etc)",
                    "Generative Networks (GAN, VAE, etc)",
                    "None",
                    "Other"
                )
            )
            
            NLP = st.multiselect(
                "**NLP Methods**: Which of the following natural language processing (NLP) methods do you use on a regular basis? (Select all that apply)",
                (
                    "Word embeddings/vectors (GLoVe, fastText, word2vec)",
                    "Encoder-decoder models (seq2seq, vanilla transformers)",
                    "Contextualized embeddings (ELMo, CoVe)",
                    "Transformer language models (GPT-3, BERT, XLnet, etc)",
                    "None",
                    "Other"
                )
            )
            
            TPU = st.selectbox(
                "**TPU Using Times**: Approximately how many times have you used a TPU (tensor processing unit)?",
                ("Never", "Once", "2-5 times", "6-25 times", "More than 25 times")
            )
            
            activities = st.multiselect(
                "**Expected Role Activities**: Select any activities that you expect it to make up an important part of your role at work: (Select all that apply)",
                (
                    "Analyze and understand data to influence product or business decisions", 
                    "Build and/or run the data infrastructure that my business uses for storing, analyzing, and operationalizing data",
                    "Build prototypes to explore applying machine learning to new areas",
                    "Build and/or run a machine learning service that operationally improves my product or workflows",
                    "Experimentation and iteration to improve existing ML models",
                    "Do research that advances the state of the art of machine learning",
                    "Other"
                )
            )
            
            course_platfroms = st.multiselect(
                "**Courses Platforms**: On which platforms have you begun or completed data science courses? (Select all that apply)",
                (
                    "Coursera", 
                    "edX", 
                    "Kaggle Learn Courses", 
                    "DataCamp", 
                    "Fast.ai", 
                    "Udacity", 
                    "Udemy", 
                    "LinkedIn Learning", 
                    "Could-certification programs (direct from AWS, Azure, GCP, or similar)", 
                    "University Courses (resulting in a university degree)", 
                    "None", 
                    "Other"
                )
            )
            
            media_sources = st.multiselect(
                "**Favorite Media Sources**: Who/what are your favorite media sources that report on data science topics? (Select all that apply)",
                (
                    "Twitter (data science influencers)", 
                    "Email newsletters (Data Elixir, O'Reilly Data & AI, etc)", 
                    "Reddit (r/machinelearning, etc)", 
                    "Kaggle (notebooks, forums, etc)", 
                    "Course Forums (forums.fast.ai, Coursera forums, etc)", 
                    "YouTube (Kaggle YouTube, Cloud AI Adventures, etc)",
                    "Podcasts (Chai Time Data Science, O’Reilly Data Show, etc)",
                    "Blogs (Towards Data Science, Analytics Vidhya, etc)",
                    "Journal Publications (peer-reviewed journals, conference proceedings, etc)",
                    "Slack Communities (ods.ai, kagglenoobs, etc)",
                    "None",
                    "Other"
                )
            )

            submitted = st.form_submit_button("Submit")
            if(submitted):
                survey.empty()

    if(submitted):
        if(not name):
            name = "Hey"
        user_data = {
            "age": [age, ],
            "gender": [gender, ],
            "highest_education": [degree, ],
            "code_writing_experience": [coding_experience,],
            "hosted_notebook_used": [str(notebooks),],
            "data_vis_used": [str(visualization_tools),],
            "work_activities": [str(activities),],
            "program_language_used": ["; ".join(programming_languages),],
            "ide_used": ["; ".join(IDEs),],
            "ml_frameworks_used": ["; ".join(ml_framework),],
            "ml_algorithms": ["; ".join(ml_algorithm),],
            "platforms": ["; ".join(course_platfroms),],
            "media_sources": ["; ".join(media_sources),],
            "TPU": [TPU, ],
            "ml_experience": [ml_experience,],
            "computer_vision_methods": [";".join(cv_methods),],
            "NLP": [";".join(NLP), ]
        }
        user_df = pd.DataFrame(user_data)
        
        # age
        age_replace = {
            "18-21" : 18,
            "22-24" : 22, 
            "30-34" : 30, 
            "35-39" : 35
        }
        user_df["age"] = user_df["age"].replace(age_replace)
        
        
        # code_writing_experience
        cwe_replace = {
            "1-3 years": 0,
            "10-20 years": 1,
            "20+ years": 2,
            "3-5 years": 3,
            "5-10 years": 4,
            "< 1 years": 5
        }
        user_df['code_writing_experience'] = user_df['code_writing_experience'].replace(cwe_replace)
        
        print(user_df)
        
        # ml_experience
        mle_replace = {
            'I do not use machine learning methods': 0,
            'Under 1 year': 1,
            '1-3 years': 2,
            '3-5 years': 3,
            '5-10 years': 4,
            '10 or more years': 5
        }
        user_df["ml_experience"] = user_df["ml_experience"].replace(mle_replace)
        
        
        try:
            for column in user_df.columns:
                if column in label_encoders:
                    print(column)
                    le = label_encoders[column]
                    user_df[column] = le.transform(user_df[column])
        except Exception as e:
            print("error in model fitting")
            print(e)
            throw_default_roles(name)
            # st.markdown("### 🙇‍♂️ Sorry, we can't find the recommended job role for you, please go back and try other choices.")
            if(st.button("Back", type="primary")):
                submitted = False
            return

        recommended_jobs = {}
        
        
        
        ########################
        # logistic regression ##
        ########################
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        import numpy as np

        # Split the original DataFrame first
        X_train, X_test, y_train, y_test = train_test_split(df.drop('job_title', axis=1), df['job_title'], test_size=0.2, random_state=42)


        columns_to_exclude = ['job_title']
        # Identify categorical and numerical columns
        categorical_cols = [col for col in X_train.select_dtypes(include=['object', 'category']).columns if col not in columns_to_exclude]
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Define the preprocessing for categorical data: impute then one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Define the preprocessing for numerical data: impute then scale
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import classification_report, confusion_matrix

            # Apply the preprocessing to the test data
            # X_test_preprocessed = preprocessor.transform(X_test)

            # Create and fit the logistic regression model
            logistic_model = LogisticRegression()
            logistic_model.fit(X_train_preprocessed, y_train)

            # Predict using the test set
            # y_pred = logistic_model.predict(X_test_preprocessed)

            # Evaluate the model
            # print("Classification Report:")
            # print(classification_report(y_test, y_pred))
            # print("Confusion Matrix:")
            # print(confusion_matrix(y_test, y_pred))
            
            # predict
            user_preprocessed = preprocessor.transform(user_df)
            user_pred = logistic_model.predict(user_preprocessed)
            job_ll = job_title_map[user_pred[0]]
            
            if job_ll in recommended_jobs:
                recommended_jobs[job_ll] = recommended_jobs[job_ll] + 0.4
            else:
                recommended_jobs[job_ll] = 0.4
                
        except Exception as e:
            print("error in logistic regression")
            print(e)
        ########################
        
        
        ########################
        # XGBoost Optimized ####
        ########################
        from sklearn.preprocessing import LabelEncoder
        import xgboost as xgb
        from sklearn.metrics import classification_report, confusion_matrix

        try:
            # Encode the target variable
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)

            # Create the XGBoost classifier
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

            # Fit the model on the training data
            xgb_model.fit(X_train_preprocessed, y_train_encoded)

            # Predict using the test data
            # y_pred = xgb_model.predict(X_test_preprocessed)

            # Convert predictions back to original labels
            # y_pred_labels = label_encoder.inverse_transform(y_pred)

            # Evaluate the model
            # print("Classification Report:")
            # print(classification_report(y_test, y_pred_labels))
            # print("Confusion Matrix:")
            # print(confusion_matrix(y_test, y_pred_labels))   
            
            from xgboost import XGBClassifier

            # Initialize the XGBoost model with the best parameters
            optimized = xgb.XGBClassifier(
                learning_rate=0.1,
                max_depth=3,
                n_estimators=300,
                subsample=0.9,
                random_state=42  # For reproducibility
            )

            # Fit the model with your training data
            # Fit the model on the training data
            optimized.fit(X_train_preprocessed, y_train_encoded)

            # Predict using the test data
            # y_pred = optimized.predict(X_test_preprocessed)

            # Convert predictions back to original labels
            # y_pred_labels = label_encoder.inverse_transform(y_pred)

            # Evaluate the model
            # print("Classification Report:")
            # print(classification_report(y_test, y_pred_labels))
            # print("Confusion Matrix:")
            # print(confusion_matrix(y_test, y_pred_labels))    
            
            # predict
            user_preprocessed = preprocessor.transform(user_df)
            user_pred = optimized.predict(user_preprocessed)
            job_xgboost = job_title_map[user_pred[0]]
        
            if job_xgboost in recommended_jobs:
                recommended_jobs[job_xgboost] = recommended_jobs[job_xgboost] + 0.5
            else:
                recommended_jobs[job_xgboost] = 0.5
        except Exception as e:
            print("error in XGBoost")
            print(e)
        
        
        
        ##############################
        # Random Forest ##############
        ##############################
        try:
            #random forest
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            from sklearn.model_selection import train_test_split

            # Assuming X_preprocessed is the preprocessed feature matrix and y_encoded contains the encoded job titles


            # Split the dataset into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_train_preprocessed, y_train, test_size=0.2, random_state=42)

            # Initialize the Random Forest classifier
            rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)

            # Fit the classifier to the training data
            rf_clf.fit(X_train, y_train)

            # # Predict the job titles on the test set
            # y_pred = rf_clf.predict(X_test)

            # # Calculate the accuracy on the test set
            # accuracy = accuracy_score(y_test, y_pred)
            # print(f'Random Forest test set accuracy: {accuracy}')
            user_preprocessed = preprocessor.transform(user_df)
            user_pred = rf_clf.predict(user_preprocessed)
            job_rf = job_title_map[user_pred[0]]
        
            if job_rf in recommended_jobs:
                recommended_jobs[job_rf] = recommended_jobs[job_rf] + 0.49
            else:
                recommended_jobs[job_rf] = 0.49
        except Exception as e:
            print("error in random forest")
            print(e)
        
        
        ##############################
        # KNN ########################
        ##############################
        try:
            from sklearn.neighbors import NearestNeighbors
            # Preprocess the data
            X_preprocessed = preprocessor.fit_transform(X)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, test_size=0.2, random_state=42)

            # Number of neighbors
            K = 20

            # Initialize the NearestNeighbors model
            nn_model = NearestNeighbors(algorithm="brute", n_neighbors=K+1)
            nn_model.fit(X_train_preprocessed)

            # Find the K+1 nearest neighbors (including the point itself)
            user_preprocessed = preprocessor.transform(user_df)
            distances, indices = nn_model.kneighbors(user_preprocessed)
            # distances, indices = nn_model.kneighbors(X_test_preprocessed)

            # Perform the majority vote for job title predictions based on the nearest neighbors
            # For each set of neighbor indices, we find the most common job title label among those neighbors
            # y_pred_nn = []
            # for neighbor_indices in indices:
            #     # Extract the job titles of the K nearest neighbors
            #     neighbor_job_titles = y_train[neighbor_indices[1:]] # skip the first one because it is the point itself
            #     # Find the most common job title among these neighbors
            #     most_common_job_title = max(set(neighbor_job_titles), key=list(neighbor_job_titles).count)
            #     y_pred_nn.append(most_common_job_title)

            # # Convert the list of predictions to a numpy array
            # y_pred_nn = np.array(y_pred_nn)

            user_pred_nn = []
            for neighbor_indices in indices:
                neighbor_job_titles = y_train[neighbor_indices[1:]]
                most_common_job_title = max(set(neighbor_job_titles), key=list(neighbor_job_titles).count)
                user_pred_nn.append(most_common_job_title)
                
            job_knn = job_title_map[user_pred[0]]
            if job_knn in recommended_jobs:
                recommended_jobs[job_knn] = recommended_jobs[job_knn] + 0.39
            else:
                recommended_jobs[job_knn] = 0.39

            # # Calculate the accuracy of the NearestNeighbors model
            # accuracy_nn = accuracy_score(y_test, y_pred_nn)

            # # Output the accuracy
            # accuracy_nn
        except Exception as e:
            print("error in KNN")
            print(e)
        
        
        
        
        
            
        if not recommended_jobs:
            print("error: no valid recommendation")
            throw_default_roles(name)
            # st.markdown("### 🙇‍♂️ Sorry, we can't find the recommended job role for you, please go back and try other choices.")
        else:
            role = max(recommended_jobs, key=recommended_jobs.get)
            st.markdown("### " + name +", here are your recommended job roles!")
            st.markdown("💁‍♀️ Our most recommended job role for you is ***" + role + "*** .")
            st.markdown("##### " + role)
            
            from PIL import Image
            col_icon, col_text = st.columns([0.3, 0.7])
            with col_icon:
                image = Image.open(job_icon_map[role])
                st.image(image)
            with col_text:
                st.markdown(job_description_map[role])
                st.markdown(job_activities_map[role])
                st.markdown(job_skill_map[role])
            
            
            if len(recommended_jobs) > 1:
                sentence = "🙋‍♂️ We also recommend "
                sorted_jobs = sorted(recommended_jobs.items(), key=lambda x: x[1], reverse=True)
                other_jobs = []
                for key, value in sorted_jobs:
                    if key == role:
                        continue
                    sentence = sentence + " **" + key + "**,"
                    other_jobs.append(key)
                sentence = sentence[:-1]
                sentence = sentence + "."
                st.markdown(sentence)
                
                for job in other_jobs:
                    st.markdown("##### " + job)
                    col_icon, col_text = st.columns([0.3, 0.7])
                    with col_icon:
                        image = Image.open(job_icon_map[job])
                        st.image(image)
                    with col_text:
                        st.markdown(job_description_map[job])
                        st.markdown(job_activities_map[job])
                        st.markdown(job_skill_map[job])
        
        if(st.button("Back", type="primary")):
            submitted = False





def show_models():
    from PIL import Image
    st.markdown("#### We use 4 different models to do the recommendation, and our recommendation is ranked based on the accuracy of each model.")
    
    st.markdown("##### Logistic Regression")
    col_icon, col_text = st.columns([0.3, 0.7], gap="large")
    with col_icon:
        image = Image.open("./pictures/lr_icon.png")
        st.image(image) 
    with col_text:
        st.markdown("###### Model Description")
        st.markdown("Logistic regression makes predictions by applying a logistic function to a linear combination of the input features.")
        st.markdown(" ")
        st.markdown("###### Model Metric")
        st.markdown("- Accuracy: 41%")
    
    st.markdown("##### XGBoost")
    col_icon, col_text = st.columns([0.3, 0.7], gap="large")
    with col_icon:
        image = Image.open("./pictures/xgb_icon.png")
        st.image(image) 
    with col_text:
        st.markdown("###### Model Description")
        st.markdown("XGBoost iteratively combines weak learners to enhance accuracy and handle complex relationships.")
        st.markdown(" ")
        st.markdown("###### Model Metric")
        st.markdown("- Accuracy: 50%")
        
    st.markdown("##### Random Forest")
    col_icon, col_text = st.columns([0.3, 0.7], gap="large")
    with col_icon:
        image = Image.open("./pictures/rf_icon.png")
        st.image(image) 
    with col_text:
        st.markdown("###### Model Description")
        st.markdown("Random forest builds a multitude of decision trees during training, and outputs the mode of the classes for classification, resulting in a robust and accurate predictive model.")
        st.markdown(" ")
        st.markdown("###### Model Metric")
        st.markdown("- Accuracy: 47.4%")
        
    st.markdown("##### KNN")
    col_icon, col_text = st.columns([0.3, 0.7], gap="large")
    with col_icon:
        image = Image.open("./pictures/knn_icon.png")
        st.image(image) 
    with col_text:
        st.markdown("###### Model Description")
        st.markdown("K-Nearest Neighbors classifies a data point based on the majority class of its k-nearest neighbors in the feature space.")
        st.markdown(" ")
        st.markdown("###### Model Metric")
        st.markdown("- Accuracy: 40.1%")



user_choose = st.sidebar.radio(
    "In recommendation, you'll fill in a technical background survey, then be showed the recommended jobs based on your survey. In recommendation models, you can explore the recommendation models we used.",
    ["***Recommendation***", 
     "***Recommendation Models***"],
    captions = ["Get Your Job Role Recommendation", "Know More About The Recommendation Models"]
)

if(user_choose == "***Recommendation***"):
    show_recommendation()
elif(user_choose == "***Recommendation Models***"):
    show_models()
