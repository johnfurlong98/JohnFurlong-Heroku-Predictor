# tabs/project_summary.py
import streamlit as st

def show_page(data, data_original, models, scaler, selected_features, skewed_features, lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details):
    st.title("House Price Prediction Dashboard")
    st.write("""
    ## Project Overview
    Welcome to the **House Price Prediction Dashboard**, an interactive tool designed to estimate the sale prices of residential properties in Ames, Iowa. This dashboard leverages advanced machine learning techniques to provide accurate and insightful predictions, aiding stakeholders such as homeowners, real estate agents, and investors in making informed decisions.

    ### Background
    The real estate market is influenced by a multitude of factors ranging from physical property attributes to economic conditions. Accurately predicting house prices requires a comprehensive analysis of these variables to understand their impact on market value.

    ### Objectives
    The primary goals of this project are:
    - **Data Analysis and Preprocessing:** Perform thorough exploratory data analysis to understand the underlying patterns and relationships within the data. Clean and preprocess the data to ensure quality and consistency.
    - **Feature Engineering:** Develop new features that enhance the model's predictive capabilities by capturing essential aspects of the properties that influence sale prices.
    - **Model Development:** Train and evaluate multiple regression models to identify the best-performing algorithm based on key performance metrics.
    - **Deployment:** Create an interactive and user-friendly dashboard that allows users to explore data insights and obtain real-time house price predictions.

    ### Methodology
    - **Data Collection:** Utilized historical house sale data, encompassing various features such as structural details, renovations, and amenities.
    - **Exploratory Data Analysis (EDA):** Conducted in-depth analysis to identify trends, outliers, and correlations between features and the sale price.
    - **Data Preprocessing:** Addressed missing values, handled categorical variables through encoding, and transformed skewed data distributions to improve model performance.
    - **Feature Selection:** Employed statistical techniques and model-based methods to select the most significant features influencing house prices.
    - **Model Training and Evaluation:** Implemented and compared several regression models, including Linear Regression, Ridge Regression, Lasso Regression, ElasticNet, Random Forest, and Gradient Boosting. Evaluated models using metrics like MAE, RMSE, and R² Score.
    - **Deployment:** Developed this Streamlit dashboard to make the predictive model accessible for real-time use and to present key findings interactively.

    ### Key Features of the Dashboard
    - **Feature Correlations:** Visualize and understand how different property features correlate with sale prices.
    - **House Price Predictions:** Input specific property details to receive instant price predictions.
    - **Hypothesis Testing:** Explore validated hypotheses that explain the impact of various features on house prices.
    - **Model Performance:** Review the performance metrics and understand the model's predictive power and limitations.

    ### How to Use This Dashboard
    - Navigate through the tabs at the top to access different sections.
    - Use the **Feature Correlations** tab to explore how individual features affect house prices.
    - Visit the **House Price Predictions** tab to estimate the sale price of a property by entering its attributes.
    - Delve into the **Project Hypotheses** tab to understand the reasoning behind feature selection and their impact on prices.
    - Check the **Model Performance** tab for detailed insights into the model's accuracy and reliability.

    ### Acknowledgments
    This project was developed by John Furlong with support from the Code Institute. I acknowledge the Code Institute for providing comprehensive course material, support staff, lectures, and resources that were instrumental in the completion of this project.  The project is based on the Ames Housing Dataset, a comprehensive compilation of property sale records.

    ---
    """)
