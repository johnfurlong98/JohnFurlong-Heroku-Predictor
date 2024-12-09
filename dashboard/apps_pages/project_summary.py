import streamlit as st

def show_page(data, data_original, models, scaler, selected_features, skewed_features, lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details):
    st.title("House Price Prediction Dashboard")

    st.write("""
    ## Project Overview

    ### Introduction
    Welcome to the **House Price Prediction Dashboard**, a sophisticated, data-driven solution designed to estimate residential property sale prices in Ames, Iowa. This interactive platform merges comprehensive data analysis, advanced Machine Learning (ML) techniques, and user-focused design to empower stakeholders—homeowners, real estate agents, investors, and analysts—to make informed, strategic decisions.

    The ultimate purpose of this project is to deliver actionable insights and accurate predictive capabilities. By integrating best practices in Data Science, adhering to the CRISP-DM methodology, and focusing on real-world business requirements, this dashboard stands as both a practical decision-support tool and a demonstration of robust data modeling techniques.

    ### Business Understanding & User Stories
    Within the real estate landscape, accurate house price estimation is key to numerous stakeholders:

    - **Homeowners:**  
      *User Story:* As a homeowner, I want to understand which property attributes (e.g., quality, square footage, lot area, recent renovations) most influence my home’s value, so I can decide on improvements that maximize resale price.

    - **Real Estate Agents:**  
      *User Story:* As an agent, I need a reliable, data-backed tool to estimate a property's market value quickly and consistently, so I can advise clients with confidence and set competitive listing prices.

    - **Investors and Developers:**  
      *User Story:* As an investor, I want insights into the key drivers of house prices to identify undervalued properties and forecast returns on renovations, ensuring I invest strategically and reduce financial risk.

    These user stories define the business requirements that this project aims to address: providing quick, accurate price estimates, revealing the importance of various features, and supporting data-informed decision-making.

    ### CRISP-DM Framework Alignment
    This project aligns with the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework, ensuring a structured and professional approach:

    1. **Business Understanding:**  
       Identified the primary need: accurate, explainable house price predictions in Ames, Iowa. Clarified success metrics (e.g., RMSE, R²) and targeted user stories to ensure the solution meets real-world needs.

    2. **Data Understanding:**  
       Explored the Ames Housing Dataset, examining distributions, correlations (e.g., `OverallQual`, `GrLivArea`, `TotalSF`, and `GarageArea` as strong predictors), and spotting trends that drive prices. Recognized skewness in certain features (like `LotArea`) and identified data quality issues.

    3. **Data Preparation:**  
       Cleaned and processed the data, addressing missing values, encoding categorical variables (e.g., basement and garage finish types), and transforming skewed distributions. Engineered new features such as `TotalSF` and `Qual_TotalSF` to capture key value-driving attributes more effectively.

    4. **Modeling:**  
       Developed and trained multiple regression models—Linear, Ridge, Lasso, ElasticNet, Random Forest, and Gradient Boosting—tuning hyperparameters to optimize predictive performance. Employed feature importance techniques and performance metrics to guide model selection.

    5. **Evaluation:**  
       Assessed each model using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score, prioritizing models that minimize prediction errors and maximize explanatory power. Established a performance baseline and selected the top-performing model that best balances accuracy and interpretability.

    6. **Deployment:**  
       Integrated the final chosen model into a user-friendly Streamlit dashboard. The dashboard provides real-time predictions, visualizations of feature correlations, hypothesis validations, and transparent model performance metrics, ensuring accessibility and practical utility.

    ### Objectives & Key Success Metrics
    The project’s primary objectives include:
    - **Data Analysis & Preprocessing:** Achieve a clean, reliable dataset suitable for ML modeling, ensuring no critical data issues remain.
    - **Feature Engineering:** Enhance predictive strength by creating domain-relevant features that capture the intricacies of property valuation.
    - **Model Selection & Validation:** Identify a best-in-class regression model that achieves a strong R² (indicating high variance explained) and low RMSE (indicating accurate price predictions).
    - **Actionable Insights & Explainability:** Provide not only price predictions but also a clear understanding of why certain features matter, supporting better decision-making.
    - **Seamless Deployment:** Offer a high-quality, professional dashboard that users can easily navigate to access all insights, ensuring the model’s value is tangible and immediate.

    ### Methodology & ML Tasks
    This project involves several key ML tasks that align with our business requirements and methodology:
    - **Feature Correlation & Selection:** Identify the most impactful features to ensure a simpler yet highly effective model.
    - **Model Development & Hyperparameter Tuning:** Utilize a variety of ML algorithms, adjusting parameters to enhance performance.
    - **Performance Evaluation (R², MAE, RMSE):** Systematically compare models and select one that meets or exceeds defined success criteria.

    These tasks reflect a rigorous, value-oriented approach to ML, ensuring the final model generates meaningful predictions for end-users.

    ### Dashboard Features & Navigation
    The dashboard presents a cohesive narrative from raw data to actionable insights:

    - **Project Summary (This Page):**  
      Gain an overview of the project’s purpose, methodology, and alignment with both business needs and the CRISP-DM framework.

    - **Feature Correlations:**  
      Dive into the relationships between critical property features and sale price. Visual correlations and heatmaps help identify patterns and justify which features are key drivers of value.

    - **House Price Predictions:**  
      Input specific property attributes (e.g., year built, basement finish type, lot area, overall quality) and instantly receive a predicted sale price. This tool helps homeowners, agents, and investors assess a property’s market value quickly.

    - **Project Hypotheses:**  
      Review how initial assumptions about what influences price (e.g., quality, square footage, renovations, garages) hold up against the data. Validate or refute these hypotheses with empirical evidence and statistical analysis.

    - **Model Performance:**  
      Examine the chosen model’s accuracy, stability, and feature importance. Understand residual distributions, verification metrics, and how well the model meets the established business and accuracy criteria.

    ### Acknowledgments & Resources
    This project was developed by John Furlong with support from the Code Institute. Their comprehensive curriculum, supportive resources, and practical assignments provided the foundational knowledge required to build this solution. The Ames Housing Dataset served as an excellent real-world data source, supplying rich feature information.

    Additionally, ChatGPT by OpenAI aided in conceptual clarification, technique refinement, and problem-solving. This synergy combining structured learning with AI driven guidance resulted in a robust, reliable, and user-focused tool.

    **In essence**, this House Price Prediction Dashboard is a professional, end-to-end Data Science project. By weaving together business requirements, CRISP-DM methodologies, ML best practices, and user-centric design, it stands as a credible demonstration of how data and predictive modeling can shape strategic decisions in the real estate market.

    ---
    """)
