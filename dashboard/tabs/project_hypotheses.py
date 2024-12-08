# tabs/project_hypotheses.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def show_page(data, data_original, models, scaler, selected_features, skewed_features, lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details):
    st.title("Project Hypotheses")
    st.write("""
    ## Hypothesis Validation
    In this section, we explore the foundational hypotheses that guided our analysis and modeling efforts. Each hypothesis is validated using statistical and machine learning techniques, providing a deeper understanding of the factors influencing house prices.
    """)
    st.subheader("Primary Hypotheses")
    st.write("""
    **Hypothesis 1:** *Higher overall quality of the house leads to a higher sale price.*
    - **Rationale:** Quality metrics such as construction standards, materials used, and overall maintenance directly impact the desirability and value of a property.
    - **Validation:** The `OverallQual` feature shows a strong positive correlation with the sale price, confirming this hypothesis.
    """)
    st.write("""
    **Hypothesis 2:** *Larger living areas result in higher sale prices.*
    - **Rationale:** Square footage is a fundamental indicator of a property's size and usability.
    - **Validation:** Features like `GrLivArea` and `TotalSF` have high correlations with the sale price, supporting this hypothesis.
    """)
    st.write("""
    **Hypothesis 3:** *Recent renovations positively impact the sale price.*
    - **Rationale:** Modern updates and renovations can enhance a property's appeal and value.
    - **Validation:** The `YearRemodAdd` feature correlates with the sale price, indicating that more recent remodels can increase house value.
    """)
    st.write("""
    **Hypothesis 4:** *The presence and quality of a garage significantly influence the sale price.*
    - **Rationale:** Garages add convenience and storage, enhancing property value.
    - **Validation:** `GarageArea` and `GarageFinish` correlate positively with sale price.
    """)
    st.write("""
    **Hypothesis 5:** *Lot size and frontage are key determinants of a house's market value.*
    - **Rationale:** Larger lots offer more outdoor space and potential for expansion.
    - **Validation:** `LotArea` and `LotFrontage` show positive correlations with sale price.
    """)
    st.write("""
    **Hypothesis 6:** *The number of bedrooms above grade influences the sale price.*
    - **Rationale:** More bedrooms can accommodate larger families, increasing appeal.
    - **Validation:** `BedroomAbvGr` correlates positively with sale price.
    """)

    data_for_corr = pd.concat([data.drop('SalePrice', axis=1), data_original], axis=1)

    st.write("### Visualization of Hypotheses")
    st.write("#### SalePrice vs OverallQual")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=data_for_corr, palette='Set2')
    plt.title('SalePrice vs OverallQual', fontsize=16)
    plt.xlabel('Overall Quality', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:** Houses with higher overall quality ratings command higher sale prices, supporting Hypothesis 1.
    """)

    st.write("#### SalePrice vs TotalSF")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalSF', y='SalePrice', data=data_for_corr, hue='OverallQual', palette='coolwarm', alpha=0.6)
    plt.title('SalePrice vs TotalSF', fontsize=16)
    plt.xlabel('Total Square Footage', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.legend(title='Overall Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:** Larger total square footage correlates with higher sale prices, supporting Hypothesis 2.
    """)

    st.write("#### SalePrice vs YearRemodeled")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='YearRemodAdd', y='SalePrice', data=data_for_corr, color='green', ci=None)
    plt.title('SalePrice vs Year Remodeled', fontsize=16)
    plt.xlabel('Year Remodeled', fontsize=12)
    plt.ylabel('Average Sale Price (USD)', fontsize=12)
    plt.tight_layout()
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
