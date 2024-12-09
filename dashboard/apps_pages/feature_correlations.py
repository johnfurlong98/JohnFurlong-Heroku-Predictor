# tabs/feature_correlations.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def show_page(data, data_original, models, scaler, selected_features, skewed_features, lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details):
    st.title("Feature Correlations")
    st.write("""
    ## Understanding Feature Relationships
    Understanding how different features correlate with the sale price is crucial for building an effective predictive model. This section visualizes the relationships between key property attributes and the sale price.
    """)

    # Combine processed and original data
    data_for_corr = pd.concat([data.drop('SalePrice', axis=1), data_original], axis=1)
    corr_matrix = data_for_corr.corr()
    
    # Check if 'SalePrice' exists in the correlation matrix
    if 'SalePrice' not in corr_matrix.columns:
        st.error("**Error:** 'SalePrice' column not found in the dataset.")
    else:
        st.write("""
        ### Correlation Heatmap
        The heatmap below shows the correlation coefficients between all features and the sale price. Use it to identify strong relationships.
        """)
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, cmap='RdBu', linewidths=0.5, annot=False)
        plt.title('Correlation Heatmap', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt)

        st.write("### Choose Any Feature vs. SalePrice Plot")
        st.write("""
        Select any feature from the dropdown below to visualize its relationship with the sale price. This interactive scatter plot allows you to explore patterns and relationships in the data.
        """)
        # Create a dropdown for all features
        selected_feature = st.selectbox(
            "Select a feature to visualize against SalePrice",
            options=[col for col in data_for_corr.columns if col != 'SalePrice']
        )
        # Interactive scatter plot with Plotly
        fig = px.scatter(
            data_for_corr,
            x=selected_feature,
            y="SalePrice",
            color="OverallQual" if "OverallQual" in data_for_corr.columns else None,
            hover_data=["GrLivArea", "GarageArea", "YearBuilt"] if all(col in data_for_corr.columns for col in ["GrLivArea", "GarageArea", "YearBuilt"]) else None,
            title=f"SalePrice vs {selected_feature}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.write("### Pairplot of Top Features")
        st.write("""
        This pairplot visualizes pairwise relationships between the most relevant features and the sale price.
        """)
        top_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).index[:5].tolist()
        if len(top_features) == 0:
            st.warning("**Warning:** Not enough features to create a pairplot.")
        else:
            # Sample data for better rendering if necessary
            sample_size = 500
            if data_for_corr.shape[0] > sample_size:
                pairplot_data = data_for_corr[top_features].sample(n=sample_size, random_state=42)
            else:
                pairplot_data = data_for_corr[top_features]

            sns.set(style="ticks")
            pairplot_fig = sns.pairplot(pairplot_data, diag_kind='kde', height=2.5)
            plt.suptitle('Pairplot of Selected Features', y=1.02)
            st.pyplot(pairplot_fig)
    
    st.write("""
    ### Interpreting Correlations
    - **Feature Selection:** Highly correlated features are prioritized for model training to enhance predictive performance.
    - **Multicollinearity Detection:** Identifying correlated features helps in mitigating multicollinearity issues, which can adversely affect certain regression models.
    - **Insight Generation:** Correlation analysis provides actionable insights into what drives house prices, aiding stakeholders in making informed decisions.
    **Note:** Correlation does not imply causation. While features may be correlated with the sale price, further analysis is required to establish causal relationships.
    """)
