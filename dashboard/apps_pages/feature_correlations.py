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
    data_for_corr = pd.concat([data.drop('SalePrice', axis=1), data_original], axis=1)
    corr_matrix = data_for_corr.corr()
    if 'SalePrice' not in corr_matrix.columns:
        st.error("**Error:** 'SalePrice' column not found in the dataset.")
    else:
        top_corr_features = corr_matrix.index[abs(corr_matrix['SalePrice']) > 0.5].tolist()
        if len(top_corr_features) == 0:
            st.warning("**Warning:** No features found with a correlation greater than 0.5 with 'SalePrice'.")
        else:
            st.write("""
            ### Top Correlated Features with Sale Price
            The heatmap below shows the correlation coefficients between the sale price and other features. Features with higher absolute correlation values have a stronger relationship with the sale price.
            """)
            plt.figure(figsize=(12, 8))
            sns.heatmap(data_for_corr[top_corr_features].corr(), annot=True, cmap='RdBu', linewidths=0.5, fmt=".2f")
            plt.title('Correlation Heatmap of Top Features', fontsize=16)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            st.pyplot(plt)
            st.write("""
            **Key Observations:**
            - **Overall Quality (`OverallQual`):** Strong positive correlation with sale price.
            - **Above Grade Living Area (`GrLivArea`):** Larger living areas are associated with higher sale prices.
            - **Total Square Footage (`TotalSF`):** Total area including basement and above-ground strongly influences sale price.
            - **Garage Area (`GarageArea`):** Larger garages contribute to higher house values.
            - **Lot Area (`LotArea`):** Bigger lots generally correlate with increased sale prices.
            """)

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

            st.write("### Pairplot of Top Correlated Features")
            top_features = [feat for feat in top_corr_features if feat != 'SalePrice'][:5]
            if len(top_features) == 0:
                st.warning("**Warning:** Not enough features to create a pairplot.")
            else:
                sample_size = 500
                if data_for_corr.shape[0] > sample_size:
                    pairplot_data = data_for_corr[top_features + ['SalePrice']].sample(n=sample_size, random_state=42)
                else:
                    pairplot_data = data_for_corr[top_features + ['SalePrice']]

                sns.set(style="ticks")
                pairplot_fig = sns.pairplot(pairplot_data, diag_kind='kde', height=2.5)
                plt.suptitle('Pairplot of Top Correlated Features', y=1.02)
                st.pyplot(pairplot_fig)
                st.write("""
                The pairplot above visualizes pairwise relationships between the top correlated features and the sale price. Sampling the data ensures quicker rendering while maintaining the overall trend insights.
                """)
    st.write("""
    ### Interpreting Correlations
    - **Feature Selection:** Highly correlated features are prioritized for model training to enhance predictive performance.
    - **Multicollinearity Detection:** Identifying correlated features helps in mitigating multicollinearity issues, which can adversely affect certain regression models.
    - **Insight Generation:** Correlation analysis provides actionable insights into what drives house prices, aiding stakeholders in making informed decisions.
    **Note:** Correlation does not imply causation. While features may be correlated with the sale price, further analysis is required to establish causal relationships.
    """)

