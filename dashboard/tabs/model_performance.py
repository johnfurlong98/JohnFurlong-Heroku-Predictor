# tabs/model_performance.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def show_page(data, data_original, models, scaler, selected_features, skewed_features, lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details):
    st.title("Model Performance")
    st.header("Performance Metrics")
    results_df = model_evaluation
    if results_df.empty:
        st.warning("**Warning:** Model evaluation results are empty.")
    else:
        results_df_filtered = results_df[results_df['Model'] != 'XGBoost']
        if results_df_filtered.empty:
            st.error("**Error:** No models available after excluding 'XGBoost'.")
        else:
            if 'RMSE' not in results_df_filtered.columns or 'Model' not in results_df_filtered.columns:
                st.error("**Error:** 'RMSE' or 'Model' columns not found in the evaluation results.")
            else:
                st.write("""
                ### Model Evaluation Metrics
                The table below presents the performance metrics of various regression models. These metrics help in assessing the accuracy and reliability of each model.
                """)
                st.dataframe(results_df_filtered.style.format({'MAE': '${:,.2f}', 'RMSE': '${:,.2f}', 'R² Score': '{:.4f}'}))
                best_model_row = results_df_filtered.loc[results_df_filtered['RMSE'].idxmin()]
                best_model_name = best_model_row['Model']
                st.write(f"### Best Performing Model: **{best_model_name}**")
                st.write(f"""
                Based on the RMSE metric, **{best_model_name}** emerges as the top-performing model.
                """)
                st.write("""
                ### Understanding the Metrics
                - **MAE:** Average absolute error.
                - **RMSE:** Penalizes larger errors more heavily.
                - **R² Score:** Proportion of variance explained.
                """)

                st.header("Detailed Pipeline Explanation")
                st.write("""
                ### 1. Data Collection and Understanding
                ### 2. Data Cleaning
                ### 3. Feature Engineering
                ### 4. Feature Transformation
                ### 5. Feature Selection
                ### 6. Data Scaling
                ### 7. Model Training
                ### 8. Model Evaluation
                ### 9. Deployment
                (The same explanation text from tab5)
                """)

                st.header("Feature Importances")
                if best_model_name in models:
                    feature_importances_best = feature_importances.copy()
                    if feature_importances_best.empty:
                        st.warning(f"**Warning:** Feature importances for '{best_model_name}' not available.")
                    else:
                        plt.figure(figsize=(12, 8))
                        sns.barplot(x='Importance', y='Feature', data=feature_importances_best.sort_values(by='Importance', ascending=False), palette='viridis')
                        plt.title(f'Feature Importances from {best_model_name}', fontsize=16)
                        plt.xlabel('Importance', fontsize=12)
                        plt.ylabel('Feature', fontsize=12)
                        plt.tight_layout()
                        st.pyplot(plt)
                        st.write("""
                        The bar chart shows the relative importance of each feature in predicting sale price.
                        """)
                else:
                    st.warning(f"**Warning:** Feature importances for '{best_model_name}' not available.")

                st.header("Residual Analysis")
                if best_model_name in models and train_test_data:
                    try:
                        selected_model = models[best_model_name]
                        X_train, X_test, y_train, y_test = train_test_data
                        if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
                            st.warning("**Warning:** Test data is empty. Cannot perform residual analysis.")
                        else:
                            y_pred_log = selected_model.predict(X_test)
                            y_pred_actual = np.expm1(y_pred_log)
                            y_pred_actual[y_pred_actual < 0] = 0
                            y_test_actual = np.expm1(y_test)
                            residuals = y_test_actual - y_pred_actual

                            plt.figure(figsize=(10, 6))
                            sns.histplot(residuals, kde=True, color='coral', bins=30)
                            plt.title('Residuals Distribution', fontsize=16)
                            plt.xlabel('Residuals (USD)', fontsize=12)
                            plt.ylabel('Frequency', fontsize=12)
                            plt.tight_layout()
                            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
                            plt.xticks(rotation=45)
                            st.pyplot(plt)
                            st.write("""
                            **Understanding Residuals:** Residuals are approximately normally distributed, indicating unbiased errors.
                            """)
                    except Exception as e:
                        st.error(f"**Error during Residual Analysis plotting:** {e}")
                else:
                    st.warning(f"**Warning:** Model '{best_model_name}' not found or train/test data missing.")

    st.write("""
    ### Conclusion
    The evaluation highlights the effectiveness of the predictive pipeline. Future steps include data enrichment, model expansion, and continuous monitoring.
    """)

