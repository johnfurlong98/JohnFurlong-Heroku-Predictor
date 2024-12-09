# tabs/house_price_predictions.py
import streamlit as st
import pandas as pd
import numpy as np

def show_page(data, data_original, models, scaler, selected_features, skewed_features, lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details, inherited_houses):
    from scipy.stats import boxcox
    from scipy.special import inv_boxcox

    # We must define the same functions user_input_features as in original code and replicate logic exactly.
    # The code inside with tab3: block is copied here as-is.
    
    def feature_engineering(df):
        df = df.copy()
        df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
        df['Qual_TotalSF'] = df.get('OverallQual', 0) * df.get('TotalSF', 0)
        return df

    def preprocess_data_tab3(df, data_reference=None):
        df_processed = df.copy()

        user_to_model_mappings = {
            'BsmtFinType1': {
                'No Basement': 'None',
                'Unfinished': 'Unf',
                'Low Quality': 'LwQ',
                'Rec Room': 'Rec',
                'Basement Living Quarters': 'BLQ',
                'Average Living Quarters': 'ALQ',
                'Good Living Quarters': 'GLQ'
            },
            'BsmtExposure': {
                'No Basement': 'None',
                'No Exposure': 'No',
                'Minimum Exposure': 'Mn',
                'Average Exposure': 'Av',
                'Good Exposure': 'Gd'
            },
            'GarageFinish': {
                'No Garage': 'None',
                'Unfinished': 'Unf',
                'Rough Finished': 'RFn',
                'Finished': 'Fin'
            },
            'KitchenQual': {
                'Poor': 'Po',
                'Fair': 'Fa',
                'Typical/Average': 'TA',
                'Good': 'Gd',
                'Excellent': 'Ex'
            }
        }
        for col, mapping in user_to_model_mappings.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map(mapping)

        zero_fill_features = ['2ndFlrSF', 'EnclosedPorch', 'MasVnrArea', 'WoodDeckSF',
                              'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'BsmtUnfSF']
        for feature in zero_fill_features:
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna(0)

        categorical_mode_fill = {
            'BsmtFinType1': 'None',
            'GarageFinish': 'Unf',
            'BsmtExposure': 'No',
            'KitchenQual': 'TA'
        }
        for feature, value in categorical_mode_fill.items():
            if feature in df_processed.columns:
                df_processed[feature] = df_processed[feature].fillna(value)

        numerical_median_fill = ['BedroomAbvGr', 'GarageYrBlt', 'LotFrontage',
                                 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']
        for feature in numerical_median_fill:
            if feature in df_processed.columns:
                if data_reference is not None and feature in data_reference.columns:
                    median_value = data_reference[feature].median()
                else:
                    median_value = df_processed[feature].median()
                df_processed[feature] = df_processed[feature].fillna(median_value)

        ordinal_mappings = {
            'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
            'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
        }
        for col, mapping in ordinal_mappings.items():
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].map(mapping)

        df_processed = feature_engineering(df_processed)

        for feat in skewed_features:
            if feat == 'SalePrice':
                continue
            if feat in df_processed.columns:
                if (df_processed[feat] <= 0).any():
                    df_processed[feat] = np.log1p(df_processed[feat])
                else:
                    lam = lam_dict.get(feat)
                    if lam is not None:
                        try:
                            df_processed[feat] = boxcox(df_processed[feat], lmbda=lam)
                        except ValueError:
                            df_processed[feat] = np.log1p(df_processed[feat])
                    else:
                        df_processed[feat] = np.log1p(df_processed[feat])

        return df_processed

    st.title("House Price Predictions")
    st.header("Inherited Houses")
    st.write("""
    ## Predicted Sale Prices for Inherited Houses
    In this section, we provide estimated sale prices for the inherited houses. Utilizing our best-performing regression model, these predictions offer valuable insights into the potential market value of these properties.
    """)
    inherited_processed = preprocess_data_tab3(inherited_houses, data_reference=data)
    if selected_features is None or len(selected_features) == 0:
        st.error("**Error:** No selected features found for prediction.")
    else:
        try:
            inherited_scaled = scaler.transform(inherited_processed[selected_features])
            if model_evaluation.empty:
                st.error("**Error:** Model evaluation results are empty.")
                st.stop()
            else:
                if 'Model' in model_evaluation.columns:
                    available_evaluations = model_evaluation[model_evaluation['Model'] != 'XGBoost']
                else:
                    available_evaluations = model_evaluation
                if available_evaluations.empty:
                    st.error("**Error:** No models available after excluding 'XGBoost'.")
                    st.stop()
                if 'RMSE' not in available_evaluations.columns or 'Model' not in available_evaluations.columns:
                    st.error("**Error:** 'RMSE' or 'Model' columns not found in the evaluation results.")
                    st.stop()
                best_model_row = available_evaluations.loc[available_evaluations['RMSE'].idxmin()]
                best_model_name = best_model_row['Model']
            if best_model_name not in models:
                st.error(f"**Error:** Best model '{best_model_name}' not found among loaded models.")
            else:
                selected_model = models[best_model_name]
                predictions_log = selected_model.predict(inherited_scaled)
                predictions_actual = np.expm1(predictions_log)
                predictions_actual[predictions_actual < 0] = 0
                inherited_processed['Predicted SalePrice'] = predictions_actual
                display_columns = ['Predicted SalePrice'] + list(selected_features)
                missing_cols = [col for col in display_columns if col not in inherited_processed.columns]
                if missing_cols:
                    st.warning(f"The following columns are missing in the inherited houses data: {missing_cols}")
                    display_columns = [col for col in display_columns if col in inherited_processed.columns]
                inherited_processed['Predicted SalePrice'] = inherited_processed['Predicted SalePrice'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(inherited_processed[display_columns].style.format({"Predicted SalePrice": lambda x: x}))
                total_predicted_price = predictions_actual.sum()
                st.success(f"The total predicted sale price for all inherited houses is **${total_predicted_price:,.2f}**.")
        except Exception as e:
            st.error(f"**Error during prediction:** {e}")

    st.header("Real-Time House Price Prediction")
    st.write("""
    ## Predict Sale Prices in Real-Time
    Harness the power of our predictive model by inputting specific house attributes to receive instant sale price estimates. This feature is particularly useful for assessing the value of a property based on its characteristics.
    """)

    def user_input_features():
        input_data = {}
        with st.form(key='house_features'):
            st.write("### Enter House Attributes")
            feature_groups = {
                'General': ['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd'],
                'Area': ['GrLivArea', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'LotArea', 'LotFrontage'],
                'Basement': ['BsmtFinType1', 'BsmtExposure', 'BsmtFinSF1', 'BsmtUnfSF'],
                'Garage': ['GarageFinish', 'GarageYrBlt', 'GarageArea'],
                'Porch/Deck': ['WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch'],
                'Other': ['BedroomAbvGr', 'KitchenQual', 'MasVnrArea'],
            }
            for group_name, features in feature_groups.items():
                st.subheader(group_name)
                cols = st.columns(2)
                idx = 0
                for feature in features:
                    if feature in feature_input_details:
                        details = feature_input_details[feature]
                        input_type = details['input_type']
                        label = details['label']
                        help_text = details['help_text']
                        with cols[idx % 2]:
                            if input_type == 'number_input':
                                input_data[feature] = st.number_input(
                                    label,
                                    min_value=details['min_value'],
                                    max_value=details['max_value'],
                                    value=details['value'],
                                    step=details['step'],
                                    help=help_text
                                )
                            elif input_type == 'slider':
                                input_data[feature] = st.slider(
                                    label,
                                    min_value=details['min_value'],
                                    max_value=details['max_value'],
                                    value=details['value'],
                                    step=details['step'],
                                    help=help_text
                                )
                            elif input_type == 'selectbox':
                                input_data[feature] = st.selectbox(
                                    label,
                                    options=details['options'],
                                    index=0,
                                    help=help_text
                                )
                        idx += 1
            submit_button = st.form_submit_button(label='Predict Sale Price')
        if submit_button:
            input_df = pd.DataFrame(input_data, index=[0])
            input_df = feature_engineering(input_df)
            return input_df
        else:
            return None

    user_input = user_input_features()
    if user_input is not None:
        try:
            user_processed = preprocess_data_tab3(user_input, data_reference=data)
            user_scaled = scaler.transform(user_processed[selected_features])
            user_pred_log = models[best_model_name].predict(user_scaled)
            user_pred_actual = np.expm1(user_pred_log)
            user_pred_actual[user_pred_actual < 0] = 0
            st.success(f"The predicted sale price is **${user_pred_actual[0]:,.2f}**.")
        except Exception as e:
            st.error(f"**Error during prediction:** {e}")

    st.write("""
    ### How It Works
    1. **Input Features:** Enter the specific attributes of the house you're evaluating.
    2. **Data Preprocessing:** The input data undergoes the same preprocessing steps as the training data to ensure consistency.
    3. **Feature Scaling:** Numerical features are scaled to match the scale of the training data, enhancing model performance.
    4. **Prediction:** The processed data is fed into the best-performing regression model to generate an estimated sale price.
    5. **Output:** Receive an instant prediction of the house's market value, aiding in informed decision-making.
    """)

