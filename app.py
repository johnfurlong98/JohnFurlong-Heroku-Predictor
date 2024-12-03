# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from scipy.special import inv_boxcox  # Added for inverse Box-Cox transformation
import pickle
from pathlib import Path

# --------------------------- #
#       Configuration          #
# --------------------------- #

# Get the directory where the script is located
BASE_DIR = Path(__file__).resolve().parent

# Set up page configuration with custom theme
st.set_page_config(
    page_title="House Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------- #
#       Helper Functions       #
# --------------------------- #

def check_file_exists(file_path, description):
    """
    Checks if a given file exists. If not, displays an error and stops the app.
    """
    if not file_path.exists():
        st.error(f"**Error:** The {description} file was not found at `{file_path}`.")
        st.stop()

@st.cache_data(show_spinner=False)
def load_data():
    """
    Loads the main dataset and the inherited houses dataset.
    """
    data_path = BASE_DIR / 'dashboard' / 'notebook' / 'raw_data' / 'house_prices_records.csv'
    inherited_houses_path = BASE_DIR / 'dashboard' / 'notebook' / 'raw_data' / 'inherited_houses.csv'
    
    # Check if files exist
    check_file_exists(data_path, "house_prices_records.csv")
    check_file_exists(inherited_houses_path, "inherited_houses.csv")
    
    try:
        data = pd.read_csv(data_path)
    except Exception as e:
        st.error(f"**Error loading house_prices_records.csv:** {e}")
        st.stop()
    
    try:
        inherited_houses = pd.read_csv(inherited_houses_path)
    except Exception as e:
        st.error(f"**Error loading inherited_houses.csv:** {e}")
        st.stop()
    
    return data, inherited_houses

@st.cache_resource(show_spinner=False)
def load_models():
    """
    Loads all necessary models and related objects.
    """
    models_dir = BASE_DIR / 'dashboard' / 'notebook' / 'data' / 'models'
    
    # Check if models directory exists
    if not models_dir.exists():
        st.error(f"**Error:** The models directory was not found at `{models_dir}`.")
        st.stop()
    
    models = {}
    model_files = {
        'Linear Regression': 'linear_regression_model.joblib',
        'Ridge Regression': 'ridge_regression_model.joblib',
        'ElasticNet': 'elasticnet_model.joblib',
        'Lasso Regression': 'lasso_regression_model.joblib',
        'Gradient Boosting': 'gradient_boosting_model.joblib',
        'Random Forest': 'random_forest_model.joblib'
    }
    
    for name, filename in model_files.items():
        model_path = models_dir / filename
        check_file_exists(model_path, f"{filename}")
        try:
            models[name] = joblib.load(model_path)
        except Exception as e:
            st.error(f"**Error loading {filename}:** {e}")
            st.stop()
    
    # Load scaler and other related objects
    scaler_path = models_dir / 'scaler.joblib'
    selected_features_path = models_dir / 'selected_features.pkl'
    skewed_features_path = models_dir / 'skewed_features.pkl'
    lam_dict_path = models_dir / 'lam_dict.pkl'
    feature_importances_path = models_dir / 'feature_importances.csv'
    model_evaluation_path = models_dir / 'model_evaluation.csv'
    train_test_data_path = models_dir / 'train_test_data.joblib'
    
    # Check if all necessary files exist
    required_files = {
        "scaler.joblib": scaler_path,
        "selected_features.pkl": selected_features_path,
        "skewed_features.pkl": skewed_features_path,
        "lam_dict.pkl": lam_dict_path,
        "feature_importances.csv": feature_importances_path,
        "model_evaluation.csv": model_evaluation_path,
        "train_test_data.joblib": train_test_data_path
    }
    
    for description, path in required_files.items():
        check_file_exists(path, description)
    
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"**Error loading scaler.joblib:** {e}")
        st.stop()
    
    try:
        selected_features = pickle.load(open(selected_features_path, 'rb'))
    except Exception as e:
        st.error(f"**Error loading selected_features.pkl:** {e}")
        st.stop()
    
    try:
        skewed_features = pickle.load(open(skewed_features_path, 'rb'))
    except Exception as e:
        st.error(f"**Error loading skewed_features.pkl:** {e}")
        st.stop()
    
    try:
        lam_dict = pickle.load(open(lam_dict_path, 'rb'))
    except Exception as e:
        st.error(f"**Error loading lam_dict.pkl:** {e}")
        st.stop()
    
    try:
        feature_importances = pd.read_csv(feature_importances_path)
    except Exception as e:
        st.error(f"**Error loading feature_importances.csv:** {e}")
        st.stop()
    
    try:
        model_evaluation = pd.read_csv(model_evaluation_path)
    except Exception as e:
        st.error(f"**Error loading model_evaluation.csv:** {e}")
        st.stop()
    
    try:
        train_test_data = joblib.load(train_test_data_path)
    except Exception as e:
        st.error(f"**Error loading train_test_data.joblib:** {e}")
        st.stop()
    
    return models, scaler, selected_features, skewed_features, lam_dict, feature_importances, model_evaluation, train_test_data

@st.cache_data(show_spinner=False)
def feature_engineering(df):
    """
    Performs feature engineering by creating new features.
    """
    df = df.copy()
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['Qual_TotalSF'] = df.get('OverallQual', 0) * df.get('TotalSF', 0)
    return df

@st.cache_data(show_spinner=False)
def preprocess_data(df, data_reference=None):
    """
    Preprocesses the input data by handling missing values, encoding categorical variables,
    and transforming skewed features.
    
    **Note:** The 'SalePrice' is excluded from skewed feature transformations to preserve its original scale for visualization purposes.
    """
    df_processed = df.copy()
    
    # Map full words back to codes
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
    
    # Handle missing values
    zero_fill_features = ['2ndFlrSF', 'EnclosedPorch', 'MasVnrArea', 'WoodDeckSF',
                          'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', 'BsmtUnfSF']
    for feature in zero_fill_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(0)
    
    # Fill categorical features
    categorical_mode_fill = {
        'BsmtFinType1': 'None',
        'GarageFinish': 'Unf',
        'BsmtExposure': 'No',
        'KitchenQual': 'TA'
    }
    for feature, value in categorical_mode_fill.items():
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(value)
    
    # Fill numerical features using median from training data
    numerical_median_fill = ['BedroomAbvGr', 'GarageYrBlt', 'LotFrontage', 
                             'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd']
    for feature in numerical_median_fill:
        if feature in df_processed.columns:
            if data_reference is not None and feature in data_reference.columns:
                median_value = data_reference[feature].median()
            else:
                median_value = df_processed[feature].median()
            df_processed[feature] = df_processed[feature].fillna(median_value)
    
    # Encode categorical features
    ordinal_mappings = {
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'KitchenQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3}
    }
    for col, mapping in ordinal_mappings.items():
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(mapping)
    
    # Feature engineering
    df_processed = feature_engineering(df_processed)
    
    # Transform skewed features, excluding 'SalePrice' to preserve original scale for visualization
    for feat in skewed_features:
        if feat == 'SalePrice':
            continue  # Skip transforming 'SalePrice'
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

# --------------------------- #
#       Load Data & Models     #
# --------------------------- #

# Load data
data, inherited_houses = load_data()

# Create a copy of the original SalePrice before preprocessing for visualization
data_original = data[['SalePrice']].copy()

# Load models and related data
(models, scaler, selected_features, skewed_features, lam_dict, 
 feature_importances, model_evaluation, train_test_data) = load_models()

# Preprocess the main data
data = preprocess_data(data, data_reference=data)

# --------------------------- #
#       Feature Metadata       #
# --------------------------- #

# Metadata for features (from the provided metadata)
feature_metadata = {
    '1stFlrSF': 'First Floor square feet',
    '2ndFlrSF': 'Second floor square feet',
    'BedroomAbvGr': 'Bedrooms above grade (0 - 8)',
    'BsmtExposure': 'Walkout or garden level walls',
    'BsmtFinType1': 'Rating of basement finished area',
    'BsmtFinSF1': 'Type 1 finished square feet',
    'BsmtUnfSF': 'Unfinished basement area',
    'TotalBsmtSF': 'Total basement area',
    'GarageArea': 'Garage size in square feet',
    'GarageFinish': 'Garage interior finish',
    'GarageYrBlt': 'Year garage was built',
    'GrLivArea': 'Above grade living area',
    'KitchenQual': 'Kitchen quality',
    'LotArea': 'Lot size in square feet',
    'LotFrontage': 'Linear feet of street connected to property',
    'MasVnrArea': 'Masonry veneer area',
    'EnclosedPorch': 'Enclosed porch area',
    'OpenPorchSF': 'Open porch area',
    'OverallCond': 'Overall condition rating (1 - 10)',
    'OverallQual': 'Overall material and finish rating (1 - 10)',
    'WoodDeckSF': 'Wood deck area',
    'YearBuilt': 'Original construction date',
    'YearRemodAdd': 'Remodel date',
    'TotalSF': 'Total square feet of house (including basement)',
    'Qual_TotalSF': 'Product of OverallQual and TotalSF'
}

# --------------------------- #
#   Feature Input Definitions #
# --------------------------- #

# Define feature input details for the user input form
feature_input_details = {
    'OverallQual': {
        'input_type': 'slider',
        'label': 'Overall Quality (1-10)',
        'min_value': 1,
        'max_value': 10,
        'value': 5,
        'step': 1,
        'help_text': feature_metadata['OverallQual']
    },
    'OverallCond': {
        'input_type': 'slider',
        'label': 'Overall Condition (1-10)',
        'min_value': 1,
        'max_value': 10,
        'value': 5,
        'step': 1,
        'help_text': feature_metadata['OverallCond']
    },
    'YearBuilt': {
        'input_type': 'slider',
        'label': 'Year Built',
        'min_value': 1872,
        'max_value': 2024,
        'value': 1975,
        'step': 1,
        'help_text': feature_metadata['YearBuilt']
    },
    'YearRemodAdd': {
        'input_type': 'slider',
        'label': 'Year Remodeled',
        'min_value': 1950,
        'max_value': 2024,
        'value': 1997,
        'step': 1,
        'help_text': feature_metadata['YearRemodAdd']
    },
    'GrLivArea': {
        'input_type': 'number_input',
        'label': 'Above Grade Living Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 750,
        'step': 1,
        'help_text': feature_metadata['GrLivArea']
    },
    '1stFlrSF': {
        'input_type': 'number_input',
        'label': 'First Floor Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 600,
        'step': 1,
        'help_text': feature_metadata['1stFlrSF']
    },
    '2ndFlrSF': {
        'input_type': 'number_input',
        'label': 'Second Floor Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['2ndFlrSF']
    },
    'TotalBsmtSF': {
        'input_type': 'number_input',
        'label': 'Total Basement Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['TotalBsmtSF']
    },
    'LotArea': {
        'input_type': 'number_input',
        'label': 'Lot Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 1300,
        'step': 1,
        'help_text': feature_metadata['LotArea']
    },
    'LotFrontage': {
        'input_type': 'number_input',
        'label': 'Lot Frontage (linear ft)',
        'min_value': 0,
        'max_value': 1000,
        'value': 25,
        'step': 1,
        'help_text': feature_metadata['LotFrontage']
    },
    'BsmtFinType1': {
        'input_type': 'selectbox',
        'label': 'Basement Finish Type',
        'options': [
            'No Basement',
            'Unfinished',
            'Low Quality',
            'Rec Room',
            'Basement Living Quarters',
            'Average Living Quarters',
            'Good Living Quarters'
        ],
        'help_text': feature_metadata['BsmtFinType1']
    },
    'BsmtExposure': {
        'input_type': 'selectbox',
        'label': 'Basement Exposure',
        'options': [
            'No Basement',
            'No Exposure',
            'Minimum Exposure',
            'Average Exposure',
            'Good Exposure'
        ],
        'help_text': feature_metadata['BsmtExposure']
    },
    'BsmtFinSF1': {
        'input_type': 'number_input',
        'label': 'Finished Basement Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['BsmtFinSF1']
    },
    'BsmtUnfSF': {
        'input_type': 'number_input',
        'label': 'Unfinished Basement Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['BsmtUnfSF']
    },
    'GarageFinish': {
        'input_type': 'selectbox',
        'label': 'Garage Finish',
        'options': [
            'No Garage',
            'Unfinished',
            'Rough Finished',
            'Finished'
        ],
        'help_text': feature_metadata['GarageFinish']
    },
    'GarageYrBlt': {
        'input_type': 'slider',
        'label': 'Garage Year Built',
        'min_value': 1900,
        'max_value': 2024,
        'value': 1990,
        'step': 1,
        'help_text': feature_metadata['GarageYrBlt']
    },
    'GarageArea': {
        'input_type': 'number_input',
        'label': 'Garage Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 50,
        'step': 1,
        'help_text': feature_metadata['GarageArea']
    },
    'WoodDeckSF': {
        'input_type': 'number_input',
        'label': 'Wood Deck Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': 0,
        'step': 1,
        'help_text': feature_metadata['WoodDeckSF']
    },
    'OpenPorchSF': {
        'input_type': 'number_input',
        'label': 'Open Porch Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': int(data['OpenPorchSF'].median()) if 'OpenPorchSF' in data.columns else 0,
        'step': 1,
        'help_text': feature_metadata['OpenPorchSF']
    },
    'EnclosedPorch': {
        'input_type': 'number_input',
        'label': 'Enclosed Porch Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': int(data['EnclosedPorch'].median()) if 'EnclosedPorch' in data.columns else 0,
        'step': 1,
        'help_text': feature_metadata['EnclosedPorch']
    },
    'BedroomAbvGr': {
        'input_type': 'slider',
        'label': 'Bedrooms Above Grade',
        'min_value': 0,
        'max_value': 8,
        'value': int(data['BedroomAbvGr'].median()) if 'BedroomAbvGr' in data.columns else 3,
        'step': 1,
        'help_text': feature_metadata['BedroomAbvGr']
    },
    'KitchenQual': {
        'input_type': 'selectbox',
        'label': 'Kitchen Quality',
        'options': [
            'Poor',
            'Fair',
            'Typical/Average',
            'Good',
            'Excellent'
        ],
        'help_text': feature_metadata['KitchenQual']
    },
    'MasVnrArea': {
        'input_type': 'number_input',
        'label': 'Masonry Veneer Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': int(data['MasVnrArea'].median()) if 'MasVnrArea' in data.columns else 0,
        'step': 1,
        'help_text': feature_metadata['MasVnrArea']
    },
}

# --------------------------- #
#       Custom Styling         #
# --------------------------- #

# Apply custom CSS for enhanced UI (optional)
st.markdown(
    """
    <style>
    /* Custom CSS for the tabs and form */
    .stTabs [role="tablist"] {
        justify-content: center;
    }
    .st-form {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------- #
#          Main App            #
# --------------------------- #

# Create tabs for navigation
tabs = ["Project Summary", "Feature Correlations", "House Price Predictions", "Project Hypotheses", "Model Performance"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

# --------------------------- #
#      Project Summary Tab     #
# --------------------------- #

with tab1:
    st.title("House Price Prediction Dashboard")
    st.write("""
    ## Project Summary

    Welcome to the House Price Prediction Dashboard. This project aims to build a predictive model to estimate the sale prices of houses based on various features. By analyzing the data and developing robust models, we provide insights into the key factors that influence house prices.

    **Key Objectives:**

    - **Data Analysis and Preprocessing:** Understand and prepare the data for modeling.
    - **Feature Engineering:** Create new features to improve model performance.
    - **Model Development:** Train and evaluate multiple regression models.
    - **Deployment:** Develop an interactive dashboard for predictions and insights.

    **Instructions:**

    - Use the tabs at the top to navigate between different sections.
    - Explore data correlations, make predictions, and understand the model performance.
    """)

# --------------------------- #
#    Feature Correlations Tab  #
# --------------------------- #

with tab2:
    st.title("Feature Correlations")
    st.write("""
    ## Understanding Feature Relationships

    Understanding how different features correlate with the sale price is crucial for building an effective predictive model. This section visualizes the relationships between key property attributes and the sale price.
    """)

    # Prepare data for correlation using original SalePrice
    data_for_corr = pd.concat([data.drop('SalePrice', axis=1), data_original], axis=1)

    # Compute correlation matrix
    corr_matrix = data_for_corr.corr()
    if 'SalePrice' not in corr_matrix.columns:
        st.error("**Error:** 'SalePrice' column not found in the dataset.")
    else:
        # Select features with high correlation (absolute value > 0.5) with original SalePrice
        top_corr_features = corr_matrix.index[abs(corr_matrix['SalePrice']) > 0.5].tolist()

        if len(top_corr_features) == 0:
            st.warning("**Warning:** No features found with a correlation greater than 0.5 with 'SalePrice'.")
        else:
            st.write("""
            ### Top Correlated Features with Sale Price
            The heatmap below shows the correlation coefficients between the sale price and other features. Features with higher absolute correlation values have a stronger relationship with the sale price.
            """)

            # Plot correlation heatmap using original SalePrice
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

            # Additional visualization: Pairplot with top features
            st.write("### Pairplot of Top Correlated Features")
            # Select top 5 features excluding 'SalePrice'
            top_features = [feat for feat in top_corr_features if feat != 'SalePrice'][:5]
            if len(top_features) == 0:
                st.warning("**Warning:** Not enough features to create a pairplot.")
            else:
                # To optimize performance, sample the data if it's too large
                sample_size = 500  # Adjust based on performance
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

# --------------------------- #
#  House Price Predictions Tab #
# --------------------------- #

with tab3:
    st.title("House Price Predictions")

    # Inherited Houses Predictions
    st.header("Inherited Houses")
    st.write("""
    ## Predicted Sale Prices for Inherited Houses

    In this section, we provide estimated sale prices for the inherited houses. Utilizing our best-performing regression model, these predictions offer valuable insights into the potential market value of these properties.
    """)

    # Preprocess and predict for inherited houses
    inherited_processed = preprocess_data(inherited_houses, data_reference=data)
    if selected_features is None or len(selected_features) == 0:
        st.error("**Error:** No selected features found for prediction.")
    else:
        try:
            inherited_scaled = scaler.transform(inherited_processed[selected_features])
            # Determine best model based on evaluation, excluding 'XGBoost' if present
            if model_evaluation.empty:
                st.error("**Error:** Model evaluation results are empty.")
                st.stop()
            else:
                # Exclude 'XGBoost' if it's still present
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
                predictions_actual[predictions_actual < 0] = 0  # Handle negative predictions

                # Add predictions to the processed DataFrame
                inherited_processed['Predicted SalePrice'] = predictions_actual

                # Display the DataFrame with the selected features
                display_columns = ['Predicted SalePrice'] + list(selected_features)
                missing_cols = [col for col in display_columns if col not in inherited_processed.columns]
                if missing_cols:
                    st.warning(f"The following columns are missing in the inherited houses data: {missing_cols}")
                    display_columns = [col for col in display_columns if col in inherited_processed.columns]

                # Format the 'Predicted SalePrice' as currency
                inherited_processed['Predicted SalePrice'] = inherited_processed['Predicted SalePrice'].apply(lambda x: f"${x:,.2f}")

                st.dataframe(inherited_processed[display_columns].style.format({"Predicted SalePrice": lambda x: x}))
                total_predicted_price = predictions_actual.sum()
                st.success(f"The total predicted sale price for all inherited houses is **${total_predicted_price:,.2f}**.")
        except Exception as e:
            st.error(f"**Error during prediction:** {e}")

    # Real-Time Prediction
    st.header("Real-Time House Price Prediction")
    st.write("""
    ## Predict Sale Prices in Real-Time

    Harness the power of our predictive model by inputting specific house attributes to receive instant sale price estimates. This feature is particularly useful for assessing the value of a property based on its characteristics.
    """)

    def user_input_features():
        """
        Creates a form for users to input house features and returns the input data as a DataFrame.
        """
        input_data = {}
        with st.form(key='house_features'):
            st.write("### Enter House Attributes")
            # Group features into sections
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
                                    index=0,  # Default to first option
                                    help=help_text
                                )
                        idx += 1  # Increment idx to switch columns

            submit_button = st.form_submit_button(label='Predict Sale Price')

        if submit_button:
            input_df = pd.DataFrame(input_data, index=[0])
            # Calculate engineered features
            input_df = feature_engineering(input_df)
            return input_df
        else:
            return None

    user_input = user_input_features()
    if user_input is not None:
        try:
            user_processed = preprocess_data(user_input, data_reference=data)
            user_scaled = scaler.transform(user_processed[selected_features])
            user_pred_log = models[best_model_name].predict(user_scaled)  # Use the best model
            user_pred_actual = np.expm1(user_pred_log)
            user_pred_actual[user_pred_actual < 0] = 0  # Handle negative predictions
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

# --------------------------- #
#      Project Hypotheses Tab  #
# --------------------------- #

with tab4:
    st.title("Project Hypotheses")
    st.write("""
    ## Hypothesis Validation

    In this section, we explore the foundational hypotheses that guided our analysis and modeling efforts. Each hypothesis is validated using statistical and machine learning techniques, providing a deeper understanding of the factors influencing house prices.
    """)

    # Primary Hypotheses
    st.subheader("### Primary Hypotheses")

    st.write("""
    **Hypothesis 1:** *Higher overall quality of the house leads to a higher sale price.*
    
    - **Rationale:** Quality metrics such as construction standards, materials used, and overall maintenance directly impact the desirability and value of a property.
    - **Validation:** The `OverallQual` feature shows a strong positive correlation with the sale price, confirming this hypothesis.
    """)

    st.write("""
    **Hypothesis 2:** *Larger living areas result in higher sale prices.*
    
    - **Rationale:** Square footage is a fundamental indicator of a property's size and usability. Larger homes typically offer more living space, which is highly valued in the real estate market.
    - **Validation:** Features like `GrLivArea` and `TotalSF` have high correlations with the sale price, supporting this hypothesis.
    """)

    st.write("""
    **Hypothesis 3:** *Recent renovations positively impact the sale price.*
    
    - **Rationale:** Modern updates and renovations can enhance a property's appeal, functionality, and energy efficiency, thereby increasing its market value.
    - **Validation:** The `YearRemodAdd` feature correlates with the sale price, indicating that more recent remodels can increase the house value.
    """)

    st.write("""
    **Hypothesis 4:** *The presence and quality of a garage significantly influence the sale price.*
    
    - **Rationale:** Garages add convenience and storage space, enhancing the property's functionality. Higher-quality garages are often associated with better construction and maintenance.
    - **Validation:** Features like `GarageArea` and `GarageFinish` show positive correlations with the sale price, validating this hypothesis.
    """)

    st.write("""
    **Hypothesis 5:** *Lot size and frontage are key determinants of a house's market value.*
    
    - **Rationale:** Larger lots provide more outdoor space, which is desirable for families and can offer potential for future expansions or landscaping.
    - **Validation:** The `LotArea` and `LotFrontage` features have significant positive correlations with the sale price, supporting this hypothesis.
    """)

    st.write("""
    **Hypothesis 6:** *Kitchen quality is a strong predictor of a house's sale price.*
    
    - **Rationale:** Kitchens are central to modern living, and high-quality kitchens with modern appliances and finishes are highly sought after.
    - **Validation:** The `KitchenQual` feature demonstrates a positive correlation with the sale price, confirming its importance.
    """)

    st.write("""
    **Hypothesis 7:** *The number of bedrooms above grade influences the sale price.*
    
    - **Rationale:** More bedrooms can accommodate larger families, increasing the property's appeal to potential buyers.
    - **Validation:** The `BedroomAbvGr` feature shows a positive correlation with the sale price, supporting this hypothesis.
    """)

    # Visualization for Hypotheses
    st.write("### Visualization of Hypotheses")

    # OverallQual vs SalePrice_original
    st.write("#### SalePrice vs OverallQual")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=data_for_corr, palette='Set2')  # Using original SalePrice
    plt.title('SalePrice vs OverallQual', fontsize=16)
    plt.xlabel('Overall Quality', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.tight_layout()
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:**
    
    The boxplot illustrates a clear trend where houses with higher overall quality ratings command higher sale prices. This strong positive relationship validates our first hypothesis, emphasizing the significant impact of overall quality on property value.
    """)

    # TotalSF vs SalePrice_original
    st.write("#### SalePrice vs TotalSF")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TotalSF', y='SalePrice', data=data_for_corr, hue='OverallQual', palette='coolwarm', alpha=0.6)  # Using original SalePrice
    plt.title('SalePrice vs TotalSF', fontsize=16)
    plt.xlabel('Total Square Footage', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.legend(title='Overall Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:**
    
    The scatter plot reveals a positive correlation between total square footage and sale price. Larger homes with more square footage tend to have higher sale prices, supporting our second hypothesis that size is a key determinant of property value.
    """)

    # YearRemodAdd vs SalePrice_original
    st.write("#### SalePrice vs YearRemodeled")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='YearRemodAdd', y='SalePrice', data=data_for_corr, color='green', ci=None)  # Using original SalePrice
    plt.title('SalePrice vs Year Remodeled', fontsize=16)
    plt.xlabel('Year Remodeled', fontsize=12)
    plt.ylabel('Average Sale Price (USD)', fontsize=12)
    plt.tight_layout()
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:**
    
    The line plot shows an upward trend in sale prices with more recent remodeling years. This indicates that recent renovations and updates contribute positively to the property's market value, thereby validating our third hypothesis.
    """)

    # GarageArea vs SalePrice_original
    st.write("#### SalePrice vs GarageArea")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='GarageArea', y='SalePrice', data=data_for_corr, hue='GarageFinish', palette='viridis', alpha=0.6)  # Using original SalePrice
    plt.title('SalePrice vs GarageArea', fontsize=16)
    plt.xlabel('Garage Area (sq ft)', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.legend(title='Garage Finish', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:**
    
    The scatter plot indicates that larger garage areas are associated with higher sale prices. Additionally, the quality of the garage finish further enhances the property's value. These observations confirm our fourth hypothesis, highlighting the significant role of garage features in determining house prices.
    """)

    # LotArea vs SalePrice_original
    st.write("#### SalePrice vs LotArea")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='LotArea', y='SalePrice', data=data_for_corr, hue='BedroomAbvGr', palette='magma', alpha=0.6)  # Using original SalePrice
    plt.title('SalePrice vs LotArea', fontsize=16)
    plt.xlabel('Lot Area (sq ft)', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.legend(title='Bedrooms Above Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:**
    
    The positive relationship between lot area and sale price is evident from the scatter plot. Larger lots provide more outdoor space and potential for future expansions, thereby increasing the property's appeal and market value. This supports our fifth hypothesis regarding the importance of lot size and frontage in determining house prices.
    """)

    # KitchenQual vs SalePrice_original
    st.write("#### SalePrice vs KitchenQual")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='KitchenQual', y='SalePrice', data=data_for_corr, palette='Pastel1')  # Using original SalePrice
    plt.title('SalePrice vs KitchenQual', fontsize=16)
    plt.xlabel('Kitchen Quality', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.tight_layout()
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:**
    
    The boxplot clearly shows that houses with higher kitchen quality ratings have significantly higher sale prices. This strong positive association validates our sixth hypothesis, emphasizing the critical role of kitchen quality in enhancing property value.
    """)

    # BedroomAbvGr vs SalePrice_original
    st.write("#### SalePrice vs BedroomAbvGr")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='BedroomAbvGr', y='SalePrice', data=data_for_corr, palette='Set3')  # Using original SalePrice
    plt.title('SalePrice vs BedroomAbvGr', fontsize=16)
    plt.xlabel('Bedrooms Above Grade', fontsize=12)
    plt.ylabel('Sale Price (USD)', fontsize=12)
    plt.tight_layout()
    # Format y-axis with dollar signs
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '${:,.0f}'.format(x)))
    st.pyplot(plt)
    st.write("""
    **Conclusion:**
    
    The boxplot indicates a positive trend where an increasing number of bedrooms above grade correlates with higher sale prices. This finding supports our seventh hypothesis, demonstrating that more bedrooms enhance the property's appeal and market value.
    """)

    st.write("""
    ### Summary of Hypothesis Validations

    The visualizations above support our hypotheses, indicating that overall quality, living area, recent renovations, garage features, lot size, kitchen quality, and the number of bedrooms above grade are significant determinants of house sale prices. These insights can guide stakeholders in making informed decisions regarding property investments, renovations, and marketing strategies.
    """)

# --------------------------- #
#    Model Performance Tab     #
# --------------------------- #

with tab5:
    st.title("Model Performance")
    st.header("Performance Metrics")
    results_df = model_evaluation

    if results_df.empty:
        st.warning("**Warning:** Model evaluation results are empty.")
    else:
        # Exclude 'XGBoost' from evaluation results if present
        results_df_filtered = results_df[results_df['Model'] != 'XGBoost']
        if results_df_filtered.empty:
            st.error("**Error:** No models available after excluding 'XGBoost'.")
        else:
            # Check if necessary columns are present
            if 'RMSE' not in results_df_filtered.columns or 'Model' not in results_df_filtered.columns:
                st.error("**Error:** 'RMSE' or 'Model' columns not found in the evaluation results.")
            else:
                st.write("""
                ### Model Evaluation Metrics

                The table below presents the performance metrics of various regression models. These metrics help in assessing the accuracy and reliability of each model.
                """)

                # Display the evaluation table with formatted columns
                st.dataframe(results_df_filtered.style.format({'MAE': '${:,.2f}', 'RMSE': '${:,.2f}', 'R² Score': '{:.4f}'}))

                # Determine best model based on RMSE
                best_model_row = results_df_filtered.loc[results_df_filtered['RMSE'].idxmin()]
                best_model_name = best_model_row['Model']
                st.write(f"### Best Performing Model: **{best_model_name}**")
                st.write(f"""
                Based on the RMSE metric, **{best_model_name}** emerges as the top-performing model. It strikes an optimal balance between minimizing prediction errors and maintaining computational efficiency.
                """)

                st.write("""
                ### Understanding the Metrics

                - **Mean Absolute Error (MAE):** Represents the average absolute difference between predicted and actual sale prices. A lower MAE indicates better model accuracy.
                - **Root Mean Squared Error (RMSE):** Similar to MAE but penalizes larger errors more heavily. Lower RMSE values signify a more precise model.
                - **R² Score:** Measures the proportion of variance in the sale price that is predictable from the features. An R² closer to 1 indicates a model that explains a large portion of the variance.
                """)

                st.header("Detailed Pipeline Explanation")
                st.write("""
                The success of our predictive model hinges on a meticulously crafted pipeline that encompasses data preprocessing, feature engineering, model training, and evaluation. Here's an in-depth look into each stage:
                """)

                st.write("""
                ### 1. Data Collection and Understanding
                - **Datasets Used:**
                  - **Historical House Sale Data:** Contains features and sale prices of houses.
                  - **Inherited Houses Data:** Contains features of houses for which sale prices need to be predicted.
                - **Exploratory Data Analysis (EDA):**
                  - Assessed data shapes, types, and initial statistics.
                  - Identified potential relationships and patterns.
                """)

                st.write("""
                ### 2. Data Cleaning
                - **Handling Missing Values:**
                  - **Numerical Features:** Filled missing values with zeros or the median of the feature.
                  - **Categorical Features:** Filled missing values with the mode or a default category.
                  - **Verification:** Confirmed that no missing values remained after imputation.
                """)

                st.write("""
                ### 3. Feature Engineering
                - **Categorical Encoding:**
                  - Applied ordinal encoding to convert categorical features into numerical values based on domain knowledge.
                - **Creation of New Features:**
                  - **TotalSF:** Combined total square footage of the house, including basement and above-ground areas.
                  - **Qual_TotalSF:** Product of `OverallQual` and `TotalSF` to capture the combined effect of size and quality on sale price.
                """)

                st.write("""
                ### 4. Feature Transformation
                - **Addressing Skewness:**
                  - Identified skewed features using skewness metrics.
                  - Applied log transformation or Box-Cox transformation to normalize distributions.
                """)

                st.write("""
                ### 5. Feature Selection
                - **Random Forest Feature Importances:**
                  - Utilized a Random Forest model to evaluate the importance of each feature in predicting sale prices.
                  - Selected top-performing features that significantly contribute to the model's predictive accuracy.
                """)

                st.write("""
                ### 6. Data Scaling
                - **Standardization:**
                  - Employed `StandardScaler` to standardize numerical features, ensuring they have a mean of 0 and a standard deviation of 1.
                  - Essential for models sensitive to feature scales, such as Ridge and Lasso regressions.
                """)

                st.write("""
                ### 7. Model Training
                - **Algorithms Used:**
                  - Linear Regression, Ridge Regression, Lasso Regression, ElasticNet, Random Forest, Gradient Boosting.
                - **Hyperparameter Tuning:**
                  - Conducted using cross-validation techniques to identify optimal model parameters, ensuring generalizability and minimizing overfitting.
                """)

                st.write("""
                ### 8. Model Evaluation
                - **Performance Metrics:**
                  - **Mean Absolute Error (MAE):** Provides a straightforward measure of average prediction error.
                  - **Root Mean Squared Error (RMSE):** Offers insight into the magnitude of errors, with higher penalties for larger deviations.
                  - **R² Score:** Indicates the proportion of variance in the sale price explained by the model, with higher values signifying better fit.
                - **Best Model Selection:**
                  - Evaluated models based on RMSE and R² Score, selecting the one that demonstrates the lowest error and highest explanatory power.
                """)

                st.write("""
                ### 9. Deployment
                - **Interactive Dashboard:**
                  - Developed using Streamlit to provide a user-friendly interface for real-time interaction.
                  - Allows users to input house features and obtain immediate sale price estimates.
                  - Incorporates visual insights into feature correlations, model performance, and hypothesis validations to enhance user understanding.
                """)

                st.header("Feature Importances")
                # Display feature importances from the best-performing model
                if best_model_name in models:
                    # Assuming feature_importances.csv has 'Feature' and 'Importance' columns
                    feature_importances_best = feature_importances.copy()

                    if feature_importances_best.empty:
                        st.warning(f"**Warning:** Feature importances for the model '{best_model_name}' are not available.")
                    else:
                        plt.figure(figsize=(12, 8))
                        sns.barplot(x='Importance', y='Feature', data=feature_importances_best.sort_values(by='Importance', ascending=False), palette='viridis')
                        plt.title(f'Feature Importances from {best_model_name}', fontsize=16)
                        plt.xlabel('Importance', fontsize=12)
                        plt.ylabel('Feature', fontsize=12)
                        plt.tight_layout()
                        st.pyplot(plt)

                        st.write("""
                        The bar chart above illustrates the relative importance of each feature in predicting the sale price. Notably, features like `GrLivArea`, `OverallQual`, and `TotalSF` are among the most significant contributors, reaffirming their critical role in determining property values.
                        """)
                else:
                    st.warning(f"**Warning:** Feature importances for the model '{best_model_name}' are not available.")

                st.header("Feature Relationships (Excluding OverallQual)")
                # Select features excluding 'OverallQual'
                feature_relations = [feat for feat in selected_features if feat != 'OverallQual']
                if len(feature_relations) < 2:
                    st.warning("**Warning:** Not enough features to create a feature relationships plot excluding 'OverallQual'.")
                else:
                    # Select top 5 features excluding 'OverallQual' for clarity
                    top_features_rel = feature_relations[:5]
                    # Prepare data for pairplot
                    pairplot_data = data_for_corr[top_features_rel + ['SalePrice']]

                    # To optimize performance, sample the data if it's too large
                    sample_size = 500  # Adjust based on performance
                    if pairplot_data.shape[0] > sample_size:
                        pairplot_data = pairplot_data.sample(n=sample_size, random_state=42)

                    sns.set(style="ticks")
                    pairplot_fig = sns.pairplot(pairplot_data, diag_kind='kde', height=2.5)
                    plt.suptitle('Feature Relationships Excluding OverallQual', y=1.02)
                    st.pyplot(pairplot_fig)

                    st.write("""
                    **Feature Relationships:**

                    The pairplot above visualizes the relationships between selected features and the sale price, excluding the top feature `OverallQual`. This provides a clearer view of how other significant features interact with the sale price.
                    """)

                    st.header("Residual Analysis")
                    if selected_model and train_test_data:
                        try:
                            y_pred_log = selected_model.predict(X_test)
                            y_pred_actual = np.expm1(y_pred_log)
                            y_pred_actual[y_pred_actual < 0] = 0  # Handle negative predictions
                            y_test_actual = np.expm1(y_test)
                            residuals = y_test_actual - y_pred_actual

                            plt.figure(figsize=(10, 6))
                            sns.histplot(residuals, kde=True, color='coral', bins=30)
                            plt.title('Residuals Distribution', fontsize=16)
                            plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
                            plt.ylabel('Frequency', fontsize=12)
                            plt.tight_layout()
                            st.pyplot(plt)

                            st.write("""
                            *Understanding Residuals:*
                            
                            Residuals represent the differences between actual and predicted sale prices. Analyzing their distribution helps in assessing the model's performance and identifying any underlying patterns or biases.
                            
                            *Key Insights:*
                            - *Normal Distribution:* Residuals are approximately normally distributed around zero, indicating that the model's errors are random and unbiased.
                            - *Symmetry:* The symmetrical spread suggests consistent performance across different sale price ranges.
                            - *Outliers:* Minimal skewness and few outliers indicate that the model handles most data points effectively, with only a handful of predictions deviating significantly.
                            """)
                        except Exception as e:
                            st.error(f"*Error during Residual Analysis plotting:* {e}")
                    else:
                        st.warning("*Warning:* Cannot perform residual analysis without the selected model and necessary data.")
    st.write("""
    ### Conclusion

    The comprehensive evaluation of our regression models underscores the effectiveness of our predictive pipeline. By meticulously preprocessing data, engineering relevant features, and selecting robust models, we've achieved high prediction accuracy and reliability. The insights derived from feature importance and residual analysis further validate our approach, ensuring that the dashboard provides meaningful and actionable information to its users.

    **Next Steps:**

    - **Data Enrichment:** Incorporate additional features such as geographical location, proximity to amenities, and economic indicators to enhance model performance.
    - **Model Expansion:** Explore and integrate more sophisticated models or ensemble techniques to capture complex data patterns.
    - **User Feedback:** Gather feedback from users to identify areas of improvement and potential new features for the dashboard.
    - **Continuous Monitoring:** Implement mechanisms to monitor model performance over time, ensuring sustained accuracy and relevance.
    """)

# --------------------------- #
#          End of App          #
# --------------------------- #