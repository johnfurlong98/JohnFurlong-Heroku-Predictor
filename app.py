# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import pickle
from pathlib import Path
import plotly.express as px

from dashboard.tabs.project_summary import show_page as show_project_summary
from dashboard.tabs.feature_correlations import show_page as show_feature_correlations
from dashboard.tabs.house_price_predictions import show_page as show_house_price_predictions
from dashboard.tabs.project_hypotheses import show_page as show_project_hypotheses
from dashboard.tabs.model_performance import show_page as show_model_performance

# --------------------------- #
#       Configuration          #
# --------------------------- #

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="House Price Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

def check_file_exists(file_path, description):
    if not file_path.exists():
        st.error(f"**Error:** The {description} file was not found at `{file_path}`.")
        st.stop()

@st.cache_data(show_spinner=False)
def load_data():
    data_path = BASE_DIR / 'dashboard' / 'notebook' / 'raw_data' / 'house_prices_records.csv'
    inherited_houses_path = BASE_DIR / 'dashboard' / 'notebook' / 'raw_data' / 'inherited_houses.csv'

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
    models_dir = BASE_DIR / 'dashboard' / 'notebook' / 'data' / 'models'

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

    scaler_path = models_dir / 'scaler.joblib'
    selected_features_path = models_dir / 'selected_features.pkl'
    skewed_features_path = models_dir / 'skewed_features.pkl'
    lam_dict_path = models_dir / 'lam_dict.pkl'
    feature_importances_path = models_dir / 'feature_importances.csv'
    model_evaluation_path = models_dir / 'model_evaluation.csv'
    train_test_data_path = models_dir / 'train_test_data.joblib'

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
    df = df.copy()
    df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
    df['Qual_TotalSF'] = df.get('OverallQual', 0) * df.get('TotalSF', 0)
    return df

@st.cache_data(show_spinner=False)
def preprocess_data(df, data_reference=None):
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

# Load data and models
data, inherited_houses = load_data()
data_original = data[['SalePrice']].copy()
(models, scaler, selected_features, skewed_features, lam_dict,
 feature_importances, model_evaluation, train_test_data) = load_models()

data = preprocess_data(data, data_reference=data)

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
        'max_value': 100000,
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
        'help_text': feature_metadata.get('OpenPorchSF', '')
    },
    'EnclosedPorch': {
        'input_type': 'number_input',
        'label': 'Enclosed Porch Area (sq ft)',
        'min_value': 0,
        'max_value': 10000,
        'value': int(data['EnclosedPorch'].median()) if 'EnclosedPorch' in data.columns else 0,
        'step': 1,
        'help_text': feature_metadata.get('EnclosedPorch', '')
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

st.markdown(
    """
    <style>
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

tabs = ["Project Summary", "Feature Correlations", "House Price Predictions", "Project Hypotheses", "Model Performance"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

with tab1:
    show_project_summary(
        data, data_original, models, scaler, selected_features, skewed_features,
        lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details
    )

with tab2:
    show_feature_correlations(
        data, data_original, models, scaler, selected_features, skewed_features,
        lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details
    )

with tab3:
    show_house_price_predictions(
        data, data_original, models, scaler, selected_features, skewed_features,
        lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details, inherited_houses
    )

with tab4:
    show_project_hypotheses(
        data, data_original, models, scaler, selected_features, skewed_features,
        lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details
    )

with tab5:
    show_model_performance(
        data, data_original, models, scaler, selected_features, skewed_features,
        lam_dict, feature_importances, model_evaluation, train_test_data, feature_input_details
    )
