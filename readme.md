
# House Price Prediction Dashboard by John Furlong

## Introduction

This House Price Prediction Dashboard is an end-to-end Data Science and Machine Learning project designed to estimate residential property sale prices in Ames, Iowa. By applying the CRISP-DM methodology, leveraging Machine Learning (ML) models, and incorporating user-centered design, this solution addresses real-world business needs. The dashboard offers both data-driven insights into which features most influence property values and an interactive tool to generate real-time predictions for any given property’s attributes.

The primary users—homeowners, real estate agents, investors—benefit from immediate, reliable price estimates and clarity on the driving factors behind property values. The project demonstrates how domain understanding, robust data preprocessing, careful feature engineering, model evaluation, and user-friendly deployment converge to deliver actionable insights.

---

## Dataset Content

**Source:** [Kaggle Ames Housing Dataset](https://www.kaggle.com/codeinstitute/housing-prices-data)

The dataset includes nearly 1,500 housing records from Ames, Iowa, detailing structural and qualitative features such as basement finish quality, garage size, kitchen quality, lot dimensions, porch areas, and more. Each record includes a corresponding sale price, making it possible to analyze correlations and build predictive models.

- **Mechanism:** Instead of relying solely on local CSV files, we utilize the Kaggle API to programmatically download the dataset within the Jupyter Notebook environment. By setting Kaggle credentials in an `.env` file and loading them securely, the notebook runs a `kaggle datasets download` command to fetch and unzip the dataset directly from the external endpoint.
- **Outcome:** This approach ensures the project dynamically retrieves data from an external source, reflecting a professional standard of external data acquisition and integration.

**Key Features Include:**

- **1stFlrSF / 2ndFlrSF:** Floor areas above ground.
- **OverallQual / OverallCond:** Ratings of property quality and condition.
- **GrLivArea / TotalBsmtSF:** Living area and basement size.
- **GarageFinish / GarageArea:** Garage attributes influencing convenience and property value.
- **YearBuilt / YearRemodAdd:** Temporal attributes capturing historical construction and renovation efforts.
- **SalePrice:** Target variable representing the actual sale price of the property.

**Comprehensive List of Features and their Units**

| Variable          | Meaning                                                   | Units                                                                                                                                                                                                                                                                     |
|-------------------|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1stFlrSF**      | First Floor square feet                                   | 334 - 4,692                                                                                                                                                                                                                                                              |
| **2ndFlrSF**      | Second-floor square feet                                  | 0 - 2,065                                                                                                                                                                                                                                                                |
| **BedroomAbvGr**  | Bedrooms above grade (does NOT include basement bedrooms) | 0 - 8                                                                                                                                                                                                                                                                     |
| **BsmtExposure**  | Refers to walkout or garden level walls                   | Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement                                                                                                                                                                         |
| **BsmtFinType1**  | Rating of basement finished area                          | GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinished; None: No Basement                                                                                                  |
| **BsmtFinSF1**    | Type 1 finished square feet                               | 0 - 5,644                                                                                                                                                                                                                                                                 |
| **BsmtUnfSF**     | Unfinished square feet of basement area                   | 0 - 2,336                                                                                                                                                                                                                                                                 |
| **TotalBsmtSF**   | Total square feet of basement area                        | 0 - 6,110                                                                                                                                                                                                                                                                 |
| **GarageArea**    | Size of garage in square feet                             | 0 - 1,418                                                                                                                                                                                                                                                                 |
| **GarageFinish**  | Interior finish of the garage                             | Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage                                                                                                                                                                                                      |
| **GarageYrBlt**   | Year garage was built                                     | 1900 - 2010                                                                                                                                                                                                                                                               |
| **GrLivArea**     | Above grade (ground) living area square feet              | 334 - 5,642                                                                                                                                                                                                                                                               |
| **KitchenQual**   | Kitchen quality                                           | Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor                                                                                                                                                                                                          |
| **LotArea**       | Lot size in square feet                                   | 1,300 - 215,245                                                                                                                                                                                                                                                           |
| **LotFrontage**   | Linear feet of street connected to property               | 21 - 313                                                                                                                                                                                                                                                                  |
| **MasVnrArea**    | Masonry veneer area in square feet                        | 0 - 1,600                                                                                                                                                                                                                                                                 |
| **EnclosedPorch** | Enclosed porch area in square feet                        | 0 - 286                                                                                                                                                                                                                                                                   |
| **OpenPorchSF**   | Open porch area in square feet                            | 0 - 547                                                                                                                                                                                                                                                                   |
| **OverallCond**   | Rates the overall condition of the house                  | 10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor                                                                                                                                   |
| **OverallQual**   | Rates the overall material and finish of the house        | 10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor                                                                                                                                   |
| **WoodDeckSF**    | Wood deck area in square feet                             | 0 - 736                                                                                                                                                                                                                                                                   |
| **YearBuilt**     | Original construction date                                | 1872 - 2010                                                                                                                                                                                                                                                               |
| **YearRemodAdd**  | Remodel date (same as construction date if no remodeling or additions) | 1950 - 2010                                                                                                                                                                                                                                                    |
| **SalePrice**     | Sale Price                                                | \$34,900 - \$755,000                                                                                                                                                                                                                                                      |

---

## Business Requirements

**Context:** Lydia Doe inherited multiple houses in Ames, Iowa, and wants to accurately estimate their sale prices. Having reliable estimates and understanding which features matter most can guide renovation decisions, pricing strategies, and investment planning.

**Primary Business Requirements:**

1. **Discover Correlations:**
   - Understand how different property attributes correlate with sale price.
   - Visualize these correlations to help Lydia grasp which features significantly drive value in Ames, potentially differing from her home state’s market.

2. **Predict House Sale Prices:**
   - Use the historical dataset to build regression models that accurately predict sale prices.
   - Provide an interface to predict prices for Lydia’s inherited houses and any other properties by inputting relevant features.
   - Enable Lydia to quickly adapt her selling strategy based on data-driven price insights.

**User Stories:**

- *As a homeowner (Lydia):* I want to know which factors influence my property’s value and get accurate price estimates to decide on renovations and pricing strategies.
- *As a real estate agent:* I need quick, reliable price estimates and insights to advise clients and justify listing prices.
- *As an investor:* I want to identify undervalued properties and potential returns on investment by understanding feature importance and predicted prices.

---

## Hypotheses and Validation Plans

The project started with several hypotheses connecting property attributes to sale price:

1. **Hypothesis 1:** Higher overall quality (`OverallQual`) leads to higher sale prices.
   - *Validation:* Correlation analysis, scatter plots, and box plots to confirm a strong positive relationship.

2. **Hypothesis 2:** Larger living areas (`GrLivArea`) increase sale prices.
   - *Validation:* Correlation coefficients and scatter plots of `GrLivArea` vs. `SalePrice`.

3. **Hypothesis 3:** Recently remodeled homes (`YearRemodAdd`) command higher prices.
   - *Validation:* Line plots and temporal analysis to see if modern renovations align with price increases.

Validating these hypotheses helps identify key value drivers and gives Lydia confidence in the underlying logic of the model’s insights.

---

## Mapping Business Requirements to Tasks

### Alignment with CRISP-DM

This project follows the CRISP-DM methodology for a structured, professional approach:

1. **Business Understanding:** Clarify objectives and success metrics (accurate predictions, correlation insights).
2. **Data Understanding:** Conduct EDA to comprehend distributions, outliers, and relationships.
3. **Data Preparation:** Clean, encode, and transform data to ensure readiness for modeling.
4. **Modeling:** Train multiple regression models (Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting).
5. **Evaluation:** Use MAE, RMSE, and R² metrics to select the best model. Aim for R² ≥ 0.75.
6. **Deployment:** Provide an interactive Streamlit dashboard for immediate user access and decision support.

### Business Requirement 1: Discover Correlations

**Data Visualization Tasks:**
- Correlation matrices, heatmaps, scatter plots, and box plots to reveal feature-price relationships.
- Helps Lydia understand Ames-specific market drivers.

### Business Requirement 2: Predict House Sale Prices

**Machine Learning Tasks:**
- Train and evaluate regression models.
- Deploy the top model for real-time predictions.
- Enable Lydia and other users to input custom property attributes and get instant predicted sale prices.

---

## ML Business Case

**Problem Statement:**  
Lydia needs accurate, data-driven price estimates for her inherited properties to maximize profits and avoid guesswork.

**Proposed Solution:**  
Use a regression-based ML model trained on Ames historical data to predict sale prices from property attributes.

**Expected Benefits:**
- **Accurate Pricing:** Avoid mispricing and financial losses.
- **Informed Decision-Making:** Understand which renovations yield higher returns.
- **Scalability:** Apply insights to any Ames property, guiding future investments.

**Performance Goal:**  
Achieve an **R² ≥ 0.75** to ensure the model is reliable and explains a significant portion of the price variance.

---

## Dashboard Design

Built with Streamlit, the dashboard provides intuitive navigation and interactive elements:

**Pages:**

1. **Project Summary:**
   - Overview of objectives, CRISP-DM alignment, and business context.

2. **Feature Correlations:**
   - Visualizations (heatmaps, scatter plots) to identify which attributes strongly influence sale price.

3. **House Price Predictions:**
   - Predict sale prices for Lydia’s inherited houses.
   - Input forms (sliders, number inputs, dropdowns) to estimate any Ames property’s value on the fly.

4. **Project Hypotheses:**
   - Displays initial hypotheses and their validation.
   - Confirms which features matter most and justifies the predictive model’s logic.

5. **Model Performance:**
   - Showcases evaluation metrics, residual analysis, and feature importance.
   - Builds trust in the model’s accuracy and highlights its limitations.

**User Experience:**
- Real-time updates.
- Clear navigation.
- Immediate actionable insights for stakeholders.

---

## Unfixed Bugs

No known unfixed bugs. Thorough testing ensures stable performance and reliable functionality.

**Fixed Compatability issue:**
Heroku Stack 24 was unable to download the XGBoost package that I wanted to use due to size cnsraints, as a workaround I have provided a streamlit.io version which can be used to view the XGBoost results which were the best performing of all, however this version of the app is not as complete as the heroku deployment, The functionalities that required this package were removed from the final app due to the compatibility issue and now it is working with no observed bugs.

---

## Deployment

**Heroku Live Link:**  
[House Price Prediction Dashboard](https://john-furlong-price-predictor-ee67ab0394fa.herokuapp.com/)

**Note:** Remove Heroku-specific files (e.g., `Procfile`) when deploying to Streamlit, where the app and XGBoost run smoothly.

**Steps:**
1. Prepare GitHub repo with models, code, and data.
2. Add `Procfile` and `runtime.txt` for Heroku if using Heroku.
3. Deploy on Heroku or run on Streamlit.

---

## Future Enhancements

- **Additional Features:** Integrate external data (location-based amenities, economic indicators).
- **Model Tuning:** Explore ensembles or neural networks for even better accuracy.
- **Continuous Monitoring:** Retrain models as market conditions evolve.
- **Feedback Loops:** Incorporate user feedback to refine usability and relevance.

---

This README presents a comprehensive overview of the project’s aims, methodology, and value. By combining CRISP-DM principles, data-driven insights, and robust ML models, the House Price Prediction Dashboard exemplifies a professional, real-world application of Data Science to support informed decision-making in the housing market.

### Acknowledgments & Resources
This project was developed by John Furlong with support from the Code Institute. Their comprehensive curriculum, supportive resources, and practical assignments provided the foundational knowledge required to build this solution. The Ames Housing Dataset served as an excellent real-world data source, supplying rich feature information. The notebook obtains the data via the Kaggle endpoint, providing a rich, real-world testing ground.

Additionally, ChatGPT by OpenAI aided in conceptual clarification, technique refinement, and problem-solving. This synergy combining structured learning with AI driven guidance resulted in a robust, reliable, and user-focused tool.