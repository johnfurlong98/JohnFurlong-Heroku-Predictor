# House Price Prediction Dashboard

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1,500 rows and represents housing records from Ames, Iowa. It includes house profiles (e.g., Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and their respective sale prices for houses built between 1872 and 2010.

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

## Business Requirements

As a good friend, you are requested by your friend, **Lydia Doe**, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to help in maximizing the sales price for the inherited properties.

Although Lydia has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and has provided you with that.

**Business Requirements:**

1. **Discover Correlations:**
   - The client is interested in discovering how the house attributes correlate with the sale price.
   - She expects data visualizations of the correlated variables against the sale price to illustrate these relationships.

2. **Predict House Sale Prices:**
   - The client is interested in predicting the house sale price for her four inherited houses.
   - She also wants the ability to predict the sale price for any other house in Ames, Iowa.

## Hypothesis and How to Validate

### Hypothesis 1

**Higher overall quality of the house (`OverallQual`) leads to a higher sale price.**

**Validation Plan:**

- Calculate the correlation coefficient between `OverallQual` and `SalePrice`.
- Create scatter plots and box plots to visualize the relationship.
- Analyze the strength and significance of the correlation.

### Hypothesis 2

**Larger living areas (`GrLivArea`) result in higher sale prices.**

**Validation Plan:**

- Compute the correlation between `GrLivArea` and `SalePrice`.
- Plot `GrLivArea` against `SalePrice` using scatter plots.
- Examine any outliers or anomalies that may affect the relationship.

### Hypothesis 3

**Houses that have been recently remodeled (`YearRemodAdd`) have higher sale prices.**

**Validation Plan:**

- Assess the correlation between `YearRemodAdd` and `SalePrice`.
- Use line plots to visualize average sale prices over remodel years.
- Determine if newer remodel years correspond to higher prices.

## The Rationale to Map the Business Requirements to the Data Visualizations and ML Tasks

### Business Requirement 1

**Data Visualization Tasks:**

- Perform exploratory data analysis (EDA) to identify key features that correlate with `SalePrice`.
- Use correlation matrices and heatmaps to present the strength of relationships.
- Generate scatter plots, box plots, and line plots to visually explore the correlations.
- These visualizations will help Lydia understand which attributes most significantly affect house prices in Ames, Iowa.

### Business Requirement 2

**Machine Learning Tasks:**

- Develop regression models to predict `SalePrice` based on house attributes.
- Use the historical dataset to train models such as Linear Regression, Random Forest, and XGBoost.
- Evaluate models using performance metrics (e.g., MAE, RMSE, R² Score) to select the best-performing model.
- Apply the model to predict sale prices for Lydia's four inherited houses.
- Provide a user interface (dashboard) where Lydia can input attributes of any house in Ames, Iowa, and get a predicted sale price.

## ML Business Case

### Problem Statement

Lydia needs an accurate and reliable method to estimate the sale prices of her inherited houses in Ames, Iowa, to maximize her profits and make informed decisions for future property investments.

### Proposed Solution

Build a predictive regression model using historical house sale data from Ames, Iowa, to estimate the sale prices based on various house attributes.

### Expected Benefits

- **Accurate Pricing:** Helps Lydia avoid underpricing or overpricing her properties.
- **Strategic Decision-Making:** Understanding key factors affecting house prices assists in making informed decisions.
- **Future Investments:** The predictive model can be used for any house in Ames, Iowa, aiding future investment considerations.

### Performance Goal

Achieve an **R² score of at least 0.75** on both the training and test sets to ensure the model's reliability.

### Model Inputs and Outputs

- **Inputs:** House attributes such as `OverallQual`, `GrLivArea`, `YearBuilt`, `TotalBsmtSF`, etc.
- **Output:** Predicted sale price of the house.

## Dashboard Design

The dashboard will be built using **Streamlit** and will include the following pages:

### Project Summary Page

**Content:**

- Introduction to the project and its objectives.
- Overview of the datasets used.
- Summary of Lydia's business requirements.

**Widgets/Elements:**

- Text blocks with project description.
- Images or icons representing key aspects.

### Feature Correlations Page

**Content:**

- Visualizations showing how house attributes correlate with sale price.
- Insights from the data analysis.

**Widgets/Elements:**

- Heatmaps of correlation matrices.
- Scatter plots of top correlated features.
- Interactive elements to select features for visualization.

### House Price Predictions Page

**Content:**

- Display of predicted sale prices for the four inherited houses.
- Total predicted sale price for all inherited properties.
- Real-time prediction tool for any house in Ames, Iowa.

**Widgets/Elements:**

- Tables showing predictions for inherited houses.
- Input forms with sliders, number inputs, and dropdowns for user inputs.
- Button to trigger prediction.
- Display of predicted sale price based on user input.

### Project Hypotheses Page

**Content:**

- Listing of project hypotheses.
- Explanation of validation methods and results.
- Visualizations supporting the hypotheses.

**Widgets/Elements:**

- Text blocks explaining hypotheses and findings.
- Plots and charts demonstrating validation.

### Model Performance Page

**Content:**

- Presentation of model evaluation metrics.
- Comparison of different models tested.
- Explanation of the machine learning pipeline.

**Widgets/Elements:**

- Tables showing performance metrics (MAE, RMSE, R² Score).
- Graphs of actual vs. predicted prices.
- Residual plots.
- Text blocks outlining pipeline steps.

## Unfixed Bugs

At the time of deployment, there are **no known unfixed bugs**. All features and functionalities have been thoroughly tested and are working as expected.

## Deployment

### Heroku

*The app live link is: [House Price Prediction Dashboard](https://your-app-name.herokuapp.com/)*

**Deployment Steps:**

1. **Set Up Repository:**

   - Ensure that all code, including `app.py`, models, data files, and `requirements.txt`, is pushed to a GitHub repository.
   - Include a `Procfile` and `runtime.txt` for Heroku deployment.

2. **Create a Heroku Account:**

   - Go to [Heroku](https://www.heroku.com/) and sign up for an account.

3. **Install Heroku CLI (if deploying via command line):**

   - Download and install the Heroku CLI from the [official website](https://devcenter.heroku.com/articles/heroku-cli).

4. **Log In to Heroku via CLI:**

   ```bash
   heroku login
