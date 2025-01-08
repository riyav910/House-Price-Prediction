# House Price Prediction

This project predicts house prices based on various features using the California Housing dataset. The goal is to build a predictive model that estimates the median house value for different California districts. The model uses the XGBoost regression algorithm to make predictions based on features such as median income, house age, number of rooms, population, and location.

## Project Overview

The project utilizes the California Housing dataset, which contains information about various attributes related to housing. The objective is to train a machine learning model to predict the median house value for different areas in California based on the given features.

### Dataset
The dataset consists of the following attributes:
- **MedInc**: Median income in block group
- **HouseAge**: Median house age in block group
- **AveRooms**: Average number of rooms per household
- **AveBedrms**: Average number of bedrooms per household
- **Population**: Population of the block group
- **AveOccup**: Average number of household members
- **Latitude**: Latitude of the block group
- **Longitude**: Longitude of the block group
- **Price (Target)**: Median house value in California district (in hundreds of thousands of dollars)

### Steps Involved

1. **Data Import and Exploration**:
   - Load the California Housing dataset using `sklearn.datasets.fetch_california_housing`.
   - Convert the dataset into a pandas DataFrame.
   - Analyze the dataset to check for missing values and basic statistics.

2. **Data Visualization**:
   - Use correlation heatmaps and histograms to understand the relationships between various features.
   - Visualize the distribution of house prices and other attributes.

3. **Modeling**:
   - Split the data into features (`X`) and target (`Y`).
   - Further split the data into training and testing datasets.
   - Train an XGBoost regressor model on the training dataset.

4. **Prediction and Evaluation**:
   - Make predictions using the trained model on both the training and testing datasets.
   - Evaluate the model's performance using graphical comparison and accuracy metrics.

### Libraries Used
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib** and **Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning utilities, including splitting the dataset and metrics.
- **XGBoost**: For implementing the regression model.

### Requirements

Make sure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```
### How to Run the Code

Clone this repository:
```bash
git clone https://github.com/riyav910/house-price-prediction.git
```
Navigate to the project folder:
```bash
cd house-price-prediction
```
Run the script:
```bash
python house_price_prediction.py
```
Check the output for predictions on the testing dataset.
