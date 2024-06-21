# Importing required libraries
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Initialize a ticker class to download price history for TSMC
tsmc = yf.Ticker("TSM")

# Get historical market data for the maximum available period
tsmc = tsmc.history(period="max")

# Removing the dividends and stock splits columns
tsmc.drop(columns=["Dividends", "Stock Splits"], inplace=True)

# Create a new column for the next day's closing price
tsmc["Tomorrow"] = tsmc["Close"].shift(-1)

# Return 1 if the tomorrow price > today's close price (a price increase), otherwise 0
tsmc["Target"] = (tsmc["Tomorrow"] > tsmc["Close"]).astype(int)

# Define the predictors (features) for the model
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Split the data into training and testing sets
train = tsmc.iloc[:-100]
test = tsmc.iloc[-100:]

# Fit the model
model.fit(train[predictors], train["Target"])

# Make predictions on the test set
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)

# Plot predictions with actual values
combined = pd.concat([test["Target"], preds.rename("Predictions")], axis=1)
combined.plot()

# Define a function to make predictions
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Define a function to backtest the model
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# Perform backtesting
predictions = backtest(tsmc, model, predictors)

# Calculate and print precision score
precision = precision_score(predictions["Target"], predictions["Predictions"])
print(f"Precision score: {precision}")

# Calculate the mean closing price and other metrics
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = tsmc.rolling(horizon).mean()

    # Column calculating the ratio of today's closing price to the rolling average closing price
    ratio_column = f"Close_Ratio_{horizon}"
    tsmc[ratio_column] = tsmc["Close"] / rolling_averages["Close"]

    # Column counting the number of days the stock price increased over the past x days
    trend_column = f"Trend_{horizon}"
    tsmc[trend_column] = tsmc["Target"].shift(1).rolling(horizon).sum()

    new_predictors += [ratio_column, trend_column]

# Removing rows with missing data
tsmc.dropna(inplace=True)

# Re-initialize the RandomForestClassifier with different parameters
model = RandomForestClassifier(n_estimators=1000, min_samples_split=50, random_state=1)

# Modify the predict function to return probabilities
def predict_with_proba(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = (preds >= 0.6).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Perform backtesting with the new model
predictions = backtest(tsmc, model, new_predictors)

# Print the value counts of predictions
print(predictions["Predictions"].value_counts())

# Calculate and print precision score for the new model
precision = precision_score(predictions["Target"], predictions["Predictions"])
print(f"Precision score: {precision}")
