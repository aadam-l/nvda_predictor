# Yahoo Finance API
import yfinance as yf

# Initialize a ticker class to download price history for a specified symbol
tsmc = yf.Ticker("TSM")

# Get historical market data
tsmc = tsmc.history(period="max")
print(tsmc)

# Display the DataFrame head to inspect the initial rows
print(tsmc.head())

# Print the index of the DataFrame
print(tsmc.index)

# Plot data using the index and closing price
tsmc.plot.line(y="Close", use_index=True)

# Remove the dividends and stock splits columns
tsmc.drop(columns=["Dividends", "Stock Splits"], inplace=True)

# Add a column for tomorrow's price using the Pandas shift() function
tsmc["Tomorrow"] = tsmc["Close"].shift(-1)

# Display the DataFrame head to inspect the changes
print(tsmc.head())

# Create a target column: 1 if the tomorrow's price > today's close price (price increase), else 0
tsmc["Target"] = (tsmc["Tomorrow"] > tsmc["Close"]).astype(int)

# Display the DataFrame head to inspect the changes
print(tsmc.head())

# Utilize a RandomForestClassifier, which trains individual decision trees with
# randomized parameters and averages their results. This method offers greater resistance
# to overfitting compared to other data models.
from sklearn.ensemble import RandomForestClassifier

# Define the model with specific parameters
model = RandomForestClassifier(n_estimators=1000, min_samples_split=100, random_state=1)

# Split the data into training and testing sets
train = tsmc.iloc[:-100]
test = tsmc.iloc[-100:]

# Define the predictors
predictors = ["Close", "Volume", "Open", "High", "Low"]

# Train the model
model.fit(train[predictors], train["Target"])

# Print the model details
print(model)
