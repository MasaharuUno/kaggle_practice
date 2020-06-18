import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Read the data
X_full = pd.read_csv("train.csv", index_col="Id")
X_test_full = pd.read_csv("test.csv", index_col="Id")

# Obtain target and predictors
y = X_full.SalePrice
features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data 
X_train, X_valid, y_train, y_valid = train_test_split(
    X, 
    y, 
    train_size=0.8, 
    random_state=0
)

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion="mae", random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=50, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]

# Function for comparison
def score_model(model, X_train, X_valid, y_train, y_valid,):
    """
    Train the model and calculate MAE (mean absolute error).
    Return MAE

    Parameters:
    model -- model: model for training
    X_train -- dataframe: predictors for training
    X_valid -- dataframe: predictors for validation
    y_train -- series: target for training
    y_valid -- series: target for validation

    Return:
    mae_score -- float: MAE score
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae_score = mean_absolute_error(y_valid, preds)
    return mae_score

# Compare 5 models
scores = []
for model in models:
    mae = score_model(model, X_train, X_valid, y_train, y_valid)
    scores.append(mae)
    # print(f"Model: {models.index(model)+1}: {mae:.0f}")

# Best model
best_model = models[scores.index(min(scores))]

# Fit the model to the training data
best_model.fit(X, y)

# Generate test prediction
preds_test = best_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({
    "Id": X_test.index,
    "SalePrice": preds_test})

output.to_csv("submission_20200616.csv", index=False)