import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Load the data into a variable data
data = pd.read_csv("train.csv")

# Select target
y = data.SalePrice

# Use only numeric predictors
predictors = data.drop(["SalePrice"], axis=1)
X = predictors.select_dtypes(exclude=["object"])

# Divide data into training and validation subset
X_train, X_valid, y_train, y_valid = train_test_split(
    X, 
    y, 
    train_size=0.8, 
    random_state=0
)

# Define functions to measure quality of each approach
def score_dataset(X_train, X_valid, y_train, y_valid):
    """
    Train RandomForestRegressor using training dataset.
    Predict target using validation dataset and evaluate the model.
    Return MAE (mean absolute error) score.

    Parameters: 
    X_train -- dataframe: predictors for training
    X_valid -- dataframe: predictors for validation
    y_train -- series: target for training
    y_valid -- series: target for validation

    Return:
    score -- float: MAE score
    """
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    score = mean_absolute_error(y_valid, preds)
    return score

### Approach 1: drop columns with missing values
# Get names of columns with missing value
cols_with_missing = [
    col for col in X.columns 
    if X[col].isnull().any()
]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Measure quality of the approach 1
print("MAE (drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

### Approach 2: imputation
# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removes column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Measure quality of the approach 2
print("MAE (imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

### Approach 3: an extension to imputation
# Make copies to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be inputed
for col in cols_with_missing:
    X_train_plus[col + "_was_missing"] = X_train_plus[col].isnull()
    X_valid_plus[col + "_was_missing"] = X_valid_plus[col].isnull()

# Imputed
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removes columns names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

# Measure quality of the approach 3
print("MAE (an extension to imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))