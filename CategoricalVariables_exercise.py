# Import libraries 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load the data
X = pd.read_csv("train.csv", index_col="Id")
X_test = pd.read_csv("test.csv", index_col="Id")

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = X.SalePrice
X.drop(["SalePrice"], axis=1, inplace=True)

# Drop columns with missing values
cols_with_missing = [
    col for col in X.columns 
    if X[col].isnull().any()
]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, 
    y, 
    train_size=0.8, 
    random_state=0
)

# Approach 1
# Define function for comparing differnt approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    """
    Train RandomForestRegressor model using training dataset.
    Calculate MAE (mean absolute error) score using validation dataset.

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

# Drop columns in training and validation data
drop_X_train = X_train.select_dtypes(exclude=["object"])
drop_X_valid = X_valid.select_dtypes(exclude=["object"])

# Evaluate an approach 1
print("MAE (drop columns with categorical data):")
score_1 = score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
print(f"{score_1:.0f}")

# Approach 2
# All categorical columns
object_cols = [
    col for col in X.columns 
    if X[col].dtype =="object"
]

# Columns that can be safely label encoded
good_label_cols = [
    col for col in object_cols
    if set(X_train[col]) == set(X_valid[col])
]

# Problematiccolumns that will be dropped from the dataset 
bad_label_cols = set(object_cols) - set(good_label_cols)

# Drop categorical columns that will not be encoded
label_X_train = X_train.drop(bad_label_cols, axis=1)
label_X_valid = X_valid.drop(bad_label_cols, axis=1)

# Apply label encoder
label_enc = LabelEncoder()
for col in good_label_cols:
    label_X_train[col] = label_enc.fit_transform(label_X_train[col])
    label_X_valid[col] = label_enc.transform(label_X_valid[col])

# Evaluate an approach 2
print("MAE (label encoding):")
score_2 = score_dataset(label_X_train, label_X_valid, y_train, y_valid)
print(f"{score_2:.0f}")

# Approach 3
# columns that will be one-hot encoded
low_cardinality_cols = [
    col for col in X.columns 
    if X[col].dtype == "object" 
    and X[col].nunique() < 10
]

# Columns that will be dropped for the dataset
high_cardinality_cols = list(set(object_cols) - set(low_cardinality_cols))

# Apply one-hot encoder to each column with categorical data
OH_enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
OH_cols_train = pd.DataFrame(OH_enc.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_enc.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removes index; put them back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical columns
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Evaluate an approach 3
print("MAE (one-hot encoding):")
score_3 = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)
print(f"{score_3:.0f}")