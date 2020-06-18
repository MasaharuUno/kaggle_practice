# Import libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Load the data
data = pd.read_csv("train.csv", index_col="Id")

# Drop the missing target
data.dropna(axis=0, subset=["SalePrice"], inplace=True)

# Separate target from predictors
y = data.SalePrice
X_full = data.drop(["SalePrice"], axis=1)

# Break off vaidation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, 
    y, 
    train_size=0.8, 
    random_state=0
)

# Drop columns with missing values
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True) 

# Select categorical columns with relatively low cardinality
low_cardinality_cols = [
    col for col in X_train_full.columns 
    if X_train_full[col].dtype == "object" 
    and X_train_full[col].nunique() < 10
    ]

# Select numerical columns
numerical_cols = [
    col for col in X_train_full.columns 
    if X_train_full[col].dtype in ["int64", "float64"]
    ]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()

# Define function to  measure quality of each approach
def score_dataset(X_train, X_valid, y_train, y_valid):
    """
    Train RandomForestRegressor model using training dataset.
    Calculate MAE (mean absolute error) using validation dataset.
    Return MAE score.

    Parameters: 
    X_train -- dataframe: predictors for training
    X_valid -- dataframe: predictors for validation
    y_train -- series: target for training
    y_valid -- series: target for validation

    Return: 
    score -- float: MAE score
    """
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    score = mean_absolute_error(y_valid, preds)
    return score

### Approach 1: drop categorical variabls
# drop categorical columns
drop_X_train = X_train.select_dtypes(exclude=["object"])
drop_X_valid = X_valid.select_dtypes(exclude=["object"])

# Evaluate the approach 1
print("MAE (drop categorical variables):")
score_1 = score_dataset(drop_X_train, drop_X_valid, y_train, y_valid)
print(f"{score_1:.0f}")

### Approach 2: label encoding
# Make copy to avoid shanging original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply label encoder to each column with categorical data
s = (X_train.dtypes == "object")
object_cols = list(s[s].index)

label_enc = LabelEncoder()
for col in object_cols:
    label_enc.fit(X_full[col])
    label_X_train[col] = label_enc.transform(X_train[col])
    label_X_valid[col] = label_enc.transform(X_valid[col])

# Evaluate the approach 2
print("MAE (label encoding):")
score_2 = score_dataset(label_X_train, label_X_valid, y_train, y_valid)
print(f"{score_2:.0f}")

### Approach 3: one-hot encoding
# Apply one-hot encoder to each column with categorical data
OH_enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
OH_cols_train = pd.DataFrame(OH_enc.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_enc.transform(X_valid[object_cols]))

# One-hot encoding remove index; put them back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns, which will replaced with one-hot encoding
num_X_train = X_train[numerical_cols]
num_X_valid = X_valid[numerical_cols]

# Add one-hot encoding to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Evaluate the approach 3
print("MAE (one-hot encoding):")
score_3 = score_dataset(OH_X_train, OH_X_valid, y_train, y_valid)
print(f"{score_3:.0f}")