# Import libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
# Read the data
X_full = pd.read_csv("train.csv", index_col="Id")
X_test_full = pd.read_csv("test.csv", index_col="Id")

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)
y = X_full.SalePrice
X_full.drop(["SalePrice"], axis=1, inplace=True)

# Use only numerical predictors
X = X_full.select_dtypes(exclude=["object"])
X_test = X_test_full.select_dtypes(exclude=["object"])

# Break off validation from training data
X_train, X_valid, y_train, y_valid = train_test_split(
    X, 
    y, 
    train_size=0.8, 
    random_state=0
)

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    """
    Train RandomForestRegressor model using training dataset.
    Calculate MAE (mean absolute error) using validation dataset.
    Return MAE score

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

# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

# Evaluate an approach (drop columns with missing values)
print("MAE (drop columns with missing values):")
score_1 = score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid)
print(f"{score_1:.0f}")

# Impute
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removes column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Evaluate an approach (imputarion)
print("MAE (imputation):")
score_2 = score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid)
print(f"{score_2:.0f}")

# Make new columns indicating what will be imputed
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
for col in cols_with_missing:
    X_train_plus[col + "_is_missing"] = X_train_plus[col].isnull()
    X_valid_plus[col + "_is_missing"] = X_valid_plus[col].isnull()

# Impute
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removes column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

# Evaluate an approach (an extension to imputation)
print("MAE (an extension to imputation):")
score_3 = score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid)
print(f"{score_3:.0f}")

# Train the model
my_model = RandomForestRegressor(n_estimators=100, random_state=0)

my_imputer = SimpleImputer()
X_plus = X.copy()
X_test_plus = X_test.copy()
for col in cols_with_missing:
    X_plus[col + "_is_missing"] = X_plus[col].isnull()
    X_test_plus[col + "_is_missing"] = X_test_plus[col].isnull()

imputed_X_plus = pd.DataFrame(my_imputer.fit_transform(X_plus))
imputed_X_test_plus = pd.DataFrame(my_imputer.transform(X_test_plus)) 

imputed_X_plus.columns = X_plus.columns
imputed_X_test_plus.columns = X_test_plus.columns

my_model.fit(imputed_X_plus, y)

# Prediction
predictions = my_model.predict(imputed_X_test_plus)

# save test predictions to file
output = pd.DataFrame({
    "Id": imputed_X_test_plus.index, 
    "SalePrice": predictions
    })

output.to_csv("submission_20200616_1.csv")