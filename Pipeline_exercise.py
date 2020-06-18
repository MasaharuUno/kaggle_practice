# import libraries
import pandas as pd 
from datetime import datetime
from sklearn.model_selection import train_test_split 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data
X_full = pd.read_csv("train.csv", index_col="Id")
X_test_full = pd.read_csv("test.csv", index_col="Id")

# Remove rows with missing value
X_full.dropna(axis=0, subset=["SalePrice"], inplace=True)

# Separate target from predictors
y = X_full.SalePrice 
X_full.drop(["SalePrice"], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X_full, 
    y, 
    train_size=0.8, 
    random_state=0
)

# Select categorical columns with relatively low cardinality
low_cardinality_cols = [
    col for col in X_train_full.columns 
    if X_train_full[col].dtype == "object" 
    and X_train_full[col].nunique() < 10
]

# Select numercal columns
numerical_cols = [
    col for col in X_train_full.columns 
    if X_train_full[col].dtype in ["int64", "float64"]
]

# Keep selected columns only 
my_cols = numerical_cols + low_cardinality_cols
X_train = X_train_full[my_cols]
X_valid = X_valid_full[my_cols]
X_test = X_test_full[my_cols]

"""
Define preprocessing steps, then pipeline
"""
# Preprocessing for numerical data
numerical_transform = SimpleImputer(strategy="constant")

# Preprocessing for categorical data
categorical_transform = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transform, numerical_cols), 
    ("cat", categorical_transform, low_cardinality_cols)
])

# define the model 
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle Preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("model", model)
])

"""
Evaluate the pipeline
"""
# Preprocess training data, and  fit the model
my_pipeline.fit(X_train, y_train)

# Preprocess vaidation data, and get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the pipeline
score = mean_absolute_error(y_valid, preds)
print("MAE:", score)


"""
Impove the performance
"""
def get_score(X_train, X_valid, y_train, y_valid,num_strategy="mean", cat_strategy="most_frequent", n_estimators=100):
    """
    Train the RandomForestRegressor model using training dataset.
    Evalate the trained model using validation dataset.
    Returns MAE (mean absolute error) and trained model.

    Parameters: 
    num_strategy -- str: strategy of SimpleImputer for numerical columns
    cat_strategy -- str: strategy of SimpleImputer for categorical columns
    n_estimators -- int: n_estimators of RandomForestRegressor
    X_train -- dataframe: predictors of training data
    X_valid -- dataframe: predictors of validation data
    y_train -- series: target of training data
    y_valid -- series: target of validation data
    """
    # preprocessing for numerical values
    numerical_transformer = SimpleImputer(strategy=num_strategy)

    # preprocessing for categorical values
    categorical_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy=cat_strategy)),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Bundle preprocessing for numerical an categorica values
    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, low_cardinality_cols)
    ])

    # define the model 
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=0)

    # Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor), 
        ("model", model)
    ])

    # Preprocess training data and fit model
    my_pipeline.fit(X_train, y_train)
    
    # Preprocess validation data and get predictions
    preds = my_pipeline.predict(X_valid)
    
    # Evaluate the pipeline
    score = mean_absolute_error(y_valid, preds)

    return score, my_pipeline


scores = []
score_dict = {}
for num_strategy in ["mean", "median", "most_frequent", "constant"]:
    for cat_strategy in ["most_frequent", "constant"]:
        for n_estimators in [5, 10, 50, 100, 500]:
            score, _ = get_score(X_train, X_valid, y_train, y_valid, num_strategy, cat_strategy, n_estimators)
            scores.append(score)   
            score_dict[f"num_strategy: {num_strategy}, cat_strategy: {cat_strategy}, n_estimators: {n_estimators}"] = score


# print(score_dict)
keys = [k for k in score_dict.keys() if score_dict[k] == min(scores)]
print(keys)

_, best_model = get_score(X_train, X_valid, y_train, y_valid, "median", "constant", 500)
predictions = best_model.predict(X_test)
output = pd.DataFrame(
    {"Id": X_test.index, "SalePrice": predictions}
)
now = datetime.now()
path = now.strftime("%Y%m%d_%H%M")
output.to_csv(f"submission_{path}.csv")