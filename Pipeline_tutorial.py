# Import libraries
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the data 
data = pd.read_csv("melb_data.csv")

# Separate target ("Price") from predictors
y = data.Price 
X = data.drop(["Price"], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X, 
    y, 
    train_size=0.8, 
    random_state=0
)

# Select numerical columns
numerical_cols = [
    col for col in X_train_full.columns 
    if X_train_full[col].dtype in ["int64", "float64"]
]

# Select categoical columns with relatively low caridinality (10)
low_cardinality_cols = [
    col for col in X_train_full.columns 
    if X_train_full[col].dtype == "object" 
    and X_train_full[col].nunique() < 10
]

# Keep selected columns only
my_cols = numerical_cols + low_cardinality_cols
X_train = X_train_full[my_cols]
X_valid = X_valid_full[my_cols]

"""
Define preprocessing Steps
"""
# Preprocessing for numerica data
numerical_transformer = SimpleImputer(strategy="constant")

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")), 
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ("num", numerical_transformer, numerical_cols), 
    ("cat", categorical_transformer, low_cardinality_cols)
])

# Define the model 
model = RandomForestRegressor(n_estimators=100, random_state=0)

# Bundle preprocessing and modeling code in a pipeline
my_pipelin = Pipeline(steps=[
    ("preprocessor", preprocessor), 
    ("model", model)
])

"""
Evaluate the pipeline
"""
# Preprocess training data, and fit model
my_pipelin.fit(X_train, y_train)

# Preprocess validation data, and get predictions
preds = my_pipelin.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print("MAE:", score)