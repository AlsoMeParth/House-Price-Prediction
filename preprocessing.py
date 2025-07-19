import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
# from sklearn.preprocessing import OrdinalEncoder  # Uncomment if you prefer ordinal
 
# 1. Load the data
housing = pd.read_csv("housing.csv")
 
# 2. Create a stratified test set based on income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)
 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)
 
# Work on a copy of training data
housing = strat_train_set.copy()
 
# 3. Separate predictors and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)
 
# 4. Separate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]
 
# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
 
# Categorical pipeline
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
 
# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])
 
# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
 
# housing_prepared is now a NumPy array ready for training

# 7. Train models
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
 
# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
 
# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
 
# 8. Predict using training data
lin_preds = lin_reg.predict(housing_prepared)
tree_preds = tree_reg.predict(housing_prepared)
forest_preds = forest_reg.predict(housing_prepared)
 
# Evaluation of models 
# Calculate RMSE
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
tree_rmse =root_mean_squared_error(housing_labels, tree_preds)
forest_rmse = root_mean_squared_error(housing_labels, forest_preds)
 
print("Linear Regression RMSE:", lin_rmse)
print("Decision Tree RMSE:", tree_rmse)   
print("Random Forest RMSE:", forest_rmse)

# Here Decision Tree shows a RMSE of 0 which is beacuse of overfitting, so to get actual error value we will now apply cross validation score on each of the three models

models = {
    "Linear Regression": lin_reg,
    "Decision Tree": tree_reg,
    "Random Forest": forest_reg
}

for name, model in models.items():
    cv_rmse = -cross_val_score(
        clone(model),
        housing_prepared,
        housing_labels,
        scoring="neg_root_mean_squared_error",
        cv=10
    )
    print(f"{name} CV RMSEs: {cv_rmse}")
    print(f"{name} CV RMSE Mean: {cv_rmse.mean():.2f}, Std: {cv_rmse.std():.2f}\n")

# As Random Forest has the lowest RMSE, we will use it for further predictions