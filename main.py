import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # TRAINING PHASE
    housing = pd.read_csv("housing.csv")
    housing['income_cat'] = pd.cut(housing["median_income"], 
                                   bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        housing.loc[test_index].drop("income_cat",axis=1).to_csv("input.csv",index=False)
        housing = housing.loc[train_index].drop("income_cat", axis=1)

    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attribs = housing_features.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.")

else:
    # INFERENCE PHASE
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    def predict_from_input(model, pipeline, input_dict):
        input_df = pd.DataFrame([input_dict])
        transformed = pipeline.transform(input_df)
        prediction = model.predict(transformed)
        return prediction[0]
    def get_user_input():
        print("\nEnter values for the following features:")
        user_data = {}

        user_data["longitude"] = float(input("Longitude: "))
        user_data["latitude"] = float(input("Latitude: "))
        user_data["housing_median_age"] = float(input("Housing Median Age: "))
        user_data["total_rooms"] = float(input("Total Rooms: "))
        user_data["total_bedrooms"] = float(input("Total Bedrooms: "))
        user_data["population"] = float(input("Population: "))
        user_data["households"] = float(input("Households: "))
        user_data["median_income"] = float(input("Median Income: "))
    
        # Categorical value
        print("Ocean Proximity Options: ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']")
        user_data["ocean_proximity"] = input("Ocean Proximity: ").upper().strip()

        return user_data

    choice = input("\nDo you want to enter data manually for prediction? (yes/no): ").lower()
    if choice == 'yes':
        user_input = get_user_input()
        prediction = predict_from_input(model, pipeline, user_input)
        print(f"Predicted Median House Value: {prediction}")    
    else:
        file_name = input("Enter the name of the input CSV file (NOTE->attribute names must match training data i.e. housing.csv) :")
        input_data = pd.read_csv(f"{file_name}.csv")
        transformed_input = pipeline.transform(input_data)
        predictions = model.predict(transformed_input)
        input_data["median_house_value"] = predictions
        input_data.to_csv("output.csv", index=False)
        print("Inference complete. Results saved to output.csv")

