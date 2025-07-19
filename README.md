# 🏠 House Price Prediction using Machine Learning

A machine learning project that predicts California house prices using features like location, income, and proximity to the ocean. It includes EDA, preprocessing, model evaluation, and a user-friendly inference setup.

---

## 📁 Project Structure

```bash
house-price-prediction/
├── housing.csv          # Full dataset
├── input.csv            # Sample input for inference
├── model.ipynb          # EDA and visualization
├── preprocessing.py     # Preprocessing and model evaluation
├── main.py              # Final training + inference
├── requirements.txt     # Project dependencies
├── .gitignore           # Ignored files like model.pkl
└── README.md            # This file
```

---

## 🧠 Project Overview

The project uses the California Housing Prices dataset to build a regression model that predicts median house values. It evaluates multiple models and finalizes the Random Forest Regressor based on performance.

### Key Features:

* 📊 EDA and insights using pandas, matplotlib, seaborn
* ⚙️ Preprocessing pipeline with missing value handling, scaling, encoding
* 🧪 Model comparison (Linear, Decision Tree, Random Forest)
* 🧠 Final inference using manual input or CSV batch

---

## 📊 EDA Notebook

**File:** `model.ipynb`

Includes:

* Histograms, scatter plots, and correlation analysis
* Insights on income, population, location effects
* Markdown summary of findings and steps for modeling

---

## 🛠️ Preprocessing & Model Evaluation

**File:** `preprocessing.py`

Includes:

* Feature engineering and pipeline setup
* Model training and performance comparison
* Saves best-performing model details for reference

---

## ✅ Final Model & Inference

**File:** `main.py`

Performs:

* Final model training if not already saved
* Inference with `model.pkl` and `pipeline.pkl`
* Prediction from:

  * ⌨️ Manual user input
  * 📄 Batch input from `input.csv`

**Note:**
`input.csv` is a pre-formatted file containing sample test data (taken from the original dataset). It is provided to demonstrate batch prediction by passing multiple records at once.

### Sample Run

```bash
$ python main.py
Do you want to enter data manually for prediction? (yes/no): yes
Enter values for the following features:
... [feature inputs]
✅ Predicted House Price: $208,643.15
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

```bash
python main.py
```

* Trains and saves model if not available
* Otherwise loads and performs prediction

---

## 🔁 How to Fork

1. Click **Fork** on the top right of this repo
2. Clone your fork:

```bash
git clone https://github.com/AlsoMeParth/house-price-prediction.git
cd house-price-prediction
```

3. Install dependencies and run the project

---

📦 Built with Python, scikit-learn, and 💡
