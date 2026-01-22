🏡 House Price Prediction


📌 Project Overview
This project focuses on predicting house prices using machine learning techniques. The goal is to build a robust model that can estimate property values based on various features such as location, size, number of rooms, and other attributes.

After experimenting with multiple approaches, the final model uses XGBoost, which achieved an accuracy of ~83% after hyperparameter tuning.

⚙️ Data Preprocessing
Before modeling, several preprocessing steps were applied to ensure clean and reliable data:

Basic Checks

Handling missing values (imputation or dropping depending on feature importance).

Removing duplicate records.

Checking for inconsistencies in categorical values.

Feature Engineering

Encoding categorical variables (e.g., one‑hot encoding for location, property type).

Creating new derived features (e.g., price per square foot, age of property).

Binning continuous variables where appropriate.

Scaling & Transformation

Standardization/normalization of numerical features to stabilize model training.

Log transformation for skewed variables (like price).

📊 Model Development
Algorithm Used: XGBoost (Extreme Gradient Boosting).

Hyperparameter Tuning: Grid search and cross‑validation were applied to optimize parameters such as:

Learning rate

Max depth

Number of estimators

Subsample ratio

Performance: Achieved ~83% accuracy on the test set after tuning.
