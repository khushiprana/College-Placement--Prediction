# CareerPredict — College Placement Prediction & Analytics

A Python + Streamlit machine-learning web app for exploring student placement data and predicting placement outcomes.

## Features
- Dataset overview, missing-value and duplicate checks
- Exploratory graphs: placement distribution, feature boxplot, correlation heatmap
- Missing-value imputation, numeric scaling, categorical one-hot encoding
- Comparison of Logistic Regression, Decision Tree, and Random Forest
- Accuracy, precision, recall, F1-score, confusion matrix and classification report
- Permutation feature importance
- Single-student prediction with model class probabilities
- Batch CSV prediction and downloadable results
- Basic rule-based suggestions for selected low feature values

## Dataset
Kaggle: https://www.kaggle.com/datasets/sahilislam007/college-student-placement-factors-dataset

The dataset is described as simulated. Do not claim that it represents verified real-world placement outcomes.

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended):
   - Windows: `python -m venv .venv` then `.venv\\Scripts\\activate`
   - macOS/Linux: `python -m venv .venv` then `source .venv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Download data: `python download_data.py` (KaggleHub may require Kaggle credentials).
   Alternatively download the CSV manually and put it in `data/`.
6. Start app: `streamlit run app.py`


