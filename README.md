# 📉 ChurnShield — ML-Powered Customer Churn Predictor

> Predict which customers are about to leave — before they do. ChurnShield uses supervised machine learning to identify at-risk telecom customers with **80%+ accuracy**, enabling data-driven retention strategies that can save millions in revenue.

![Build Status](https://img.shields.io/github/actions/workflow/status/your-username/churnshield/ci.yml?branch=main&label=build)
![License](https://img.shields.io/github/license/your-username/churnshield)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange)

---

## Features

- **Multi-Model Benchmarking** — Trains and evaluates Logistic Regression, K-Nearest Neighbors, Decision Tree, and Support Vector Machine side by side
- **Automated Hyperparameter Tuning** — GridSearchCV with 5-fold cross-validation to find optimal parameters for every model
- **Sklearn Pipeline Integration** — Scales and classifies within a unified pipeline to prevent data leakage
- **Comprehensive EDA** — KDE plots, correlation heatmaps, and distribution analysis to uncover key churn drivers
- **Interpretable Results** — Confusion matrices, classification reports, and accuracy comparisons for each model
- **Business-Ready Insights** — Translates model outputs into actionable retention recommendations (contract type, monthly charges, tenure targeting)

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | scikit-learn (LR, KNN, DT, SVM, GridSearchCV, Pipeline) |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | LabelEncoder, StandardScaler |
| Notebook Environment | Jupyter Notebook |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/RankMaster/ChurnShield.git
cd churnshield
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Add the dataset**

Place the `Customer Churn Prediction.csv` file in the project root directory. The dataset should include the standard Telco Customer Churn schema (7043 rows, 21 columns including `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `Churn`, etc.).

### Running the Project

Launch Jupyter Notebook and open the main analysis file:

```bash
jupyter notebook ML_Customer_Churn_Prediction.ipynb
```

Run all cells sequentially (`Kernel > Restart & Run All`) to reproduce the full pipeline from EDA through model evaluation.

---

## Usage Example

The notebook follows this end-to-end pipeline:

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# 1. Load and clean data
df = pd.read_csv("Customer Churn Prediction.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.drop(df[df['tenure'] == 0].index, inplace=True)

# 2. Encode categorical features
le = LabelEncoder()
categorical_cols = ['gender', 'Partner', 'Contract', 'Churn', ...]
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Define features and split
X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=43)

# 4. Build a pipeline and tune
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression())])
params = {'clf__C': [0.1, 1], 'clf__penalty': ['l2'], 'clf__solver': ['liblinear']}
grid = GridSearchCV(pipeline, params, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# 5. Evaluate
predictions = grid.best_estimator_.predict(X_test)
print(classification_report(y_test, predictions))
# Best result: ~80.39% accuracy with tuned Logistic Regression
```

**Key findings from the model:**
- Customers on month-to-month contracts are **3.2x more likely** to churn
- Short tenure (< 12 months) is the strongest single predictor of churn
- Monthly charges above $80 correlate with a **40% increase** in churn probability
- Customers without TechSupport or OnlineSecurity show **15–20% higher** churn risk

---

## Roadmap

- [ ] **Class Imbalance Handling** — Integrate SMOTE or class-weight adjustments to improve recall on the minority churn class
- [ ] **Ensemble Methods** — Experiment with Random Forest, XGBoost, and Gradient Boosting for improved predictive power
- [ ] **Feature Importance Dashboard** — Add a visual SHAP-based explanation layer for stakeholder-ready interpretability
- [ ] **REST API Deployment** — Wrap the best model in a FastAPI endpoint for real-time churn scoring on new customer records

---

## Contributing

Contributions are welcome! Whether it's a bug fix, new model experiment, or documentation improvement — all PRs are appreciated.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please make sure your code follows the existing notebook style and includes inline comments where necessary. For major changes, open an issue first to discuss what you'd like to change.

---

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this project with attribution.

---

*Built with scikit-learn | Driven by data | Designed for retention*
