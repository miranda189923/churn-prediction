# End-to-End Customer Retention Analytics

In the telecommunications industry, customer acquisition costs are significantly higher than retention costs. Churn is the "leaky bucket" problem where a small reduction in the attrition rate leads to a massive increase in long-term profitability.

This project is an end-to-end predictive analytics suite designed to identify at-risk telecommunications subscribers. By combining a **Stacked Ensemble** machine learning pipeline with a modern React web interface, this project provides a professional-grade tool for reducing customer attrition and protecting revenue.

## Technical Stack
- **Frontend:** React (Vite)
- **Backend:** FastAPI (Python)
- **Base Models:** LightGBM, XGBoost, and CatBoost (independently tuned via Optuna)
- **Meta-Learner:** Logistic Regression aggregator for maximum prediction stability
- **Performance:** Achieved a **Stacked OOF AUC of 0.84730**, demonstrating top-tier predictive power

## Dataset
The project uses the **Telco Customer Churn** dataset from Kaggle. It contains information about 7,043 customers, including demographics, account information, and service usage.
[Link to Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/miranda189923/churnguard-customer-retention-ml.git](https://github.com/miranda189923/churnguard-customer-retention-ml.git)
   cd ChurnGuard
    ```

2. **Install Dependencies**

    **React Frontend:**
    ```bash
    npm install
    ```

    **Python Backend & ML:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Train:** 

    This script performs feature engineering, hyperparameter tuning, and saves the ensemble to ml/churn_model.joblib.

    ```bash
    python ml/train.py
    ```

4. **Start the website:**

    The application will be available at `http://localhost:3000`. The backend will automatically detect the trained model file and use it for real-time predictions.

    ```bash
    npm run dev
    ```