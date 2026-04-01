# End-to-End Customer Retention Analytics

In the telecommunications industry, customer acquisition costs are significantly higher than retention costs. Churn is the "leaky bucket" problem where a small reduction in the attrition rate leads to a massive increase in long-term profitability.

This project is an end-to-end predictive analytics suite designed to identify at-risk telecommunications subscribers. By combining a **Stacked Ensemble** machine learning pipeline with a modern React web interface, this project provides a professional-grade tool for reducing customer attrition and protecting revenue.

## Technical Stack
- **Frontend:** React
- **Backend:** FastAPI
- **Base Models:** LightGBM, XGBoost, and CatBoost (independently tuned via Optuna)
- **Meta-Learner:** Logistic Regression aggregator for maximum prediction stability
- **Performance:** Achieved a **Stacked OOF AUC of 0.85**, demonstrating top-tier predictive power

## Dataset
The project uses the **Telco Customer Churn** dataset from Kaggle. It contains information about 7,043 customers, including demographics, account information, and service usage.
[Link to Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Installation & Usage

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/miranda189923/churn-prediction.git
   cd ChurnGuard
    ```

2. **Install Dependencies:** 
    ```bash
    python -m venv .venv
    source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Training (optional):**
    
    A pre-trained stacked ensemble model (ml/churn_model.joblib) is included in the repository, allowing the application to run immediately. If you would like to modify the feature engineering pipeline, tune hyperparameters, or retrain the ensemble from scratch, run:

    ```bash
    python ml/train.py
    ```

4.  **Backend:**
    ```bash
    uvicorn backend.main:app --reload
    ```

5. **Start the Website:**

    The application will be available at `http://localhost:5173/`. The backend will automatically detect the trained model file and use it for real-time predictions.

    ```bash
    cd frontend
    npm install
    npm run dev
    ```