# ChurnGuard: End-to-End Customer Retention Analytics

ChurnGuard is an end-to-end machine learning project designed to predict customer churn for a telecommunications provider. By identifying customers who are likely to cancel their subscription, businesses can implement proactive retention strategies to reduce revenue loss and improve customer lifetime value.

This project demonstrates the complete ML workflow from data preprocessing and feature engineering to model training, evaluation, and real-time inference through an interactive dashboard.

The solution is powered by a high-performance XGBoost model, an automated preprocessing pipeline, and a Streamlit dashboard for interactive predictions.

The project uses the **Telco Customer Churn** dataset from Kaggle. It contains information about 7,043 customers, including demographics, account information, and service usage.
[Link to Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/miranda189923/churnguard-customer-retention-ml.git](https://github.com/miranda189923/churnguard-customer-retention-ml.git)
   cd ChurnGuard

2. **Install dependencies:**
    ```bash
    pip install --upgrade pip setuptools wheel
    pip install -r requirements.txt

3. **Train:**
    This will run hyperparameter search and save models/pipeline.pkl. Training time depends on machine resources.
    ```bash
    python -m src.train

4. **Launch dashboard:**
    ```bash
    python -m streamlit run app\streamlit_app.py