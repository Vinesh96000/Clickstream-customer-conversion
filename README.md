# Customer Conversion Analysis for E-commerce

This project develops an end-to-end data science application to analyze customer clickstream data from an e-commerce website. The goal is to predict customer behavior, estimate potential revenue, and segment customers into distinct groups to enable data-driven marketing strategies.

---

## 🚀 Features

This project accomplishes three core machine learning tasks, all wrapped in an interactive web application:

1.  **Classification:** Predicts whether a customer session will result in a purchase (`Conversion`) or not (`No Conversion`).
2.  **Regression:** Estimates the potential revenue (the `price` of the item) a customer is likely to generate if they convert.
3.  **Clustering:** Segments customers into four distinct behavioral personas using unsupervised learning.
4.  **Interactive Application:** A user-friendly web app built with Streamlit to demonstrate the models' capabilities.
5.  **Experiment Tracking:** Uses MLflow to log and compare the performance of different classification models.

---

## 🛠️ Technologies Used

-   **Programming Language:** Python 3.11
-   **Data Manipulation & Analysis:** Pandas, NumPy
-   **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn
-   **MLOps:** MLflow for experiment tracking
-   **Web Application:** Streamlit
-   **Data Visualization:** Matplotlib, Seaborn

---

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
    cd Your-Repo-Name
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your web browser.

---

## 📊 Model Results

### Classification (Predicting Conversion)
The XGBoost model was selected as the champion model for its high precision and best overall F1-Score.
-   **Precision (for purchases):** 0.67
-   **Recall (for purchases):** 0.38
-   **F1-Score (for purchases):** 0.48

### Regression (Predicting Revenue)
The XGBRegressor model demonstrated high accuracy in predicting the price of items for converting customers.
-   **Root Mean Squared Error (RMSE):** $1.50
-   **R-squared (R²):** 0.98

### Clustering (Customer Personas)
Four distinct customer segments were identified using K-Means:
1.  **Quick, High-Value Shoppers:** Decisive, high-spending customers.
2.  **Explorers / Window Shoppers:** Highly engaged users who browse extensively.
3.  **Bargain Hunters:** Price-sensitive users focused on sales.
4.  **Focused, Everyday Shoppers:** Users looking for specific, non-sale items.

---

## ✒️ Author

-   Project completed by **[Vinesh J]** as part of the Guvi AML course.
