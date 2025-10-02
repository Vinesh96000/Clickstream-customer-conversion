import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Customer Conversion Analysis", layout="wide")

# --- Load Artifacts ---
@st.cache_resource
def load_artifacts():
    """Loads all the necessary models and data."""
    classification_model = joblib.load('classification_model.joblib')
    classification_cols = joblib.load('classification_model_columns.joblib')
    regression_model = joblib.load('regression_model.joblib')
    regression_cols = joblib.load('regression_model_columns.joblib')
    clustered_data = pd.read_csv('data/clustered_sessions.csv')
    return classification_model, classification_cols, regression_model, regression_cols, clustered_data

classification_model, classification_cols, regression_model, regression_cols, clustered_data = load_artifacts()

# --- Page Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Classification", "💰 Regression", "🧑‍🤝‍🧑 Clustering"])

# --- Home Page ---
if page == "🏠 Home":
    st.title("Customer Conversion Analysis for E-commerce 🛍️")
    st.markdown("""
    Welcome to the Customer Conversion Analysis application. This tool leverages machine learning to provide key insights into customer behavior.
    Navigate through the different sections using the sidebar to predict customer conversion, estimate revenue, and explore customer segments.

    ### Project Overview
    This application is the final product of a comprehensive data science project that involved:
    - **Classification:** Predicting whether a customer will make a purchase.
    - **Regression:** Estimating the revenue from a purchasing customer.
    - **Clustering:** Segmenting customers into distinct behavioral groups.
    """)
    st.image('plots/category_distribution.png', caption='Popularity of Main Product Categories')

# --- Classification Page ---
elif page == "🔮 Classification":
    st.header("Predict Customer Conversion")
    st.markdown("Input customer session details to predict if they will make a purchase.")

    price = st.slider("Price of Item Viewed", min_value=18, max_value=82, value=45)
    page1_cat = st.selectbox("Main Product Category", options=[1, 2, 3, 4], format_func=lambda x: f"Category {x}")
    country = st.selectbox("Country ID", options=list(range(1, 48)), index=28)
    clicks = st.number_input("Total Clicks in Session", min_value=1, value=10)

    if st.button("Predict Conversion"):
        input_df = pd.DataFrame(columns=classification_cols)
        new_row = pd.DataFrame([pd.Series(0, index=input_df.columns)])
        input_df = pd.concat([input_df, new_row], ignore_index=True)
        
        input_df['order'] = clicks
        input_df['price'] = price
        
        if f'country_{country}' in input_df.columns:
            input_df[f'country_{country}'] = 1
        if f'page1_main_category_{page1_cat}' in input_df.columns:
            input_df[f'page1_main_category_{page1_cat}'] = 1
            
        input_df = input_df[classification_cols]

        # --- THE FIX IS HERE ---
        # Convert all columns to a numeric type before prediction
        input_df = input_df.astype(float)

        prediction = classification_model.predict(input_df)
        prediction_proba = classification_model.predict_proba(input_df)

        if prediction[0] == 1:
            st.success(f"Prediction: Customer WILL Convert! (Probability: {prediction_proba[0][1]:.2%})")
        else:
            st.error(f"Prediction: Customer will NOT Convert (Probability: {prediction_proba[0][1]:.2%})")

# --- Regression Page ---
elif page == "💰 Regression":
    st.header("Estimate Potential Revenue")
    st.markdown("Input details for a *converting* customer to estimate the price of the item they will purchase.")

    page1_cat_reg = st.selectbox("Main Product Category", options=[1, 2, 3, 4], format_func=lambda x: f"Category {x}")
    country_reg = st.selectbox("Country ID", options=list(range(1, 48)), index=28)
    clicks_reg = st.number_input("Total Clicks in Session", min_value=1, value=15)
    day_of_week_reg = st.selectbox("Day of the Week", options=list(range(7)), format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])

    if st.button("Estimate Revenue"):
        input_df_reg = pd.DataFrame(columns=regression_cols)
        new_row_reg = pd.DataFrame([pd.Series(0, index=input_df_reg.columns)])
        input_df_reg = pd.concat([input_df_reg, new_row_reg], ignore_index=True)
        
        input_df_reg['order'] = clicks_reg
        input_df_reg['day_of_week'] = day_of_week_reg
        if f'country_{country_reg}' in input_df_reg.columns:
            input_df_reg[f'country_{country_reg}'] = 1
        if f'page1_main_category_{page1_cat_reg}' in input_df_reg.columns:
            input_df_reg[f'page1_main_category_{page1_cat_reg}'] = 1
            
        input_df_reg = input_df_reg[regression_cols]
        
        # --- THE FIX IS HERE ---
        # Convert all columns to a numeric type before prediction
        input_df_reg = input_df_reg.astype(float)
        
        revenue_prediction = regression_model.predict(input_df_reg)
        st.success(f"Estimated Revenue from this session: ${revenue_prediction[0]:.2f}")

# --- Clustering Page ---
elif page == "🧑‍🤝‍🧑 Clustering":
    st.header("Explore Customer Segments")
    st.markdown("Here are the four customer personas discovered from the data.")
    
    personas = {
        0: {"title": "Explorers / Window Shoppers", "desc": "These users spend the most time on the site, clicking through many different product categories. They are highly engaged but may not have a specific purchase in mind."},
        1: {"title": "Focused, Everyday Shoppers", "desc": "These users look at lower-priced, non-sale items and don't browse much. They are likely looking for specific, everyday products."},
        2: {"title": "Quick, High-Value Shoppers", "desc": "The most valuable segment. They are decisive, view expensive items, and do not look for discounts. They know what they want and buy it efficiently."},
        3: {"title": "Bargain Hunters", "desc": "This group is defined by their search for deals. Every session involves a visit to the sale page, and they focus on lower-priced items."}
    }
    
    selected_cluster = st.selectbox("Choose a Persona to Explore", options=list(personas.keys()), format_func=lambda x: personas[x]["title"])
    
    st.subheader(personas[selected_cluster]["title"])
    st.write(personas[selected_cluster]["desc"])
    
    st.markdown("---")
    
    # Filter the dataframe based on the selected persona and display it.
    st.subheader(f"Data for {personas[selected_cluster]['title']}")
    filtered_df = clustered_data[clustered_data['cluster'] == selected_cluster]
    st.dataframe(filtered_df)