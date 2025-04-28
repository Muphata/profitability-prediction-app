# ============================================
# Streamlit App - Profitability Prediction
# ============================================

# Step 1: Import Libraries
# --------------------------------------------
import streamlit as st
import pandas as pd
import joblib

# Step 2: Load the trained model
# --------------------------------------------
model = joblib.load('model.joblib')

# Step 3: App Title and Introduction
# --------------------------------------------
st.title('💵 Profitability Prediction App')

st.markdown("""
Welcome to the Profitability Prediction App!  
Fill in the order details below and click **Predict Profit** to estimate the expected profit.
""")

# Step 4: User Inputs
# --------------------------------------------

# Numeric Inputs
sales = st.number_input('Enter Sales Amount ($)', min_value=0.0, format="%.2f")
discount = st.number_input('Enter Discount (between 0 and 1)', min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
quantity = st.number_input('Enter Quantity', min_value=1, step=1)
days_to_ship = st.number_input('Enter Days to Ship (Order to Shipping)', min_value=0, step=1)

# Categorical Inputs
category = st.selectbox('Select Product Category', ['Furniture', 'Office Supplies', 'Technology'])
sub_category = st.selectbox('Select Sub-Category', [
    'Bookcases', 'Chairs', 'Tables', 'Furnishings', 
    'Appliances', 'Binders', 'Envelopes', 'Labels', 
    'Paper', 'Storage', 'Art', 'Fasteners', 'Supplies', 
    'Accessories', 'Copiers', 'Phones', 'Machines']
)
ship_mode = st.selectbox('Select Shipping Mode', ['First Class', 'Second Class', 'Standard Class', 'Same Day'])
region = st.selectbox('Select Region', ['West', 'East', 'Central', 'South'])
# Predefined dictionary of States and their Cities
state_city_dict = {
    'California': ['Los Angeles', 'San Francisco', 'San Diego', 'Sacramento'],
    'New York': ['New York City', 'Buffalo', 'Rochester'],
    'Texas': ['Houston', 'Dallas', 'Austin'],
    'Washington': ['Seattle', 'Spokane', 'Tacoma'],
    'Florida': ['Miami', 'Orlando', 'Tampa'],
    'Illinois': ['Chicago', 'Springfield', 'Naperville'],
    'Pennsylvania': ['Philadelphia', 'Pittsburgh', 'Allentown'],
    'Ohio': ['Columbus', 'Cleveland', 'Cincinnati'],
    'Colorado': ['Denver', 'Colorado Springs', 'Aurora']
}

# Step 1: Select State
state = st.selectbox('Select State', list(state_city_dict.keys()))

# Step 2: Select City based on State
city = st.selectbox('Select City', state_city_dict[state])

# Step 5: Preprocess the Inputs
# --------------------------------------------

# Create DataFrame for a single input
input_dict = {
    'Sales': [sales],
    'Discount': [discount],
    'Quantity': [quantity],
    'Days_to_Ship': [days_to_ship],
    
    # Encoding category features
    f'Category_{category}': [1],
    f'Sub-Category_{sub_category}': [1],
    f'Ship Mode_{ship_mode}': [1],
    f'Region_{region}': [1],
    f'State_{state}': [1],
    f'City_{city}': [1],
}

# Model was trained with many dummy columns, some inputs might be missing
# So create a DataFrame then add missing columns with 0
input_df = pd.DataFrame(input_dict)

# Make sure all columns that the model expects exist
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0

# Arrange columns in the same order
input_df = input_df[model.feature_names_in_]

# Step 6: Make Prediction
# --------------------------------------------

if st.button('Predict Profit'):
    prediction = model.predict(input_df)[0]
    
    # Step 7: Show Prediction
    # --------------------------------------------
    if prediction >= 0:
        st.success(f'✅ Estimated Profit: ${prediction:.2f}')
    else:
        st.error(f'⚠️ Estimated Loss: ${prediction:.2f}')
