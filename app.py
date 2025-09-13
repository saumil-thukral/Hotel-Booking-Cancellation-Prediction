import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt

# Set up the Streamlit page. This MUST be the first Streamlit command.
st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="wide")

# Define the numerical and categorical features based on the user's new lists.
numerical_features = [
    'lead_time', 'previous_cancellations', 'booking_changes', 'days_in_waiting_list',
    'adr', 'total_of_special_requests', 'total_guests', 'total_nights',
    'prev_cancel_rate', 'country_freq'
]

categorical_features = [
    'hotel', 'meal', 'market_segment', 'distribution_channel', 'reserved_room_type',
    'assigned_room_type', 'deposit_type', 'customer_type', 'is_family'
]

# A comprehensive list of possible values for each categorical feature,
# assuming they are the same as in the original dataset.
categorical_values = {
    'hotel': ['City Hotel', 'Resort Hotel'],
    'meal': ['BB', 'FB', 'HB', 'SC', 'Undefined'],
    'market_segment': ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Complementary', 'Groups', 'Undefined', 'Aviation'],
    'distribution_channel': ['Direct', 'Corporate', 'TA/TO', 'Undefined', 'GDS'],
    'reserved_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'P'],
    'assigned_room_type': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'P'],
    'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
    'customer_type': ['Transient', 'Contract', 'Transient-Party', 'Group'],
    'is_family': [0, 1]
}

# Load the trained model. Make sure the pkl file is in the same directory.
try:
    model = joblib.load('hotel_cancellation_model.pkl')
except FileNotFoundError:
    st.error("Error: The 'hotel_cancellation_model.pkl' file was not found. Please make sure it's in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# Create tabs for the dashboard, summary, visualizations, and code explanation
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Summary", "Visualizations", "Code Explanation"])

with tab1:
    st.title("Hotel Booking Cancellation Prediction")
    st.markdown("Enter the booking details below to predict if the booking will be canceled.")
    
    # Create input widgets
    st.sidebar.header("Booking Details")
    with st.sidebar:
        st.markdown("### Numerical Features")
        # Using sliders for a better user experience
        lead_time = st.slider("Lead Time (days)", min_value=0, max_value=500, value=100)
        previous_cancellations = st.slider("Previous Cancellations", min_value=0, max_value=10, value=0)
        booking_changes = st.slider("Booking Changes", min_value=0, max_value=20, value=0)
        days_in_waiting_list = st.slider("Days in Waiting List", min_value=0, max_value=200, value=0)
        adr = st.slider("ADR (Average Daily Rate)", min_value=0.0, max_value=500.0, value=100.0, step=0.1)
        total_of_special_requests = st.slider("Total Special Requests", min_value=0, max_value=5, value=0)
        total_guests = st.slider("Total Guests", min_value=0, max_value=10, value=2)
        total_nights = st.slider("Total Nights", min_value=0, max_value=30, value=3)
        prev_cancel_rate = st.slider("Previous Cancellation Rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        country_freq = st.slider("Country Frequency", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

        st.markdown("### Categorical Features")
        hotel = st.selectbox("Hotel Type", categorical_values['hotel'])
        meal = st.selectbox("Meal", categorical_values['meal'])
        market_segment = st.selectbox("Market Segment", categorical_values['market_segment'])
        distribution_channel = st.selectbox("Distribution Channel", categorical_values['distribution_channel'])
        reserved_room_type = st.selectbox("Reserved Room Type", categorical_values['reserved_room_type'])
        assigned_room_type = st.selectbox("Assigned Room Type", categorical_values['assigned_room_type'])
        deposit_type = st.selectbox("Deposit Type", categorical_values['deposit_type'])
        customer_type = st.selectbox("Customer Type", categorical_values['customer_type'])
        is_family = st.selectbox("Is Family?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")


    # Create a dictionary of the input values from the widgets
    user_input = {
        'lead_time': lead_time,
        'previous_cancellations': previous_cancellations,
        'booking_changes': booking_changes,
        'days_in_waiting_list': days_in_waiting_list,
        'adr': adr,
        'total_of_special_requests': total_of_special_requests,
        'total_guests': total_guests,
        'total_nights': total_nights,
        'prev_cancel_rate': prev_cancel_rate,
        'country_freq': country_freq,
        'hotel': hotel,
        'meal': meal,
        'market_segment': market_segment,
        'distribution_channel': distribution_channel,
        'reserved_room_type': reserved_room_type,
        'assigned_room_type': assigned_room_type,
        'deposit_type': deposit_type,
        'customer_type': customer_type,
        'is_family': is_family
    }

    # --- Rebuilding the input data to match the model's expected format ---
    try:
        # Get the list of features the model was trained on
        model_features = model.feature_names_in_

        # Create a dictionary to hold the full input data with all one-hot encoded columns
        full_input_data = {}

        # Add numerical features
        for col in numerical_features:
            full_input_data[col] = user_input[col]

        # Add one-hot encoded categorical features
        for col in categorical_features:
            for val in categorical_values[col]:
                full_input_data[f'{col}_{val}'] = 1 if user_input[col] == val else 0

        # Create the DataFrame from the dictionary, ensuring the column order is correct
        input_df = pd.DataFrame([full_input_data], columns=model_features)

    except Exception as e:
        st.error(f"Error preparing input data for the model: {e}")
        st.stop()


    # Prediction button
    if st.button("Predict Cancellation"):
        try:
            # Get the prediction from the model
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)

            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"Prediction: This booking is **likely to be canceled**.")
                st.markdown(f"**Probability of Cancellation:** {prediction_proba[0][1]:.2f}")
            else:
                st.success(f"Prediction: This booking is **unlikely to be canceled**.")
                st.markdown(f"**Probability of Cancellation:** {prediction_proba[0][1]:.2f}")

        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your model and input data. Error: {e}")


with tab2:
    st.title("Model and Feature Summary")
    st.markdown("This tab provides a summary of the machine learning model used for prediction and the features it relies on.")

    st.subheader("Model Overview")
    st.info("""
    The model is a **Random Forest Classifier** trained to predict hotel booking cancellations.
    It was trained on a comprehensive dataset of hotel bookings and tuned to optimize its performance.
    """)

    st.subheader("Model Performance")
    st.markdown("Based on the evaluation of the tuned model on a test set, it achieved the following performance metrics:")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "83%")
    with col2:
        st.metric("ROC AUC Score", "0.745")

    st.subheader("Features Used for Prediction")
    st.markdown("The model uses a combination of numerical and categorical features to make its predictions. These are based on information available at the time of booking.")

    st.markdown("#### Numerical Features")
    st.markdown("- **`lead_time`**: The number of days between the booking date and the arrival date.")
    st.markdown("- **`previous_cancellations`**: The number of previous bookings that were canceled by the customer.")
    st.markdown("- **`booking_changes`**: The number of changes made to the booking before check-in.")
    st.markdown("- **`days_in_waiting_list`**: The number of days the booking was in the waiting list before confirmation.")
    st.markdown("- **`adr`**: Average daily rate, which is the total cost of the booking divided by the number of nights.")
    st.markdown("- **`total_of_special_requests`**: The number of special requests made by the customer.")
    st.markdown("- **`total_guests`**: The total number of adults and children in the booking.")
    st.markdown("- **`total_nights`**: The total number of nights the guest is staying.")
    st.markdown("- **`prev_cancel_rate`**: The ratio of previous cancellations to total previous bookings.")
    st.markdown("- **`country_freq`**: The frequency of the customer's country of origin in the dataset.")

    st.markdown("#### Categorical Features")
    st.markdown("These features are converted into a numerical format (one-hot encoding) by the model's preprocessing pipeline.")
    st.markdown("- **`hotel`**: Whether the booking is at a 'City Hotel' or 'Resort Hotel'.")
    st.markdown("- **`meal`**: The type of meal booked (e.g., 'BB' for Bed & Breakfast, 'FB' for Full Board).")
    st.markdown("- **`market_segment`**: The market segment designation (e.g., 'Online TA', 'Direct', 'Corporate').")
    st.markdown("- **`distribution_channel`**: The channel through which the booking was made.")
    st.markdown("- **`reserved_room_type`**: The room type initially reserved by the customer.")
    st.markdown("- **`assigned_room_type`**: The room type assigned to the customer.")
    st.markdown("- **`deposit_type`**: Whether a deposit was made ('No Deposit', 'Non Refund', 'Refundable').")
    st.markdown("- **`customer_type`**: The type of customer ('Transient', 'Contract', etc.).")
    st.markdown("- **`is_family`**: A feature indicating if the booking includes children or babies.")
    
with tab3:
    st.title("Booking Data Visualizations")
    st.markdown("This tab provides a visual overview of key booking trends and patterns.")
    st.info("Note: The visualizations below are based on hypothetical data for demonstration purposes, as the original dataset is not available within the app.")

    st.subheader("Cancellations by Hotel Type")
    # Hypothetical data for cancellations by hotel type
    df_cancellation_by_hotel = pd.DataFrame({
        'Hotel': ['City Hotel', 'Resort Hotel'],
        'Cancellations': [30000, 10000]
    })
    st.bar_chart(df_cancellation_by_hotel, x='Hotel', y='Cancellations', color="#ff4b4b")
    st.markdown("This chart shows that city hotels generally have a higher number of cancellations compared to resort hotels.")

    st.subheader("Lead Time Distribution")
    # Hypothetical data for lead time distribution
    df_lead_time = pd.DataFrame({
        'Lead Time': np.random.randint(0, 500, 500),
        'Count': 1
    })
    hist_chart = alt.Chart(df_lead_time).mark_bar(color='#63B2C8').encode(
        alt.X("Lead Time", bin=True),
        y='count()',
        tooltip=['count()']
    )
    st.altair_chart(hist_chart, use_container_width=True)
    st.markdown("This histogram shows the distribution of lead times for bookings, highlighting common booking patterns.")

    st.subheader("Market Segment Counts")
    df_market_segment = pd.DataFrame({
        'market_segment': ['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate', 'Complementary', 'Aviation'],
        'counts': [50000, 25000, 15000, 10000, 5000, 2000, 500]
    })
    df_market_segment = df_market_segment.sort_values('counts', ascending=False)
    chart = alt.Chart(df_market_segment).mark_bar(color='#7C89A3').encode(
        x=alt.X('market_segment', sort='-y', title='Market Segment'),
        y=alt.Y('counts', title='Number of Bookings'),
        tooltip=['market_segment', 'counts']
    ).properties(
        title='Market Segment Counts'
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("This chart displays the number of bookings per market segment, with 'Online TA' being the most popular.")

    st.subheader("Number of Bookings per Month")
    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    df_bookings_month = pd.DataFrame({
        'arrival_date_month': month_order,
        'bookings': [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 3500, 3000, 2000, 1500]
    })
    chart = alt.Chart(df_bookings_month).mark_bar(color='#63B2C8').encode(
        x=alt.X('arrival_date_month', sort=month_order, title='Arrival Month'),
        y=alt.Y('bookings', title='Number of Bookings'),
        tooltip=['arrival_date_month', 'bookings']
    ).properties(
        title='Number of Bookings per Month'
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("This chart shows the distribution of bookings throughout the year, with a clear peak during the summer months.")

    st.subheader("Cancellation Rate by Arrival Month")
    df_cancellation_rate_month = pd.DataFrame({
        'arrival_date_month': month_order,
        'cancellation_rate': [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.48, 0.42, 0.35, 0.3, 0.25]
    })
    chart = alt.Chart(df_cancellation_rate_month).mark_bar(color='#ff4b4b').encode(
        x=alt.X('arrival_date_month', sort=month_order, title='Arrival Month'),
        y=alt.Y('cancellation_rate', title='Cancellation Rate'),
        tooltip=['arrival_date_month', 'cancellation_rate']
    ).properties(
        title='Cancellation Rate by Arrival Month'
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("This chart highlights how the cancellation rate varies by month. It can show which seasons have higher risk of cancellation.")

    st.subheader("ADR by Market Segment")
    df_adr_market = pd.DataFrame({
        'market_segment': np.random.choice(['Online TA', 'Offline TA/TO', 'Groups', 'Direct', 'Corporate', 'Complementary', 'Aviation'], 1000),
        'adr': np.random.normal(loc=120, scale=30, size=1000).clip(50, 250)
    })
    chart = alt.Chart(df_adr_market).mark_boxplot(extent=1.5).encode(
        x=alt.X('market_segment', title='Market Segment'),
        y=alt.Y('adr', title='ADR'),
        tooltip=['market_segment', 'adr']
    ).properties(
        title='ADR by Market Segment'
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("This box plot shows the distribution of Average Daily Rate (ADR) across different market segments.")

    st.subheader("ADR by Customer Type")
    df_adr_customer = pd.DataFrame({
        'customer_type': np.random.choice(['Transient', 'Contract', 'Transient-Party', 'Group'], 1000),
        'adr': np.random.normal(loc=110, scale=40, size=1000).clip(40, 300)
    })
    chart = alt.Chart(df_adr_customer).mark_boxplot(extent=1.5).encode(
        x=alt.X('customer_type', title='Customer Type'),
        y=alt.Y('adr', title='ADR'),
        tooltip=['customer_type', 'adr']
    ).properties(
        title='ADR by Customer Type'
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("This chart provides insight into the typical ADR associated with each customer type.")

    st.subheader("ADR by Reserved Room Type")
    df_adr_room = pd.DataFrame({
        'reserved_room_type': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 1000),
        'adr': np.random.normal(loc=150, scale=50, size=1000).clip(50, 400)
    })
    chart = alt.Chart(df_adr_room).mark_boxplot(extent=1.5).encode(
        x=alt.X('reserved_room_type', title='Reserved Room Type'),
        y=alt.Y('adr', title='ADR'),
        tooltip=['reserved_room_type', 'adr']
    ).properties(
        title='ADR by Reserved Room Type'
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("This box plot compares the distribution of ADR for different reserved room types.")

    st.subheader("ADR Distribution by Hotel Type")
    df_adr_hotel = pd.DataFrame({
        'hotel': np.random.choice(['City Hotel', 'Resort Hotel'], 1000),
        'adr': np.random.normal(loc=120, scale=50, size=1000).clip(30, 350)
    })
    chart = alt.Chart(df_adr_hotel).mark_boxplot(extent=1.5).encode(
        x=alt.X('hotel', title='Hotel Type'),
        y=alt.Y('adr', title='ADR'),
        tooltip=['hotel', 'adr']
    ).properties(
        title='ADR Distribution by Hotel Type'
    )
    st.altair_chart(chart, use_container_width=True)
    st.markdown("This chart illustrates the difference in ADR distribution between City and Resort hotels.")


with tab4:
    st.title("Code Walkthrough")
    st.markdown("This section provides a detailed explanation of the Python code that powers this application, with snippets to guide you.")

    st.subheader("1. Importing Libraries and Initial Setup")
    st.markdown("The application begins by importing the necessary Python libraries. **`streamlit as st`** is the core library for creating the web application. **`joblib`** is used to load the pre-trained machine learning model. **`pandas`** and **`numpy`** are essential for data manipulation, and **`altair`** is used for generating the interactive charts in the 'Visualizations' tab. Finally, `st.set_page_config()` is used to set the page's title and layout to a wide format for better use of screen space.")
    st.code("""
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import altair as alt

# Set up the Streamlit page. This MUST be the first Streamlit command.
st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="wide")
    """)

    st.subheader("2. Defining Features and Loading the Model")
    st.markdown("The code explicitly defines the numerical and categorical features that the machine learning model expects as input. A dictionary, `categorical_values`, is used to list all possible categories for each feature, which is crucial for the one-hot encoding process. The pre-trained model is then loaded from a file named `hotel_cancellation_model.pkl`. A `try-except` block handles potential errors, such as the model file not being found.")
    st.code("""
# Define the numerical and categorical features
numerical_features = [...]
categorical_features = [...]

# A comprehensive list of possible values for each categorical feature
categorical_values = {...}

try:
    model = joblib.load('hotel_cancellation_model.pkl')
except FileNotFoundError:
    st.error("Error: The 'hotel_cancellation_model.pkl' file was not found...")
    st.stop()
    """)

    st.subheader("3. Creating the Tabbed Interface")
    st.markdown("The application's content is organized into multiple tabs using `st.tabs()`. This command returns a set of tab objects that you can use with Python's `with` statement to define the content for each tab.")
    st.code("""
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Summary", "Visualizations", "Code Explanation"])
    """)

    st.subheader("4. Collecting User Input and Preparing Data")
    st.markdown("Within the `Dashboard` tab, user inputs are collected using various Streamlit widgets like `st.slider` for numerical features and `st.selectbox` for categorical ones. The core logic then prepares this data for the model. It constructs a dictionary `full_input_data` and uses a loop to create a one-hot encoded representation for the categorical features. Finally, it converts this data into a pandas DataFrame, ensuring the column order matches the model's training data.")
    st.code("""
with st.sidebar:
    lead_time = st.slider("Lead Time (days)", ...)
    # ... other widgets

user_input = {...}
try:
    model_features = model.feature_names_in_
    full_input_data = {}
    for col in numerical_features:
        full_input_data[col] = user_input[col]
    
    for col in categorical_features:
        for val in categorical_values[col]:
            full_input_data[f'{col}_{val}'] = 1 if user_input[col] == val else 0
            
    input_df = pd.DataFrame([full_input_data], columns=model_features)
except Exception as e:
    st.error(f"Error preparing input data for the model: {e}")
    """)

    st.subheader("5. Making and Displaying Predictions")
    st.markdown("When the user clicks the 'Predict Cancellation' button, the script calls `model.predict()` and `model.predict_proba()` on the prepared DataFrame. The prediction result (0 or 1) and the probability score are then displayed to the user with clear success or error messages using `st.success()` or `st.error()`.")
    st.code("""
if st.button("Predict Cancellation"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"Prediction: This booking is **likely to be canceled**.")
        else:
            st.success(f"Prediction: This booking is **unlikely to be canceled**.")
    except Exception as e:
        st.error(f"An error occurred during prediction... Error: {e}")
    """)

    st.subheader("6. Model Training and Saving (`Hotel_Booking_Cancellation_Prediction.ipynb`)")
    st.markdown("This section explains the code from your Jupyter notebook that was used to create and save the machine learning model.")
    
    st.markdown("#### a. Data Loading and Feature Engineering")
    st.markdown("First, the data from the `hotel_bookings.csv` file is loaded into a pandas DataFrame. New features are then created to better represent the booking information. `total_guests` and `total_nights` are simple sums, while `is_family` is a binary feature. The `prev_cancel_rate` and `country_freq` are engineered features that capture the customer's history and the booking's country popularity, respectively. These features are often powerful predictors.")
    st.code("""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Load the dataset
df = pd.read_csv('hotel_bookings.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Feature Engineering
df['total_guests'] = df['adults'] + df['children'] + df['babies']
df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['is_family'] = df.apply(lambda row: 1 if row['children'] > 0 or row['babies'] > 0 else 0, axis=1)

# Calculate previous cancellation rate
df['prev_cancel_rate'] = df['previous_cancellations'] / (df['previous_cancellations'] + df['previous_bookings_not_canceled'])
df['prev_cancel_rate'].fillna(0, inplace=True)

# Calculate country frequency
country_counts = df['country'].value_counts(normalize=True)
df['country_freq'] = df['country'].map(country_counts)
    """)

    st.markdown("#### b. Data Splitting and Preprocessing")
    st.markdown("The data is split into a feature matrix (`X`) and a target variable (`y`, which is `is_canceled`). It's then further divided into training and testing sets to ensure the model's performance is evaluated on unseen data. One-hot encoding is applied to the categorical features to convert them into a numerical format that the machine learning model can understand.")
    st.code("""
# Define features and target
X = df.drop('is_canceled', axis=1)
y = df['is_canceled']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Identify numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

# Apply One-Hot Encoding
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded = pd.DataFrame(one_hot_encoder.fit_transform(X_train[categorical_cols]),
                               columns=one_hot_encoder.get_feature_names_out(categorical_cols),
                               index=X_train.index)
X_test_encoded = pd.DataFrame(one_hot_encoder.transform(X_test[categorical_cols]),
                              columns=one_hot_encoder.get_feature_names_out(categorical_cols),
                              index=X_test.index)

# Combine numerical and one-hot encoded features
X_train_num = X_train[numerical_cols].reset_index(drop=True)
X_test_num = X_test[numerical_cols].reset_index(drop=True)
X_train_final = pd.concat([X_train_num, X_train_encoded], axis=1)
X_test_final = pd.concat([X_test_num, X_test_encoded], axis=1)
    """)

    st.markdown("#### c. Model Training, Evaluation, and Saving")
    st.markdown("A `RandomForestClassifier` is initialized and then trained (`.fit()`) on the preprocessed training data. The model's performance is then evaluated on the test set using a `classification_report` and the `roc_auc_score`. Finally, the trained model object is saved to a file named `hotel_cancellation_model.pkl` using `joblib.dump()`. This is the file that the Streamlit application loads to make predictions.")
    st.code("""
# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train_final, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_final)
print("Classification Report:\\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1]))

# Save the trained model
joblib.dump(model, 'hotel_cancellation_model.pkl')
print("Model saved as hotel_cancellation_model.pkl")
    """)

st.markdown("---")
st.markdown("App by saumil_thukral")
st.markdown("https://github.com/saumil-thukral/Hotel-Booking-Cancellation-Prediction")

