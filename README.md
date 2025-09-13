# Hotel Booking Cancellation Prediction  

## 📌 Project Overview  
Hotel booking cancellations can significantly affect revenue and resource management in the hospitality industry.  
This project predicts whether a hotel booking will be **canceled or not** using **machine learning models** trained on booking data.  

The project is built with:  
- **Python** for data processing & model building  
- **Scikit-learn** for machine learning  
- **Streamlit** for web app deployment  
- **GitHub** for version control  

---

## 🚀 Features  
- Data preprocessing and cleaning  
- Exploratory Data Analysis (EDA) with visualizations  
- Feature engineering for better model performance  
- Multiple machine learning models tested (Logistic Regression, Random Forest, XGBoost, etc.)  
- Final model saved as `.pkl` file  
- Interactive **Streamlit app** for predictions  

---

## 🛠️ Tech Stack  
- **Programming Language**: Python  
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit  
- **Version Control**: Git + GitHub  
- **Deployment**: Streamlit  

---

## 📂 Project Structure  
Hotel-Booking-Cancellation-Prediction/
│── data/ # Dataset files (ignored in .gitignore)
│── notebooks/ # Jupyter notebooks (EDA, training)
│── hotel_cancellation_model.pkl # Trained ML model (stored externally, link below)
│── app.py # Streamlit app
│── requirements.txt # Python dependencies
│── .gitignore # Ignored files
│── README.md # Project documentation

---

## 📊 Dataset  
The dataset contains booking-related features such as:  
- **Hotel type**  
- **Lead time**  
- **Arrival date**  
- **Number of guests**  
- **Meal plan**  
- **Deposit type**  
- **Special requests**  
- **Cancellation status (target)**  

---

## 🧠 Model  
The final trained model is a **Random Forest Classifier**.  
- Achieved good accuracy on test data  
- Handles both categorical and numerical features  

---

## 📦 Installation  

Clone the repo:  
```bash
git clone https://github.com/saumil-thukral/Hotel-Booking-Cancellation-Prediction.git
cd Hotel-Booking-Cancellation-Prediction
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

pip install -r requirements.txt
Running the Streamlit App
streamlit run app.py
## Model File
The trained model (`hotel_cancellation_model.pkl`) is too large to store on GitHub.  
You can download it from here: [Download Model](https://drive.google.com/file/d/1f34OB6Uk9vB6ilv8VWdmLlQYY7_1buZF/view?usp=drive_link).
