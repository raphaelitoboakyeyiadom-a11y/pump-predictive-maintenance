🛢️ Pump Predictive Maintenance – AI Fault Detection Dashboard

A complete AI-powered predictive maintenance system for centrifugal pump equipment.
This application uses machine learning and sensor data to predict pump failures and provide actionable maintenance recommendations.

🚀 Features

Real-time sensor input via sliders:

Temperature (°C)

Vibration (mm/s)

Discharge Pressure (bar)

Flow Rate (m³/h)

Motor Current (A)

Machine learning prediction using Random Forest

🟢 Normal Operation

🟡 Early Warning

🔴 Bearing Fault Likely

97.5% model accuracy

Confusion matrix visualization

Feature importance chart

Sensor distribution plots

Technician advisory messages

Downtime cost savings calculator

🧠 Machine Learning Workflow

Synthetic pump dataset generation

Train/test split (75/25)

Random Forest classifier training

Evaluation (accuracy + confusion matrix)

Feature importance analysis

Deployment with Streamlit

🛠️ Technologies Used

Python

Streamlit

Pandas

NumPy

Scikit-learn

Matplotlib

📈 Business Impact

The dashboard estimates downtime cost savings:

Example: Avoiding 4 hours of pump failure at $10,000 per hour
👉 Savings: $40,000

📂 How to Run

Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

🌍 Future Enhancements

Integrate real pump sensor data (IoT)

Add Remaining Useful Life (RUL) prediction

Cloud deployment (AWS / Azure)

👤 Author

Raphael Boakye-Yiadom
Mechanical Engineering + AI/ML Engineer
MIT License included.

## 📸 Dashboard Screenshots

![Dashboard View 1](./Screenshot_214619.png)

![Dashboard View 2](./Screenshot_214758.png)

![Dashboard View 3](./Screenshot_215011.png)

![Dashboard View 4](./Screenshot_215043.png)

![Dashboard View 5](./Screenshot_215220.png)

![Dashboard View 6](./pump_prediction_pic_1.png)










