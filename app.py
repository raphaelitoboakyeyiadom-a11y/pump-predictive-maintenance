import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ---------- DATA GENERATION ----------
def generate_pump_data(n_samples=800, random_state=42):
    np.random.seed(random_state)

    n_normal = int(n_samples * 0.5)
    n_warning = int(n_samples * 0.25)
    n_fault = n_samples - n_normal - n_warning

    temp_normal = np.random.normal(70, 3, n_normal)
    vib_normal = np.random.normal(2.0, 0.4, n_normal)
    press_normal = np.random.normal(6, 0.5, n_normal)
    flow_normal = np.random.normal(50, 5, n_normal)
    curr_normal = np.random.normal(18, 2, n_normal)
    label_normal = np.zeros(n_normal, dtype=int)

    temp_warn = np.random.normal(80, 4, n_warning)
    vib_warn = np.random.normal(3.5, 0.5, n_warning)
    press_warn = np.random.normal(6.5, 0.6, n_warning)
    flow_warn = np.random.normal(48, 5, n_warning)
    curr_warn = np.random.normal(20, 2, n_warning)
    label_warn = np.ones(n_warning, dtype=int)

    temp_fault = np.random.normal(90, 5, n_fault)
    vib_fault = np.random.normal(5.0, 0.6, n_fault)
    press_fault = np.random.normal(7.0, 0.7, n_fault)
    flow_fault = np.random.normal(45, 6, n_fault)
    curr_fault = np.random.normal(23, 3, n_fault)
    label_fault = np.full(n_fault, 2, dtype=int)

    temperature = np.concatenate([temp_normal, temp_warn, temp_fault])
    vibration = np.concatenate([vib_normal, vib_warn, vib_fault])
    pressure = np.concatenate([press_normal, press_warn, press_fault])
    flow_rate = np.concatenate([flow_normal, flow_warn, flow_fault])
    motor_current = np.concatenate([curr_normal, curr_warn, curr_fault])
    label = np.concatenate([label_normal, label_warn, label_fault])

    df = pd.DataFrame({
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "flow_rate": flow_rate,
        "motor_current": motor_current,
        "label": label
    })

    return df


df = generate_pump_data()

X = df[["temperature", "vibration", "pressure", "flow_rate", "motor_current"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# feature importance for later plotting
feature_names = X.columns.tolist()
importances = model.feature_importances_
fi_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
    "importance", ascending=False
)

# ---------- STATUS & COST LOGIC ----------
def get_status_and_message(label: int):
    if label == 0:
        status = "GREEN"
        title = "🟢 MACHINE HEALTHY"
        message = (
            "The pump is operating within normal ranges.\n"
            "- No immediate maintenance is required.\n"
            "- Continue routine monitoring."
        )
    elif label == 1:
        status = "YELLOW"
        title = "🟡 WARNING – EARLY ABNORMALITY"
        message = (
            "Early signs of abnormal behaviour detected.\n"
            "- Monitor vibration and temperature more frequently.\n"
            "- Plan maintenance in the next suitable shutdown window."
        )
    elif label == 2:
        status = "RED"
        title = "🔴 CRITICAL – BEARING FAULT LIKELY"
        message = (
            "Patterns are consistent with a bearing-related fault.\n"
            "- Please change the bearings as soon as possible.\n"
            "- Inspect lubrication and alignment.\n"
            "- Avoid running at full load until inspected."
        )
    else:
        status = "UNKNOWN"
        title = "⚪ STATUS UNKNOWN"
        message = "Model output is unclear. Please recheck sensor values."
    return status, title, message


def estimate_cost_saving(status: str, hourly_cost: float = 10000.0):
    if status == "RED":
        hours_avoided = 4
    elif status == "YELLOW":
        hours_avoided = 1
    else:
        hours_avoided = 0

    saved = hours_avoided * hourly_cost
    return hours_avoided, saved

# ---------- STREAMLIT APP ----------
st.set_page_config(page_title="Pump Predictive Maintenance", layout="wide")

st.title("🛢️ Pump Predictive Maintenance – Professional Dashboard")
st.write(
    "Estimate the health of a centrifugal pump based on sensor readings and "
    "view maintenance recommendations for technicians."
)

st.sidebar.header("🔧 Input Sensor Readings")

temp_input = st.sidebar.slider("Temperature (°C)", 50.0, 110.0, 75.0, 0.5)
vib_input = st.sidebar.slider("Vibration (mm/s)", 0.5, 8.0, 3.0, 0.1)
press_input = st.sidebar.slider("Discharge Pressure (bar)", 3.0, 10.0, 6.5, 0.1)
flow_input = st.sidebar.slider("Flow Rate (m³/h)", 20.0, 80.0, 50.0, 1.0)
curr_input = st.sidebar.slider("Motor Current (A)", 5.0, 35.0, 20.0, 0.5)

st.sidebar.write("---")
hourly_cost = st.sidebar.number_input(
    "Estimated downtime cost per hour (USD)",
    min_value=1000.0,
    max_value=500000.0,
    value=10000.0,
    step=1000.0
)

run_pred = st.sidebar.button("Run Prediction")

if run_pred:
    input_data = np.array([[temp_input, vib_input, press_input, flow_input, curr_input]])
    pred_label = int(model.predict(input_data)[0])

    status, status_title, recommendation = get_status_and_message(pred_label)
    hours_avoided, cost_saved = estimate_cost_saving(status, hourly_cost)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        if status == "GREEN":
            st.success(status_title)
        elif status == "YELLOW":
            st.warning(status_title)
        elif status == "RED":
            st.error(status_title)
        else:
            st.info(status_title)

        st.write("### 📋 Recommendation for Technician")
        st.write(recommendation)

        st.write("### 💰 Potential Cost Impact")
        if hours_avoided > 0:
            st.write(
                f"- Estimated unplanned downtime avoided: **~{hours_avoided} hours**\n"
                f"- Approximate cost saving: **${cost_saved:,.0f} USD** "
                f"(at ${hourly_cost:,.0f}/hour)"
            )
        else:
            st.write(
                "No immediate downtime saving estimated for this condition.\n"
                "Maintaining normal operation still protects production."
            )

    with col2:
        st.write("### 📊 Current Sensor Snapshot")
        sensor_df = pd.DataFrame(
            {
                "Sensor": [
                    "Temperature (°C)",
                    "Vibration (mm/s)",
                    "Pressure (bar)",
                    "Flow (m³/h)",
                    "Motor Current (A)",
                ],
                "Value": [temp_input, vib_input, press_input, flow_input, curr_input],
            }
        )
        st.table(sensor_df)

        st.write("### 📈 Model Performance")
        st.write(f"**Validation accuracy:** {acc:.2%}")

        # Confusion matrix
        fig, ax = plt.subplots()
        im = ax.imshow(cm)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Normal", "Warning", "Fault"])
        ax.set_yticklabels(["Normal", "Warning", "Fault"])

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha="center", va="center")

        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

        # Feature importance chart
        st.write("### 🔍 Feature Importance")
        fig2, ax2 = plt.subplots()
        ax2.barh(fi_df["feature"], fi_df["importance"])
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        ax2.invert_yaxis()
        st.pyplot(fig2)

        # Sensor distributions vs current values (temperature & vibration)
        st.write("### 📉 Sensor Value vs Typical Range")

        # Temperature distribution
        fig3, ax3 = plt.subplots()
        ax3.hist(df["temperature"], bins=30)
        ax3.axvline(temp_input)
        ax3.set_xlabel("Temperature (°C)")
        ax3.set_ylabel("Count")
        ax3.set_title("Temperature Distribution")
        st.pyplot(fig3)

        # Vibration distribution
        fig4, ax4 = plt.subplots()
        ax4.hist(df["vibration"], bins=30)
        ax4.axvline(vib_input)
        ax4.set_xlabel("Vibration (mm/s)")
        ax4.set_ylabel("Count")
        ax4.set_title("Vibration Distribution")
        st.pyplot(fig4)

else:
    st.info("Adjust sensor values on the left and click **Run Prediction** to see the pump status.")
