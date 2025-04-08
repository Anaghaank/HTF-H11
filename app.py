from flask import Flask, render_template, request
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
import base64

app = Flask(__name__)

# Load model and data
model = joblib.load("hospital_admission_model.pkl")
df = pd.read_csv("sdoh_karnataka_dataset.csv")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.dropna(subset=['HospitalAdmissions'])

explainer = shap.TreeExplainer(model)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    chart_data = None

    if request.method == "POST":
        try:
            user_pincode = int(request.form["pincode"])
            df_pin = df[df["Pincode"] == user_pincode]

            if not df_pin.empty:
                latest_row = df_pin.sort_values("Date").iloc[-1:]
                X_input = latest_row.drop(columns=["Date", "HospitalAdmissions"])
                prediction = model.predict(X_input)[0]
                shap_values = explainer.shap_values(X_input)[0]

                impact_df = pd.DataFrame({
                    "Feature": X_input.columns,
                    "Impact": abs(shap_values)
                }).sort_values(by="Impact", ascending=False)

                # Generate pie chart
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.pie(impact_df["Impact"], labels=impact_df["Feature"], autopct="%1.1f%%", startangle=140)
                ax.set_title(f"SHAP Impact (Predicted: {prediction:.2f})")

                buf = BytesIO()
                plt.tight_layout()
                plt.savefig(buf, format="png")
                plt.close(fig)
                buf.seek(0)
                chart_data = base64.b64encode(buf.getvalue()).decode("utf-8")
            else:
                prediction = "Pincode not found."

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, chart_data=chart_data)

if __name__ == "__main__":
    app.run(debug=True)
