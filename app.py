from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
app.secret_key = 'dev_secret_123'  # Replace with a strong secret in production

# Load model and transformers
model = joblib.load("models/linear_model.pkl")
qt_X = joblib.load("models/quantile_transformer_X.pkl")
qt_y = joblib.load("models/quantile_transformer_y.pkl")
feature_names = joblib.load("models/feature_names.pkl")
if not isinstance(feature_names, list):
    feature_names = list(feature_names)

numcols = ['squareMeters', 'basement', 'attic', 'garage']

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # --------- Extract form data as before ----------
        input_data = {
            "squareMeters": float(request.form["squareMeters"]),
            "basement": float(request.form["basement"]),
            "attic": float(request.form["attic"]),
            "garage": float(request.form["garage"]),
            "hasYard_1": int("hasYard_1" in request.form),
            "hasPool_1": int("hasPool_1" in request.form),
            "isNewBuilt_1": int("isNewBuilt_1" in request.form),
            "hasStormProtector_1": int("hasStormProtector_1" in request.form),
            "hasStorageRoom_1": int("hasStorageRoom_1" in request.form),
        }

        # Region
        region = request.form.get("region")
        input_data["nom_region_Nouvelle-Aquitaine"] = int(region == "Nouvelle-Aquitaine")
        input_data["nom_region_Occitanie"] = int(region == "Occitanie")

        # City part
        # City part (1 means no encoding)
        city_part = request.form.get("cityPartRange")
        for i in range(2, 11):
            key = f"cityPartRange_{i}"
            input_data[key] = int(city_part == str(i))


        # Age category
        age_cat = request.form.get("ageCat")
        input_data["ageCat_New"] = int(age_cat == "New")
        input_data["ageCat_Old"] = int(age_cat == "Old")
        # ageCat_Mid is NOT part of model input; ensure both others are zero if Mid is selected
        if age_cat == "Mid":
            input_data["ageCat_New"] = 0
            input_data["ageCat_Old"] = 0

        # 🔍 Log input
        print("\n📥 Raw form input data:")
        print(json.dumps(input_data, indent=2))

        X_input = pd.DataFrame([input_data])

        # Align columns
        for col in feature_names:
            if col not in X_input.columns:
                X_input[col] = 0
        X_input = X_input[feature_names]

        # Debug shape
        print(f"\n✅ Aligned input columns: {list(X_input.columns)}")

        # Split numeric / categorical
        X_input_num = X_input[numcols]
        X_input_cat = X_input.drop(columns=numcols)

        # Transform + predict
        X_input_num_trans = qt_X.transform(X_input_num)
        X_trans = np.hstack([X_input_num_trans, X_input_cat.values])
        y_pred_trans = model.predict(X_trans)
        y_pred = qt_y.inverse_transform(y_pred_trans.reshape(-1, 1))[0, 0]
        prediction = round(y_pred, 2)

        # ✅ NEW: Store in session and redirect
        session["prediction"] = prediction  # Store in session
        return redirect(url_for("predict"))  # Redirect to avoid POST on refresh

        
    # Only display on GET (after redirect)
    prediction = session.pop("prediction", None)
    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
