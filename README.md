# 🏠 Paris Housing Price Prediction Web App

This project is a Flask-based web application that predicts housing prices in Paris based on various input features. It uses a Linear Regression model trained on the Paris Housing dataset and includes data preprocessing via `QuantileTransformer`.

---

## 🚀 Features

- User-friendly web interface to input housing features
- Real-time prediction of price using trained Linear Regression model
- Quantile transformation applied to inputs and inverse-transformed outputs
- Deployable via Azure App Service or locally

---

## 📂 Project Structure

```bash
paris-housing-price-app/
│
├── app.py                       # Main Flask application
├── models/                      # Saved model and transformer files (.pkl)
│   ├── linear_model.pkl
│   ├── quantile_transformer_X.pkl
│   ├── quantile_transformer_y.pkl
│   └── feature_names.pkl
│
├── templates/
│   └── index.html               # Web page for input form and results
├── static/
│   └── style.css                # (Optional) CSS styling
│
├── requirements.txt             # Python dependencies
├── .gitignore
├── README.md
└── LICENSE