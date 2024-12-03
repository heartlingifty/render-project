import pandas as pd
import numpy as np
import plotly.graph_objects as go
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from google.oauth2.service_account import Credentials
import gspread

# Flask app initialization
app = Flask(__name__)

# Google Sheets setup
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive"
]
creds = Credentials.from_service_account_file("ds-project-440913-993d0814d106.json", scopes=scope)
client = gspread.authorize(creds)

# Load data from Google Sheets
sheet_url = "https://docs.google.com/spreadsheets/d/109WSI59_V1LEF84uIC_3Yt3DBzR4h0eooBOOBeNehf0/edit?usp=sharing"
sheet = client.open_by_url(sheet_url)
worksheet = sheet.worksheet("table_4")
table_4 = pd.DataFrame(worksheet.get_all_records())

# Debug: Print original column names
print("Original column names:", table_4.columns.tolist())

# Preprocessing
# Standardize column names: strip, lowercase, replace newline and multiple spaces
table_4.columns = (
    table_4.columns.str.strip()
    .str.lower()
    .str.replace('\n', ' ', regex=False)
    .str.replace(r'\s+', ' ', regex=True)
    .str.replace('(', '')
    .str.replace(')', '')
    .str.replace(',', '')
)

# Debug: Print cleaned column names
print("Cleaned column names:", table_4.columns.tolist())

# Rename columns to match expected names
rename_dict = {
    "industry": "Industry",
    "business number": "Business Number",
    "employment": "Employment",
    "turnover  millions excluding vat": "Turnover (millions, excluding VAT)"
}
table_4.rename(columns=rename_dict, inplace=True)

# Debug: Print column names after renaming
print("Column names after renaming:", table_4.columns.tolist())

# Check if the target column exists
target_column = "Turnover (millions, excluding VAT)"
if target_column not in table_4.columns:
    print(f"Error: '{target_column}' column not found.")
    print("Available columns:", table_4.columns.tolist())
    exit(1)


# Ensure the target column is numeric and handle non-numeric entries
table_4[target_column] = pd.to_numeric(table_4[target_column], errors='coerce')

# Fill NaN values in the target column with the mean
mean_turnover = table_4[target_column].mean()
table_4[target_column].fillna(mean_turnover, inplace=True)

# Debug: Verify data types and first few rows
print("Data types after conversion:")
print(table_4.dtypes)
print("\nFirst few rows after conversion and filling NaN:")
print(table_4.head())

# Prepare features and target
X = table_4[["Business Number", "Employment"]]
y = table_4[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Debug: Print evaluation metrics
print(f"Model Performance: MAE={mae}, MSE={mse}, R2={r2}")

# Flask routes
@app.route("/")
def home():
    industries = table_4["Industry"].unique()
    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "R2": round(r2, 2)
    }
    return render_template("index.html", metrics=metrics, industries=industries)

@app.route("/predict", methods=["POST"])
def predict():
    selected_industry = request.form.get("industry")
    industry_data = table_4[table_4["Industry"] == selected_industry]

    if industry_data.empty:
        return jsonify({"error": "Invalid industry selected."})

    # Predict for the next 10 years
    predictions = []
    current_business = industry_data["Business Number"].values[0]
    current_employment = industry_data["Employment"].values[0]

    for year in range(1, 11):
        future_business = current_business * (1 + 0.015 * year)
        future_employment = current_employment * (1 + 0.015 * year)
        future_turnover = model.predict([[future_business, future_employment]])[0]
        predictions.append({
            "Year": 2024 + year,
            "Business Number": round(future_business, 2),
            "Employment": round(future_employment, 2),
            "Turnover (millions)": round(future_turnover, 2)
        })

    predictions_df = pd.DataFrame(predictions)

    # Visualization using Plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=predictions_df["Year"], y=predictions_df["Business Number"],
        name="Business Number", text=predictions_df["Business Number"], textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=predictions_df["Year"], y=predictions_df["Employment"],
        name="Employment", text=predictions_df["Employment"], textposition="auto"
    ))
    fig.add_trace(go.Bar(
        x=predictions_df["Year"], y=predictions_df["Turnover (millions)"],
        name="Turnover (millions)", text=predictions_df["Turnover (millions)"], textposition="auto"
    ))
    fig.update_layout(
        title=f"Predictions for {selected_industry}",
        xaxis_title="Year",
        yaxis_title="Values",
        barmode="group",
        template="plotly_white"
    )
    graph_html = fig.to_html(full_html=False)

    return render_template(
        "predictions.html",
        graph_html=graph_html,
        predictions=predictions_df.to_dict(orient="records")
    )

if __name__ == "__main__":
    app.run(debug=True)
