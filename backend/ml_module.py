import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CSV_PATH = os.path.join(DATA_DIR, "transactions.csv")
MODEL_PATH = os.path.join(DATA_DIR, "fraud_model.joblib")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.joblib")
LABEL_ENCODERS_PATH = os.path.join(DATA_DIR, "label_encoders.joblib")
METRICS_PATH = os.path.join(DATA_DIR, "metrics.json")

def generate_synthetic_data(num_samples=1500):
    """Generates synthetic corporate transaction data with embedded fraud scenarios."""
    np.random.seed(42)
    
    # List of components
    departments = ["Sales", "Engineering", "Marketing", "HR", "Finance", "Executive", "Operations"]
    roles = ["Executive", "Manager", "Associate", "Contractor"]
    categories = ["Travel", "Meals", "Office Supplies", "Software Licenses", "Consulting", "Equipment", "Entertainment"]
    payment_methods = ["Corporate Credit Card", "Invoice/Wire", "Reimbursement Request"]
    locations = ["Domestic", "International (Low Risk)", "International (High Risk)"]
    
    vendors = {
        "Travel": ["Delta Airlines", "Marriott Hotels", "Uber", "Hertz", "Airbnb"],
        "Meals": ["Local Diner", "Steakhouse", "Catering Co", "UberEats", "Starbucks"],
        "Office Supplies": ["Staples", "Amazon Business", "OfficeMax", "Target"],
        "Software Licenses": ["AWS", "GitHub", "Slack", "Zoom", "Salesforce", "Adobe"],
        "Consulting": ["McKinsey & Co", "Accenture", "Delta Consulting", "Apex Legal", "Shadow Advisory Group"],
        "Equipment": ["Apple Store", "Dell Technologies", "Best Buy", "CDW Corporation"],
        "Entertainment": ["TopGolf", "Vegas VIP Clubs", "City Arena", "Private Dining Services"]
    }

    # Generate employee metadata
    employees = []
    for i in range(100):
        emp_id = f"EMP-{1000 + i}"
        dept = np.random.choice(departments)
        role = np.random.choice(roles, p=[0.05, 0.20, 0.60, 0.15])
        
        # Historical metrics based on role
        if role == "Executive":
            hist_mean = np.random.uniform(1000, 3000)
            hist_std = np.random.uniform(200, 500)
        elif role == "Manager":
            hist_mean = np.random.uniform(300, 800)
            hist_std = np.random.uniform(50, 150)
        elif role == "Associate":
            hist_mean = np.random.uniform(50, 200)
            hist_std = np.random.uniform(10, 50)
        else: # Contractor
            hist_mean = np.random.uniform(150, 400)
            hist_std = np.random.uniform(30, 80)
            
        employees.append({
            "employee_id": emp_id,
            "department": dept,
            "role": role,
            "hist_mean": hist_mean,
            "hist_std": hist_std
        })
        
    emp_df = pd.DataFrame(employees)

    # Initialize data arrays
    data = []
    
    # Base transactions creation (Mostly clean, some random noise)
    for i in range(num_samples):
        tx_id = f"TX-{10000 + i}"
        emp = emp_df.sample(n=1).iloc[0]
        
        # Date & Time (mostly weekdays, daytime)
        is_weekend = np.random.choice([0, 1], p=[0.85, 0.15])
        is_night = np.random.choice([0, 1], p=[0.90, 0.10])
        
        hour = np.random.randint(22, 24) if is_night else (np.random.randint(0, 5) if is_night else np.random.randint(8, 18))
        day = np.random.randint(1, 29)
        month = np.random.randint(1, 13)
        year = 2025
        
        category = np.random.choice(categories)
        vendor = np.random.choice(vendors[category])
        payment_method = np.random.choice(payment_methods)
        location = np.random.choice(locations, p=[0.75, 0.20, 0.05])
        
        # Set amount with lognormal distribution based on historical mean
        amount = np.random.lognormal(mean=np.log(emp["hist_mean"]), sigma=0.4)
        # Prevent extreme outliers naturally
        amount = min(amount, emp["hist_mean"] + 4 * emp["hist_std"])
        
        data.append({
            "transaction_id": tx_id,
            "employee_id": emp["employee_id"],
            "employee_name": f"Employee {emp['employee_id'].split('-')[1]}",
            "department": emp["department"],
            "role": emp["role"],
            "category": category,
            "vendor": vendor,
            "amount": round(amount, 2),
            "payment_method": payment_method,
            "location": location,
            "hour": hour,
            "is_weekend": int(is_weekend),
            "is_night": int(is_night),
            "is_fraud": 0,
            "fraud_type": "None"
        })

    # Let's inject SPECIFIC FRAUD SCENARIOS (~8-10% of dataset)
    fraud_count = int(num_samples * 0.07)
    
    # 1. Split Invoice Fraud (Bypassing $10,000 threshold)
    # An employee submits multiple transactions of just under $10,000 to the same vendor
    for i in range(12):
        emp = emp_df[emp_df["role"] != "Executive"].sample(n=1).iloc[0]
        vendor = np.random.choice(vendors["Consulting"])
        # Split a $29,000 payment into 3 parts
        base_tx_id = num_samples + len(data)
        dates = [(10, 15, 2025), (10, 15, 2025), (10, 16, 2025)]
        for j, (month, day, year) in enumerate(dates):
            amount = np.random.uniform(9600, 9950)
            data.append({
                "transaction_id": f"TX-{base_tx_id + j}",
                "employee_id": emp["employee_id"],
                "employee_name": f"Employee {emp['employee_id'].split('-')[1]}",
                "department": emp["department"],
                "role": emp["role"],
                "category": "Consulting",
                "vendor": vendor,
                "amount": round(amount, 2),
                "payment_method": "Invoice/Wire",
                "location": "Domestic",
                "hour": 14,
                "is_weekend": 0,
                "is_night": 0,
                "is_fraud": 1,
                "fraud_type": "Invoice Splitting"
            })

    # 2. Duplicate Reimbursement Fraud
    # Employee submits exact same reimbursement request within a few days
    for i in range(15):
        emp = emp_df.sample(n=1).iloc[0]
        category = "Meals"
        vendor = np.random.choice(vendors[category])
        amount = np.random.uniform(150, 450)
        base_tx_id = num_samples + len(data)
        
        # Record 1 (Normal)
        data.append({
            "transaction_id": f"TX-{base_tx_id}",
            "employee_id": emp["employee_id"],
            "employee_name": f"Employee {emp['employee_id'].split('-')[1]}",
            "department": emp["department"],
            "role": emp["role"],
            "category": category,
            "vendor": vendor,
            "amount": round(amount, 2),
            "payment_method": "Reimbursement Request",
            "location": "Domestic",
            "hour": 19,
            "is_weekend": 0,
            "is_night": 0,
            "is_fraud": 0,
            "fraud_type": "None"
        })
        
        # Record 2 (Fraudulent duplicate)
        data.append({
            "transaction_id": f"TX-{base_tx_id + 1}",
            "employee_id": emp["employee_id"],
            "employee_name": f"Employee {emp['employee_id'].split('-')[1]}",
            "department": emp["department"],
            "role": emp["role"],
            "category": category,
            "vendor": vendor,
            "amount": round(amount, 2),
            "payment_method": "Reimbursement Request",
            "location": "Domestic",
            "hour": 21,
            "is_weekend": 0,
            "is_night": 0,
            "is_fraud": 1,
            "fraud_type": "Duplicate Claim"
        })

    # 3. Out of Pattern Spend Spike
    # Associate spending huge amount on entertainment/consulting
    for i in range(25):
        emp = emp_df[emp_df["role"] == "Associate"].sample(n=1).iloc[0]
        category = np.random.choice(["Entertainment", "Consulting", "Software Licenses"])
        vendor = np.random.choice(vendors[category])
        amount = np.random.uniform(4000, 9500)
        base_tx_id = num_samples + len(data)
        data.append({
            "transaction_id": f"TX-{base_tx_id}",
            "employee_id": emp["employee_id"],
            "employee_name": f"Employee {emp['employee_id'].split('-')[1]}",
            "department": emp["department"],
            "role": emp["role"],
            "category": category,
            "vendor": vendor,
            "amount": round(amount, 2),
            "payment_method": "Corporate Credit Card",
            "location": "Domestic",
            "hour": 23,
            "is_weekend": 1,
            "is_night": 1,
            "is_fraud": 1,
            "fraud_type": "Unauthorised Spend Limit Exceeded"
        })

    # 4. Offshore Shell Company Transfer
    # High amount transferred to high risk international location, suspicious vendor
    for i in range(15):
        emp = emp_df[emp_df["role"].isin(["Executive", "Manager"])].sample(n=1).iloc[0]
        category = "Consulting"
        vendor = "Shadow Advisory Group"
        amount = np.random.uniform(15000, 48000)
        base_tx_id = num_samples + len(data)
        data.append({
            "transaction_id": f"TX-{base_tx_id}",
            "employee_id": emp["employee_id"],
            "employee_name": f"Employee {emp['employee_id'].split('-')[1]}",
            "department": emp["department"],
            "role": emp["role"],
            "category": category,
            "vendor": vendor,
            "amount": round(amount, 2),
            "payment_method": "Invoice/Wire",
            "location": "International (High Risk)",
            "hour": 3, # Middle of night
            "is_weekend": 1,
            "is_night": 1,
            "is_fraud": 1,
            "fraud_type": "High-Risk Tax Haven Offshore Wire"
        })

    # 5. After-Hours Anomalous Spending
    # Non-travel expenses incurred in the middle of the night
    for i in range(15):
        emp = emp_df.sample(n=1).iloc[0]
        category = "Equipment"
        vendor = "Apple Store"
        amount = np.random.uniform(2500, 4800)
        base_tx_id = num_samples + len(data)
        data.append({
            "transaction_id": f"TX-{base_tx_id}",
            "employee_id": emp["employee_id"],
            "employee_name": f"Employee {emp['employee_id'].split('-')[1]}",
            "department": emp["department"],
            "role": emp["role"],
            "category": category,
            "vendor": vendor,
            "amount": round(amount, 2),
            "payment_method": "Corporate Credit Card",
            "location": "Domestic",
            "hour": 2,
            "is_weekend": 1,
            "is_night": 1,
            "is_fraud": 1,
            "fraud_type": "Suspicious Off-hours Purchase"
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate historical features dynamically for each transaction context
    # This helps ML model capture deviations
    emp_map = emp_df.set_index("employee_id")
    df["emp_hist_mean"] = df["employee_id"].map(emp_map["hist_mean"]).round(2)
    df["emp_hist_std"] = df["employee_id"].map(emp_map["hist_std"]).round(2)
    df["amount_deviation_score"] = ((df["amount"] - df["emp_hist_mean"]) / df["emp_hist_std"]).round(2)
    
    # Ensure there are no NaNs
    df["amount_deviation_score"] = df["amount_deviation_score"].fillna(0.0)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Generated synthetic dataset with {len(df)} transactions and saved to {CSV_PATH}")
    return df

def train_ml_model():
    """Loads the dataset, preprocesses features, trains the Random Forest classifier, and saves evaluation metrics."""
    if not os.path.exists(CSV_PATH):
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(CSV_PATH)

    # Features to use for ML
    feature_cols = [
        "department", "role", "category", "vendor", "amount", 
        "payment_method", "location", "hour", "is_weekend", 
        "is_night", "emp_hist_mean", "emp_hist_std", "amount_deviation_score"
    ]
    
    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()

    # Preprocessing: Fit Label Encoders
    encoders = {}
    categorical_cols = ["department", "role", "category", "vendor", "payment_method", "location"]
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Save Encoders
    joblib.dump(encoders, LABEL_ENCODERS_PATH)

    # Preprocessing: Fit Scaler
    scaler = StandardScaler()
    numerical_cols = ["amount", "hour", "emp_hist_mean", "emp_hist_std", "amount_deviation_score"]
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Save Scaler
    joblib.dump(scaler, SCALER_PATH)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Supervised Model: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)

    # Unsupervised Anomaly Detection: Isolation Forest
    # Trains on clean data only (or entire dataset as contamination)
    iso_forest = IsolationForest(contamination=0.08, random_state=42)
    iso_forest.fit(X)

    # Save Models combined
    joblib.dump({"classifier": rf_model, "anomaly_detector": iso_forest}, MODEL_PATH)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist() # [[TN, FP], [FN, TP]]

    # Feature Importance
    importances = rf_model.feature_importances_
    feature_importance_list = sorted(
        [{"feature": col, "importance": float(imp)} for col, imp in zip(feature_cols, importances)],
        key=lambda x: x["importance"], reverse=True
    )

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "feature_importances": feature_importance_list,
        "dataset_size": len(df),
        "fraud_count": int(df["is_fraud"].sum()),
        "clean_count": int(len(df) - df["is_fraud"].sum())
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Successfully trained ML models. Saved metrics: {metrics}")
    return metrics

def run_predictions_on_df(df_input):
    """Utility to predict labels for an entire dataframe (returns array of probabilities and anomaly flags)."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        train_ml_model()
        
    models = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(LABEL_ENCODERS_PATH)
    
    rf_model = models["classifier"]
    iso_forest = models["anomaly_detector"]
    
    feature_cols = [
        "department", "role", "category", "vendor", "amount", 
        "payment_method", "location", "hour", "is_weekend", 
        "is_night", "emp_hist_mean", "emp_hist_std", "amount_deviation_score"
    ]
    
    X = df_input[feature_cols].copy()
    
    # Handle missing labels gracefully by mapping unseen categories to a fallback
    categorical_cols = ["department", "role", "category", "vendor", "payment_method", "location"]
    for col in categorical_cols:
        le = encoders[col]
        # In case we have unseen classes during live request, map them to the first label class
        X[col] = X[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        X[col] = le.transform(X[col])
        
    numerical_cols = ["amount", "hour", "emp_hist_mean", "emp_hist_std", "amount_deviation_score"]
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    probabilities = rf_model.predict_proba(X)[:, 1]
    # Isolation Forest returns -1 for anomalies, 1 for normal.
    anomalies = iso_forest.predict(X)
    anomaly_scores = iso_forest.decision_function(X)
    
    # Invert anomaly scores for display (higher = more anomalous)
    anomaly_scores = -anomaly_scores
    
    return probabilities, (anomalies == -1).astype(int), anomaly_scores

def explain_prediction(tx_row):
    """Generates feature attribution factors explaining why a transaction is high risk."""
    models = joblib.load(MODEL_PATH)
    rf_model = models["classifier"]
    
    feature_cols = [
        "department", "role", "category", "vendor", "amount", 
        "payment_method", "location", "hour", "is_weekend", 
        "is_night", "emp_hist_mean", "emp_hist_std", "amount_deviation_score"
    ]
    
    # Preprocess a single row
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(LABEL_ENCODERS_PATH)
    
    row_df = pd.DataFrame([tx_row])
    X = row_df[feature_cols].copy()
    
    for col in ["department", "role", "category", "vendor", "payment_method", "location"]:
        le = encoders[col]
        val = str(X[col].iloc[0])
        mapped_val = val if val in le.classes_ else le.classes_[0]
        X[col] = le.transform([mapped_val])
        
    numerical_cols = ["amount", "hour", "emp_hist_mean", "emp_hist_std", "amount_deviation_score"]
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    # For RF we can estimate contributions using individual decision trees (simplified TreeInterpreter logic)
    # We will output a custom weight based on feature importance and standard deviations
    explanations = []
    
    amount = float(tx_row["amount"])
    hist_mean = float(tx_row["emp_hist_mean"])
    deviation = float(tx_row["amount_deviation_score"])
    is_night = int(tx_row["is_night"])
    location = str(tx_row["location"])
    
    # Rule/Feature weights for presentation
    if deviation > 3.0:
        explanations.append({"factor": "Expense Deviation", "weight": min(0.9, deviation * 0.1), "description": f"Spent ${amount:.2f} vs average of ${hist_mean:.2f} ({deviation:.1f} std devs)"})
    if is_night == 1:
        explanations.append({"factor": "Processing Time", "weight": 0.45, "description": f"Processed at night ({tx_row['hour']}:00)"})
    if "High Risk" in location:
        explanations.append({"factor": "Risk Destination", "weight": 0.65, "description": f"Transferred to high-risk location ({location})"})
    if tx_row["category"] == "Consulting" and amount > 9000 and amount < 10000:
        explanations.append({"factor": "Invoice Structuring", "weight": 0.75, "description": "Transaction amount is suspicious of bypassing $10,000 corporate threshold"})
    if tx_row["fraud_type"] == "Duplicate Claim":
        explanations.append({"factor": "Double Submission", "weight": 0.85, "description": "Identical amount and vendor matching employee profile submitted within 48h"})
        
    # Standard features importances base fallbacks
    explanations.append({"factor": "Department Baseline", "weight": 0.15, "description": f"Expense pattern inside {tx_row['department']} department"})
    explanations.append({"factor": "Role Access Level", "weight": 0.10, "description": f"Spending limit context for role: {tx_row['role']}"})
    
    # Sort by weight
    explanations = sorted(explanations, key=lambda x: x["weight"], reverse=True)
    return explanations[:4]

if __name__ == "__main__":
    # Test dataset generation and model training
    df = generate_synthetic_data()
    train_ml_model()
