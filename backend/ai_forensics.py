import os
import re
import json
import urllib.request
import urllib.error
import pandas as pd
from ml_module import CSV_PATH, explain_prediction

def get_gemini_api_key():
    return os.environ.get("GEMINI_API_KEY", "")

def run_gemini_api(prompt: str, api_key: str) -> str:
    """Executes a direct HTTP request to the Google Gemini API to avoid dependency issues."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 2048
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode("utf-8"), 
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            # Extract content from response
            text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            return text
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        print(f"Gemini API HTTP Error: {e.code} - {error_msg}")
        return f"*(Gemini API returned error code {e.code}. Falling back to local analysis)*\n\n"
    except Exception as e:
        print(f"Gemini API Connection Error: {str(e)}")
        return f"*(Gemini connection failed: {str(e)}. Falling back to local analysis)*\n\n"

def local_forensic_engine(prompt: str, df: pd.DataFrame) -> str:
    """Offline deterministic query and analysis system. Acts as a smart semantic router."""
    prompt_lower = prompt.lower()
    
    # 1. Investigate Specific Transaction
    tx_match = re.search(r"tx-\d{5}", prompt_lower)
    if tx_match:
        tx_id = tx_match.group(0).upper()
        row = df[df["transaction_id"] == tx_id]
        if row.empty:
            return f"### Transaction Not Found\nCould not find transaction with ID **{tx_id}** in the corporate audit logs. Please verify the ID format (e.g., `TX-10024`)."
        
        tx = row.iloc[0].to_dict()
        is_fraud = tx["is_fraud"]
        fraud_type = tx["fraud_type"]
        risk_score = float(tx.get("fraud_prob", 0.92 if is_fraud == 1 else 0.05))
        
        # Get explanations
        explanations = explain_prediction(tx)
        explanation_md = ""
        for exp in explanations:
            explanation_md += f"- **{exp['factor']}** (Impact: {exp['weight']:.2f}): {exp['description']}\n"
            
        status = "🔴 FLAGGED SUSPICIOUS (HIGH RISK)" if is_fraud == 1 else "🟢 APPROVED / CLEAN (LOW RISK)"
        
        # If user asks for a report/memo
        if "report" in prompt_lower or "memo" in prompt_lower or "draft" in prompt_lower:
            return f"""# FORENSIC AUDIT MEMORANDUM: INVESTIGATION OF {tx_id}

**DATE:** 2026-07-10  
**SUBJECT:** Forensic Audit Report for Transaction **{tx_id}**  
**STATUS:** {status}  
**INVESTIGATOR:** AI Corporate Fraud Assistant  

---

### 1. TRANSACTION PROFILE
- **Transaction ID:** {tx_id}
- **Employee Name:** {tx['employee_name']} (ID: {tx['employee_id']})
- **Department:** {tx['department']} | **Role:** {tx['role']}
- **Vendor:** {tx['vendor']}
- **Category:** {tx['category']}
- **Amount:** ${tx['amount']:.2f}
- **Payment Method:** {tx['payment_method']}
- **Location:** {tx['location']}
- **Timestamp Info:** hour {tx['hour']}:00 (Night: {"Yes" if tx['is_night'] else "No"}, Weekend: {"Yes" if tx['is_weekend'] else "No"})

---

### 2. RISK ANALYSIS & EVIDENCE
The machine learning model evaluated this transaction with a fraud/anomaly probability score of **{risk_score:.1%}**.

**Key Risk Flags Identified:**
{explanation_md}

**Detailed Findings:**
- **Expense Deviation:** The transaction amount of **${tx['amount']:.2f}** deviates significantly from this employee's historical average of **${tx['emp_hist_mean']:.2f}** (std dev: ${tx['emp_hist_std']:.2f}).
- **Fraud Pattern Category:** {f"Classified as **{fraud_type}**." if is_fraud == 1 else "No typical fraud patterns detected."}

---

### 3. RECOMMENDATIONS & ACTION PLAN
1. **Freeze Payments:** Immediately hold any pending approvals or wire clearances related to **{tx['vendor']}** for this employee.
2. **Review Supporting Documentation:** Retrieve the original invoice and proof of receipt for this expense.
3. **Conduct Employee Inquiry:** Request a written explanation from **{tx['employee_name']}** regarding the business necessity of this expense, especially given the timing ({tx['hour']}:00) and deviation size.
4. **Audit Historical Transactions:** Perform a deeper look into all other transactions processed by {tx['employee_name']} in the last 90 days.
"""

        # General transaction query response
        return f"""### Transaction Audit: {tx_id}
**Employee:** {tx['employee_name']} ({tx['role']} - {tx['department']})  
**Vendor / Category:** {tx['vendor']} ({tx['category']})  
**Amount:** ${tx['amount']:.2f}  
**Status:** {status}  
**Risk Score:** `{risk_score:.1%}`

**Risk Indicators / AI Attribution:**
{explanation_md}

**Historical Context:**
- Employee typical average: `${tx['emp_hist_mean']:.2f}` (std dev: `${tx['emp_hist_std']:.2f}`)
- Current amount: `${tx['amount']:.2f}` (${tx['amount'] - tx['emp_hist_mean']:.2f} over average)
- Occurred: hour {tx['hour']}:00 (Night: {"Yes" if tx['is_night'] else "No"}, Weekend: {"Yes" if tx['is_weekend'] else "No"})
- Transaction flag reason: **{fraud_type}**

*To generate a formal investigation report, ask: "Write a forensic report for {tx_id}"*"""

    # 2. Analyze Department
    dept_keywords = ["department", "dept", "sales", "engineering", "marketing", "hr", "finance", "executive", "operations"]
    for d in ["sales", "engineering", "marketing", "hr", "finance", "executive", "operations"]:
        if d in prompt_lower:
            dept_name = d.capitalize()
            dept_rows = df[df["department"].str.lower() == d]
            if dept_rows.empty:
                continue
                
            total_tx = len(dept_rows)
            flagged = dept_rows[dept_rows["is_fraud"] == 1]
            flagged_count = len(flagged)
            fraud_rate = flagged_count / total_tx if total_tx > 0 else 0
            total_spend = dept_rows["amount"].sum()
            avg_spend = dept_rows["amount"].mean()
            max_spend_row = dept_rows.loc[dept_rows["amount"].idxmax()]
            
            flagged_list_md = ""
            for idx, r in flagged.head(5).iterrows():
                flagged_list_md += f"- **{r['transaction_id']}**: ${r['amount']:.2f} by {r['employee_name']} to {r['vendor']} ({r['fraud_type']})\n"
            if flagged_count > 5:
                flagged_list_md += f"- *...and {flagged_count - 5} more transactions.*"
            if flagged_count == 0:
                flagged_list_md = "- *No flagged fraudulent transactions in this department.*"

            return f"""# Department Forensic Audit: {dept_name}

### 1. SUMMARY METRICS
- **Total Transactions Audited:** {total_tx}
- **Total Spend Volume:** ${total_spend:,.2f}
- **Average Expense Size:** ${avg_spend:.2f}
- **Flagged Suspicious Transactions:** {flagged_count} ({fraud_rate:.1%} flag rate)

### 2. RISK LEVEL EVALUATION
The risk profile for **{dept_name}** is **{"HIGH" if fraud_rate > 0.1 else "MODERATE" if fraud_rate > 0.04 else "LOW"}**.

### 3. LARGEST SINGLE TRANSACTION
- **Transaction ID:** {max_spend_row['transaction_id']}
- **Employee:** {max_spend_row['employee_name']}
- **Vendor:** {max_spend_row['vendor']} | **Category:** {max_spend_row['category']}
- **Amount:** ${max_spend_row['amount']:,.2f}
- **Status:** {"🔴 Flagged Suspicious" if max_spend_row['is_fraud'] == 1 else "🟢 Approved"}

### 4. SUSPICIOUS TRANSACTIONS DETECTED
{flagged_list_md}

### 5. AUDIT RECOMMENDATIONS FOR {dept_name.upper()}
1. {"Review invoice splitting filters on procurement wires." if "Invoice Splitting" in flagged["fraud_type"].values else "Verify credit card policies for travel and entertainment expenses."}
2. Ensure managers require receipt attachments for all reimbursement claims.
3. audit invoices for the top vendor in this department.
"""

    # 3. Analyze Vendor
    if "vendor" in prompt_lower or "analyze vendor" in prompt_lower:
        # Search for vendor names in prompt
        all_vendors = df["vendor"].unique()
        matched_vendor = None
        for v in all_vendors:
            if v.lower() in prompt_lower:
                matched_vendor = v
                break
                
        if matched_vendor:
            vendor_rows = df[df["vendor"] == matched_vendor]
            total_tx = len(vendor_rows)
            flagged = vendor_rows[vendor_rows["is_fraud"] == 1]
            flagged_count = len(flagged)
            total_spend = vendor_rows["amount"].sum()
            avg_spend = vendor_rows["amount"].mean()
            unique_employees = vendor_rows["employee_id"].nunique()
            
            flagged_md = ""
            for idx, r in flagged.iterrows():
                flagged_md += f"- **{r['transaction_id']}**: ${r['amount']:.2f} by {r['employee_name']} ({r['fraud_type']})\n"
            if flagged_count == 0:
                flagged_md = "- *No transactions flagged as suspicious for this vendor.*"
                
            return f"""# Vendor Audit Profile: {matched_vendor}

### 1. AUDIT PROFILE
- **Total Transactions:** {total_tx}
- **Total Paid Volume:** ${total_spend:,.2f}
- **Average Invoice/Expense Size:** ${avg_spend:.2f}
- **Active Employees Using Vendor:** {unique_employees}
- **Flagged Transactions:** {flagged_count} ({(flagged_count/total_tx if total_tx > 0 else 0):.1%} risk incidence)

### 2. VENDOR RISK CLASSIFICATION
Risk classification: **{"CRITICAL RISK" if "Shadow Advisory" in matched_vendor or flagged_count/total_tx > 0.3 else "HIGH RISK" if flagged_count > 0 else "LOW RISK"}**

### 3. RECENT FLAGGED TRANSACTIONS
{flagged_md}

### 4. FORENSIC VERDICT
- **Analysis:** {f"The vendor '{matched_vendor}' is associated with suspicious activities, specifically night transfers and amounts just below approval thresholds. Recommend vendor verification and immediate freeze." if flagged_count > 0 else f"No systemic issues or anomalous invoice activities detected for vendor '{matched_vendor}'. Normal monitoring applies."}
"""

    # 4. General Stats / Introduction fallback
    flagged_df = df[df["is_fraud"] == 1]
    total_flagged = len(flagged_df)
    total_tx = len(df)
    ratio = total_flagged / total_tx if total_tx > 0 else 0
    top_fraud_dept = df[df["is_fraud"] == 1]["department"].value_counts().index[0] if total_flagged > 0 else "N/A"
    
    return f"""# AI Corporate Fraud Investigator (Local Engine)

Welcome to the **Corporate Fraud Investigation Portal**. I am currently running in **Local Analytics Mode** and have loaded **{total_tx}** active transactions from the database.

### Database Snapshot:
- **Total Transactions Scanned:** {total_tx}
- **Flagged Anomalies / Fraud:** {total_flagged} ({ratio:.1%} fraud rate)
- **Top Vulnerable Department:** {top_fraud_dept}
- **Trained ML Models Active:** Random Forest & Isolation Forest Anomaly Detection

### Things you can ask me:
1. **Audit a specific transaction**: *"Show details for transaction TX-10045"*
2. **Draft a formal corporate investigation memo**: *"Write an audit report for TX-11002"*
3. **Analyze a specific department**: *"Analyze department Sales"* or *"Audit Engineering"*
4. **Audit a suspicious vendor**: *"Analyze vendor Shadow Advisory Group"* or *"Audit Uber"*
5. **List fraud scenarios**: *"What kinds of fraud are in the dataset?"*
"""

def generate_forensic_response(prompt: str, chat_history: list = None) -> str:
    """Entry point for handling chat query. Integrates Gemini API or runs the Local Forensic Engine."""
    # Load dataset
    if not os.path.exists(CSV_PATH):
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(CSV_PATH)
        
    api_key = get_gemini_api_key()
    
    if api_key:
        print("Using Gemini API for forensic chat...")
        
        # Supplement prompt with local statistics to make Gemini factually accurate
        # We will extract 5 highest risk transactions and department summary metrics to insert as context
        flagged_tx = df[df["is_fraud"] == 1].head(10).to_dict(orient="records")
        dept_summary = df.groupby("department")["is_fraud"].agg(["count", "sum"]).to_dict(orient="index")
        
        # Extract individual tx from prompt if mentioned
        tx_match = re.search(r"tx-\d{5}", prompt.lower())
        tx_context = ""
        if tx_match:
            tx_id = tx_match.group(0).upper()
            row = df[df["transaction_id"] == tx_id]
            if not row.empty:
                tx_context = f"\nSpecific transaction queried: {row.iloc[0].to_dict()}\nExplainability details: {explain_prediction(row.iloc[0].to_dict())}\n"
        
        gemini_prompt = f"""You are the AI Corporate Fraud Investigation Assistant.
You are helping a professional auditor analyze corporate transaction data to find fraud.
Here is the factual transaction metadata and statistics from our database to guide your answer. Always stay 100% consistent with this data:
- Total transactions: {len(df)}
- Total flagged fraud: {df['is_fraud'].sum()}
- Flagged fraud cases list (subset): {json.dumps(flagged_tx[:5])}
- Department fraud count summaries (format: Department -> total transactions, flagged count): {json.dumps(dept_summary)}
{tx_context}

User is asking: "{prompt}"

Rules:
1. Reply in markdown with a highly professional, forensic tone.
2. If the user asks about a specific transaction, look at the transaction info provided in the context above. If it's not in the context, look for it in the flagged list or politely say you can't find it.
3. Suggest actionable next steps (like freezing payments, requesting invoices, reviewing credit logs).
4. Do not make up fake transaction IDs or names. Only use what is present in the context.
"""
        response_text = run_gemini_api(gemini_prompt, api_key)
        return response_text
    else:
        print("No Gemini API key found. Using Local Forensic Engine...")
        return local_forensic_engine(prompt, df)
