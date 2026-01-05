import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score, accuracy_score
from sklearn.ensemble import RandomForestRegressor,StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Flatten,LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet,Ridge,Lasso
from scikeras.wrappers import KerasRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVC
import os
import re

# Optional GenAI import — use real `genai` if installed, otherwise fallback
try:
    import genai
    _GENAI_AVAILABLE = True
except Exception:
    _GENAI_AVAILABLE = False
    class _FakeGenAI:
        def configure(self, api_key):
            pass
        class GenerativeModel:
            def __init__(self, *args, **kwargs):
                pass
            def generate_content(self, prompt):
                class Resp:
                    text = "[Local fallback] GenAI not available. Improve payments, reduce balances, and check credit utilization."
                return Resp()
    genai = _FakeGenAI()

#Data Loading and Preprocessing
df1 = pd.read_csv('focused_synthetic_loan_data.csv')
print(df1.loc[:5,:])
df1["LogLoanAmount"] = np.log1p(df1["LoanAmount"])
df1["LogNetWorth"] = np.log1p(df1["NetWorth"])
df1["IncomeToLoanRatio"] = df1["MonthlyIncome"] / (df1["MonthlyLoanPayment"] + 1)
df1['AssetsToLiabilities'] = df1['TotalAssets'] / (df1['TotalLiabilities'] + 1)


X = df1[['TotalDebtToIncomeRatio', 'CreditScore','LoanDuration', 'InterestRate',"IncomeToLoanRatio" ,"LogLoanAmount","LogNetWorth",'MonthlyLoanPayment',
    'BankruptcyHistory','PaymentHistory',"LengthOfCreditHistory",'AssetsToLiabilities',"CheckingAccountBalance","SavingsAccountBalance","PreviousLoanDefaults","CreditCardUtilizationRate"]]
y = df1["RiskScore"]

#'UtilityBillsPaymentHistory',  
# Encode categorical columns
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Classification
X_C=df1[['MonthlyIncome','TotalDebtToIncomeRatio', 'CreditScore','NetWorth','LoanAmount', 'LoanDuration', 'InterestRate', 'MonthlyLoanPayment',
    'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory',"TotalAssets"]]
Y_C=df1["LoanApproved"]
XC_train,XC_test,yC_train,yc_test = train_test_split(X_C,Y_C,test_size=0.3,random_state=42)
scalerC=StandardScaler().fit(XC_train)
XC_train=scalerC.transform(XC_train)
XC_test=scalerC.transform(XC_test)


def meta_mlp():
    model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],)),LeakyReLU(alpha=0.1),
    Dense(64),LeakyReLU(alpha=0.1),
    Dense(32),LeakyReLU(alpha=0.1),
    Dense(1)
])
    model.compile(optimizer=Adam(0.0001),loss='mse')
    return model


#Model Definitions
rf = RandomForestRegressor(n_estimators=300,random_state=42)
xgb = GradientBoostingRegressor(n_estimators=210, learning_rate=0.05, max_depth=4, random_state=42)
from scikeras.wrappers import KerasRegressor
mlp_model=KerasRegressor(model=meta_mlp,epochs=100, batch_size=8, verbose=0)

#Lets skip to the good part --> Ensemble Learning :)
base_models=[
    ('rf',rf),('gbr',GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))]

Stack = StackingRegressor(
    estimators=base_models,
    final_estimator = ElasticNetCV(
        l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
        alphas=np.logspace(-3, 1, 20),
        cv=5,
        random_state=42
    ),
    passthrough=True
)

Stack.fit(X_train, y_train)

# Fit a lightweight MLP regressor
mlp_model.fit(X_train, y_train)

pred1 = Stack.predict(X_test)
pred2 = mlp_model.predict(X_test)

# Final Ensemble (Simple Average)
best_pred = (pred1 + pred2) / 2

print(X_test)
print(best_pred)

print("MSE:", mean_squared_error(y_test, best_pred))
print("R² Score:", r2_score(y_test, best_pred))


def plot_graph():
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, best_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Risk Score")
    plt.ylabel("Predicted Risk Score")
    plt.title("Actual vs Predicted Risk Score")
    plt.grid(True)
    plt.show()
plot_graph()

#Classification Begins
model=SVC(kernel='linear',C=0.5,gamma="scale")
model.fit(XC_train, yC_train)

class_preds=model.predict(XC_test)
print(class_preds)
print("Accuracy score: ", accuracy_score(yc_test,class_preds))


from sklearn.metrics import confusion_matrix
import seaborn as sns
def plot_heatmap():
    cm = confusion_matrix(yc_test, class_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

plot_heatmap()

from sklearn.metrics import precision_recall_curve, average_precision_score
y_probs = model.decision_function(XC_test)

def plot_precision():
    precision, recall, _ = precision_recall_curve(yc_test, y_probs)
    avg_precision = average_precision_score(yc_test, y_probs)
    plt.figure(figsize=(8,6))
    plt.plot(recall, precision, label=f'Avg Precision = {avg_precision:.2f}', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    plt.show()

plot_precision()

# Local explanation using GradientBoosting feature importances (no shap)
gbr = xgb
gbr.fit(X_train, y_train)
feature_names = X.columns.tolist()
importances = gbr.feature_importances_
top_idx = np.argsort(importances)[::-1][:3]
top_factors = [feature_names[i] for i in top_idx]
print(f"Top reasons for this Risk Score: {top_factors}")

# Configure GenAI if an API key is available in the environment
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    try:
        genai.configure(api_key=api_key)
    except Exception:
        pass
else:
    print("NOTE: set environment variable GOOGLE_API_KEY to enable Gemini responses; using fallback if unavailable.")

def get_credicoach_advice(risk_score, top_factors, approval_status):
    # Determine the tone based on approval
    status_msg = "Approved" if approval_status == 1 else "Rejected"
    
    # The "Prompt Engineering" - This is where the persona lives
    prompt = f"""
    You are CrediCoach, a sophisticated and empathetic financial advisor. 
    
    Context:
    - Application Status: {status_msg}
    - Predicted Risk Score: {risk_score:.2f} (Lower is better)
    - Key Factors Influencing this Decision: {', '.join(top_factors)}
    
    Task:
    1. If Rejected: Explain clearly *why* based on the key factors. Don't use math jargon. 
    2. Provide 3 specific, actionable steps the user can take to improve these specific factors.
    3. If Approved: Congratulate them and give one tip on maintaining this good standing.
    
    Keep the response under 100 words. Be professional yet warm (Old Money aesthetic).
    """
    
    # Try using Gemini via genai if available and configured
    try:
        if _GENAI_AVAILABLE and api_key:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return getattr(response, 'text', str(response))
    except Exception:
        pass

    # Fallback local advice if Gemini is unavailable or fails
    if approval_status == 1:
        return f"Approved — Congratulations. Keep monitoring {top_factors[0]} and maintain on-time payments."
    else:
        steps = [f"Improve {f}: reduce balances, negotiate rates, or make timely payments." for f in top_factors]
        return f"Rejected — Main issues: {', '.join(top_factors)}. Suggested steps: {steps[0]} {steps[1]} {steps[2]}"

# --- INTERACTIVE USER INPUT SECTION ---
print("\n" + "="*60)
print("CREDICOACH - INTERACTIVE LOAN APPLICATION ASSESSMENT")
print("="*60)

# Collect user inputs
print("\nPlease enter your financial information:\n")
def _parse_number_from_string(s: str):
    s = (s or "").strip()
    if s.startswith('&'):
        s = s[1:].strip()
    # Find first numeric token in the string (int/float, optional exponent)
    m = re.search(r'[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?', s)
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None

def read_float(prompt, min_value=None, max_value=None, default=None):
    while True:
        raw = input(prompt).strip()
        # direct parse
        try:
            val = float(raw)
        except Exception:
            val = _parse_number_from_string(raw)

        if val is None:
            if default is not None:
                return default
            print("Invalid input. Please enter a numeric value (e.g., 2500 or 2500.50).")
            continue
        if min_value is not None and val < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and val > max_value:
            print(f"Value must be <= {max_value}.")
            continue
        return val

def read_int(prompt, min_value=None, max_value=None, default=None):
    v = read_float(prompt, min_value=min_value, max_value=max_value, default=default)
    return int(round(v))

try:
    monthly_income = read_float("Monthly Income ($): ", min_value=1)
    credit_score = read_int("Credit Score (300-850): ", min_value=300, max_value=850)
    loan_amount = read_float("Loan Amount ($): ", min_value=0)
    loan_duration = read_int("Loan Duration (months): ", min_value=1)
    total_debt_payments = read_float("Total Monthly Debt Payments ($): ", min_value=0)
    bankruptcy_history = read_int("Bankruptcy History (0=No, 1=Yes): ", min_value=0, max_value=1)
    previous_defaults = read_int("Previous Loan Defaults (0=No, 1=Yes): ", min_value=0, max_value=1)
    payment_history = read_int("Payment History (0-60, higher is better): ", min_value=0, max_value=60)
    length_credit_history = read_int("Length of Credit History (years): ", min_value=0)
    checking_balance = read_float("Checking Account Balance ($): ", min_value=0)
    savings_balance = read_float("Savings Account Balance ($): ", min_value=0)
    total_assets = read_float("Total Assets ($): ", min_value=0)
    total_liabilities = read_float("Total Liabilities ($): ", min_value=0)
    credit_card_utilization = read_float("Credit Card Utilization Rate (0-1): ", min_value=0, max_value=1, default=0.0)
    
    # Calculate derived features
    net_worth = max(total_assets - total_liabilities, 1000)
    log_loan_amount = np.log1p(loan_amount)
    log_net_worth = np.log1p(net_worth)
    income_to_loan_ratio = monthly_income / (total_debt_payments + 1)
    assets_to_liabilities = total_assets / (total_liabilities + 1)
    monthly_loan_payment = (loan_amount * 0.08) / 12
    total_debt_to_income = (total_debt_payments + monthly_loan_payment) / monthly_income
    interest_rate = 0.05 + (850 - credit_score) / 2000
    
    # Create user feature vector for regression (RiskScore)
    user_features_reg = pd.DataFrame({
        'TotalDebtToIncomeRatio': [total_debt_to_income],
        'CreditScore': [credit_score],
        'LoanDuration': [loan_duration],
        'InterestRate': [interest_rate],
        'IncomeToLoanRatio': [income_to_loan_ratio],
        'LogLoanAmount': [log_loan_amount],
        'LogNetWorth': [log_net_worth],
        'MonthlyLoanPayment': [monthly_loan_payment],
        'BankruptcyHistory': [bankruptcy_history],
        'PaymentHistory': [payment_history],
        'LengthOfCreditHistory': [length_credit_history],
        'AssetsToLiabilities': [assets_to_liabilities],
        'CheckingAccountBalance': [checking_balance],
        'SavingsAccountBalance': [savings_balance],
        'PreviousLoanDefaults': [previous_defaults],
        'CreditCardUtilizationRate': [credit_card_utilization]
    })
    
    # Scale and predict risk score
    user_features_reg_scaled = scaler.transform(user_features_reg)
    user_risk_score = float(Stack.predict(user_features_reg_scaled)[0])
    user_risk_score_mlp = float(mlp_model.predict(user_features_reg_scaled)[0])
    user_risk_score = (user_risk_score + user_risk_score_mlp) / 2
    
    # Create user feature vector for classification (LoanApproved)
    user_features_class = pd.DataFrame({
        'MonthlyIncome': [monthly_income],
        'TotalDebtToIncomeRatio': [total_debt_to_income],
        'CreditScore': [credit_score],
        'NetWorth': [net_worth],
        'LoanAmount': [loan_amount],
        'LoanDuration': [loan_duration],
        'InterestRate': [interest_rate],
        'MonthlyLoanPayment': [monthly_loan_payment],
        'BankruptcyHistory': [bankruptcy_history],
        'PreviousLoanDefaults': [previous_defaults],
        'PaymentHistory': [payment_history],
        'TotalAssets': [total_assets]
    })
    
    # Scale and predict approval
    user_features_class_scaled = scalerC.transform(user_features_class)
    user_approval = int(model.predict(user_features_class_scaled)[0])
    
    # Get the advice from Gemini
    advice = get_credicoach_advice(user_risk_score, top_factors, user_approval)
    
    print("\n" + "="*60)
    print("CREDICOACH AI ASSESSMENT RESULTS")
    print("="*60)
    print(f"Risk Score: {user_risk_score:.2f} (Lower is better)")
    print(f"Application Status: {'APPROVED' if user_approval == 1 else 'UNDER REVIEW'}")
    print(f"Key Influencing Factors: {', '.join(top_factors)}")
    print("="*60)
    print("RECOMMENDATION:")
    print("-"*60)
    print(advice)
    print("-"*60)
    
except ValueError as e:
    print(f"\nError: Invalid input. Please enter valid numbers. {e}")
except Exception as e:
    print(f"\nError processing application: {e}")
