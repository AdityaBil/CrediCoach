from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
import os
import json
from datetime import datetime

# Serve frontend from dist folder (built frontend) or fall back to static
static_folder = 'dist' if os.path.exists('dist') else 'static'
app = Flask(__name__, static_folder=static_folder, static_url_path='')
# allow the frontend to call the API
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure GenAI (Gemini) if available via env
try:
    import genai
    _GENAI_AVAILABLE = True
except Exception:
    genai = None
    _GENAI_AVAILABLE = False


def meta_mlp(X_train_shape):
    def _model():
        model = Sequential([
            Dense(128, input_shape=(X_train_shape,)), LeakyReLU(alpha=0.1),
            Dense(64), LeakyReLU(alpha=0.1),
            Dense(32), LeakyReLU(alpha=0.1),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.0001), loss='mse')
        return model
    return _model


print("Loading data and training models (startup). This may take a moment...")

MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

df1 = pd.read_csv('focused_synthetic_loan_data.csv')

# feature engineering (same as original script)
df1["LogLoanAmount"] = np.log1p(df1["LoanAmount"])
df1["LogNetWorth"] = np.log1p(df1["NetWorth"])
df1["IncomeToLoanRatio"] = df1["MonthlyIncome"] / (df1["MonthlyLoanPayment"] + 1)
df1['AssetsToLiabilities'] = df1['TotalAssets'] / (df1['TotalLiabilities'] + 1)

X = df1[[
    'TotalDebtToIncomeRatio', 'CreditScore', 'LoanDuration', 'InterestRate', 'IncomeToLoanRatio',
    'LogLoanAmount', 'LogNetWorth', 'MonthlyLoanPayment', 'BankruptcyHistory', 'PaymentHistory',
    'LengthOfCreditHistory', 'AssetsToLiabilities', 'CheckingAccountBalance', 'SavingsAccountBalance',
    'PreviousLoanDefaults', 'CreditCardUtilizationRate'
]]

y = df1['RiskScore']

# encode and scale
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# regression models (support FAST_MODE to speed up training inside containers)
FAST_MODE = str(os.environ.get('FAST_MODE', '0')).lower() in ('1', 'true', 'yes')
if FAST_MODE:
    rf = RandomForestRegressor(n_estimators=20, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=20, learning_rate=0.05, max_depth=3, random_state=42)
    mlp_epochs = 3
else:
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    mlp_epochs = 50

base_models=[('rf',rf),('gbr',gbr)]

mlp_model = KerasRegressor(model=meta_mlp(X.shape[1]), epochs=mlp_epochs, batch_size=8, verbose=0)

Stack = StackingRegressor(
    estimators=base_models,
    final_estimator = ElasticNetCV(
        l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],
        alphas=np.logspace(-3, 1, 20),
        cv=3,
        random_state=42
    ),
    passthrough=True
)

from joblib import dump, load

# Save/load models to speed up restarts
stack_path = os.path.join(MODELS_DIR, 'stack.joblib')
scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
scalerC_path = os.path.join(MODELS_DIR, 'scalerC.joblib')
clf_path = os.path.join(MODELS_DIR, 'clf.joblib')
topf_path = os.path.join(MODELS_DIR, 'top_factors.json')

LOAD_MODELS = str(os.environ.get('LOAD_MODELS', '1')).lower() in ('1', 'true', 'yes')

# Initialize top_factors_default first (will be computed later)
top_factors_default = []

if LOAD_MODELS and os.path.exists(stack_path) and os.path.exists(scaler_path) and os.path.exists(scalerC_path) and os.path.exists(clf_path) and os.path.exists(topf_path):
    print('Loading models from disk...')
    Stack = load(stack_path)
    scaler = load(scaler_path)
    scalerC = load(scalerC_path)
    clf = load(clf_path)
    with open(topf_path, 'r') as f:
        top_factors_default = json.load(f)
else:
    print('Training models (this may take a moment)...')
    Stack.fit(X_scaled, y)
    mlp_model.fit(X_scaled, y)
    gbr.fit(X_scaled, y)
    
    # classification training
    X_C = df1[['MonthlyIncome','TotalDebtToIncomeRatio', 'CreditScore','NetWorth','LoanAmount', 'LoanDuration', 'InterestRate', 'MonthlyLoanPayment',
        'BankruptcyHistory', 'PreviousLoanDefaults', 'PaymentHistory','TotalAssets']]
    Y_C = df1['LoanApproved']
    scalerC = StandardScaler().fit(X_C)
    XC_scaled = scalerC.transform(X_C)
    clf = SVC(kernel='linear', C=0.5, gamma='scale')
    clf.fit(XC_scaled, Y_C)
    
    # Compute top factors
    feature_names = X.columns.tolist()
    importances = gbr.feature_importances_
    top_idx = np.argsort(importances)[::-1][:3]
    top_factors_default = [feature_names[i] for i in top_idx]
    
    # persist
    try:
        dump(Stack, stack_path)
        dump(scaler, scaler_path)
        dump(scalerC, scalerC_path)
        dump(clf, clf_path)
        with open(topf_path, 'w') as f:
            json.dump(top_factors_default, f)
        print('Models saved to disk.')
    except Exception as e:
        print(f'Warning: failed to save models to disk: {e}; continuing without persistence')

# Simple in-memory history for the current server process
PREDICTION_HISTORY = []
MAX_HISTORY = 25


def get_credicoach_advice(risk_score, top_factors, approval_status):
    status_msg = "Approved" if approval_status == 1 else "Rejected"
    prompt = f"""
You are CrediCoach, a warm, professional financial advisor.

Context:
- Application Status: {status_msg}
- Predicted Risk Score: {risk_score:.2f}
- Key Factors: {', '.join(top_factors)}

Task:
1) If Rejected, explain why in plain language and give 3 actionable steps to improve the listed factors.
2) If Approved, congratulate and give one sustaining tip.
Keep it under 100 words.
"""

    # Try using Gemini via genai if configured
    api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GENAI_API_KEY')
    if _GENAI_AVAILABLE and api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return getattr(response, 'text', str(response))
        except Exception:
            pass

    # fallback local advice
    if approval_status == 1:
        return f"Approved — Congratulations. Keep monitoring {top_factors[0]} and maintain on-time payments."
    else:
        steps = [f"Improve {f}: reduce balances, negotiate rates, or make timely payments." for f in top_factors]
        return f"Rejected — Main issues: {', '.join(top_factors)}. Suggested steps: {steps[0]} {steps[1]} {steps[2]}"


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        # read inputs with defaults
        monthly_income = float(data.get('monthly_income', 3000))
        credit_score = int(data.get('credit_score', 650))
        loan_amount = float(data.get('loan_amount', 5000))
        loan_duration = int(data.get('loan_duration', 36))
        total_debt_payments = float(data.get('total_debt_payments', 500))
        bankruptcy_history = int(data.get('bankruptcy_history', 0))
        previous_defaults = int(data.get('previous_defaults', 0))
        payment_history = int(data.get('payment_history', 30))
        length_credit_history = int(data.get('length_credit_history', 5))
        checking_balance = float(data.get('checking_balance', 1000))
        savings_balance = float(data.get('savings_balance', 500))
        total_assets = float(data.get('total_assets', 5000))
        total_liabilities = float(data.get('total_liabilities', 1000))
        credit_card_utilization = float(data.get('credit_card_utilization', 0.2))

        net_worth = max(total_assets - total_liabilities, 1000)
        log_loan_amount = np.log1p(loan_amount)
        log_net_worth = np.log1p(net_worth)
        income_to_loan_ratio = monthly_income / (total_debt_payments + 1)
        assets_to_liabilities = total_assets / (total_liabilities + 1)
        monthly_loan_payment = (loan_amount * 0.08) / 12
        total_debt_to_income = (total_debt_payments + monthly_loan_payment) / monthly_income
        interest_rate = 0.05 + (850 - credit_score) / 2000

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

        user_scaled = scaler.transform(user_features_reg)
        risk1 = float(Stack.predict(user_scaled)[0])
        risk2 = float(mlp_model.predict(user_scaled)[0])
        risk_score = (risk1 + risk2) / 2.0

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

        user_class_scaled = scalerC.transform(user_features_class)
        approval = int(clf.predict(user_class_scaled)[0])

        # top factors
        top_factors = top_factors_default

        advice = get_credicoach_advice(risk_score, top_factors, approval)

        # record into simple history
        record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'risk_score': float(risk_score),
            'approval': approval,
            'top_factors': top_factors,
            'inputs': {
                'monthly_income': monthly_income,
                'loan_amount': loan_amount
            }
        }
        PREDICTION_HISTORY.append(record)
        if len(PREDICTION_HISTORY) > MAX_HISTORY:
            del PREDICTION_HISTORY[0:len(PREDICTION_HISTORY) - MAX_HISTORY]

        return jsonify({
            'risk_score': float(risk_score),
            'approval': approval,
            'top_factors': top_factors,
            'advice': advice
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def history():
    """
    Return a small list of recent predictions (in-memory, per-process).
    """
    # newest first
    return jsonify({'items': list(reversed(PREDICTION_HISTORY))})


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    """Serve frontend assets or fall back to index.html for SPA routing."""
    if path != '' and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    # SPA fallback: serve index.html for all other routes
    index_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, 'index.html')
    return {'error': 'Frontend build not found. Run "npm run build" first.'}, 404


if __name__ == '__main__':
    # Production: disable debug mode, use environment variable for port
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('1', 'true', 'yes')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
