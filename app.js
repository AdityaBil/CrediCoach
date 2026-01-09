const form = document.getElementById('appForm');
const results = document.getElementById('results');
const riskEl = document.getElementById('risk');
const statusEl = document.getElementById('status');
const factorsEl = document.getElementById('factors');
const adviceEl = document.getElementById('advice');
const submitBtn = document.getElementById('submitBtn');
const spinner = document.getElementById('spinner');
const exampleBtn = document.getElementById('exampleBtn');
const themeToggle = document.getElementById('themeToggle');
const formError = document.getElementById('formError');
const refreshHistoryBtn = document.getElementById('refreshHistory');
const historyTableBody = document.querySelector('#historyTable tbody');
const historyEmpty = document.getElementById('historyEmpty');
const heroExampleBtn = document.getElementById('heroExampleBtn');

// Base URL for the Flask API (always target the backend on port 5000)
const API_BASE ='';

// Theme toggle (simple light/dark)
themeToggle?.addEventListener('click', () => {
  const body = document.body;
  const dark = body.classList.toggle('theme-dark');
  if (dark) body.classList.remove('theme-light');
  else body.classList.add('theme-light');
});

const showLoading = (on = true) => {
  if (on) {
    spinner.classList.remove('hidden');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Assessing...';
  } else {
    spinner.classList.add('hidden');
    submitBtn.disabled = false;
    submitBtn.textContent = 'Assess';
  }
};

const clearFieldErrors = () => {
  formError.textContent = '';
  document.querySelectorAll('.field-error').forEach(el => (el.textContent = ''));
};

const setFieldError = (name, message) => {
  const span = document.querySelector(`.field-error[data-error-for="${name}"]`);
  if (span) span.textContent = message;
};

// Basic client-side validation aligned with expected ranges
const validateForm = () => {
  clearFieldErrors();
  let ok = true;

  const v = name => form.elements[name]?.value;
  const num = name => (v(name) === '' ? NaN : Number(v(name)));

  const income = num('monthly_income');
  if (!income || income <= 0) {
    setFieldError('monthly_income', 'Enter a positive monthly income.');
    ok = false;
  }

  const score = num('credit_score');
  if (!score || score < 300 || score > 850) {
    setFieldError('credit_score', 'Credit score must be between 300 and 850.');
    ok = false;
  }

  const loanAmount = num('loan_amount');
  if (!loanAmount || loanAmount <= 0) {
    setFieldError('loan_amount', 'Enter a positive loan amount.');
    ok = false;
  }

  const loanDuration = num('loan_duration');
  if (!loanDuration || loanDuration < 6 || loanDuration > 120) {
    setFieldError('loan_duration', 'Duration should be between 6 and 120 months.');
    ok = false;
  }

  const debt = num('total_debt_payments');
  if (!debt || debt < 0) {
    setFieldError('total_debt_payments', 'Debt payments cannot be negative.');
    ok = false;
  }

  const hist = num('payment_history');
  if (isNaN(hist) || hist < 0 || hist > 60) {
    setFieldError('payment_history', 'Payment history must be between 0 and 60.');
    ok = false;
  }

  const util = num('credit_card_utilization');
  if (isNaN(util) || util < 0 || util > 1) {
    setFieldError('credit_card_utilization', 'Utilization must be between 0 and 1.');
    ok = false;
  }

  const bank = num('bankruptcy_history');
  if (!isNaN(bank) && (bank < 0 || bank > 1)) {
    setFieldError('bankruptcy_history', 'Use 0 for No or 1 for Yes.');
    ok = false;
  }

  const prevDef = num('previous_defaults');
  if (!isNaN(prevDef) && (prevDef < 0 || prevDef > 1)) {
    setFieldError('previous_defaults', 'Use 0 for No or 1 for Yes.');
    ok = false;
  }

  if (!ok) {
    formError.textContent = 'Please fix the highlighted fields before assessing.';
  }
  return ok;
};

const toPayload = () => {
  const formData = new FormData(form);
  const payload = {};
  for (const [k, v] of formData.entries()) {
    if (v === '') continue;
    // Send numeric values as numbers where possible
    const num = Number(v);
    payload[k] = Number.isNaN(num) ? v : num;
  }
  return payload;
};

const renderResult = data => {
  results.classList.remove('hidden');
  const risk = Number(data.risk_score ?? 0);
  riskEl.textContent = Number.isFinite(risk) ? risk.toFixed(2) : '-';
  statusEl.textContent = data.approval === 1 ? 'APPROVED' : 'UNDER REVIEW';
  factorsEl.textContent = (data.top_factors || []).join(', ');
  adviceEl.textContent = data.advice || '';
};

const renderHistory = (items = []) => {
  historyTableBody.innerHTML = '';
  if (!items.length) {
    historyEmpty.classList.remove('hidden');
    return;
  }
  historyEmpty.classList.add('hidden');
  items.forEach(row => {
    const tr = document.createElement('tr');
    const dt = new Date(row.timestamp);
    tr.innerHTML = `
      <td>${dt.toLocaleTimeString()}</td>
      <td>${row.risk_score.toFixed(2)}</td>
      <td>${row.approval === 1 ? 'APPROVED' : 'UNDER REVIEW'}</td>
      <td>$${row.inputs.monthly_income}</td>
      <td>$${row.inputs.loan_amount}</td>
    `;
    historyTableBody.appendChild(tr);
  });
};

const loadHistory = async () => {
  try {
    const res = await fetch(`${API_BASE}/api/history`);
    if (!res.ok) throw new Error(`Failed to load history (status ${res.status})`);
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (e) {
      console.warn('History response was not valid JSON', e, text);
      data = { items: [] };
    }
    renderHistory(data.items || []);
  } catch (err) {
    // history is non-critical; fail silently in UI
    console.error(err);
  }
};

form.addEventListener('submit', async e => {
  e.preventDefault();
  if (!validateForm()) return;
  showLoading(true);

  const payload = toPayload();

  try {
    const res = await fetch(`${API_BASE}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const text = await res.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch (e) {
      console.error('Predict response was not valid JSON', e, text);
      throw new Error('Server did not return valid JSON. Make sure you opened the app from the Flask server (http://127.0.0.1:5000).');
    }
    if (res.ok && !data.error) {
      renderResult(data);
      await loadHistory();
    } else {
      alert('Error: ' + (data.error || `Status ${res.status}`));
    }
  } catch (err) {
    alert('Request failed: ' + err.message);
  } finally {
    showLoading(false);
  }
});

exampleBtn.addEventListener('click', () => {
  const examples = {
    monthly_income: 4200,
    credit_score: 720,
    loan_amount: 8000,
    loan_duration: 36,
    total_debt_payments: 450,
    payment_history: 55,
    credit_card_utilization: 0.15,
    checking_balance: 2000,
    savings_balance: 3000,
    total_assets: 15000,
    total_liabilities: 4000,
    length_credit_history: 7,
    bankruptcy_history: 0,
    previous_defaults: 0
  };
  for (const k in examples) {
    const el = form.elements[k];
    if (el) el.value = examples[k];
  }
  clearFieldErrors();
});

heroExampleBtn?.addEventListener('click', () => {
  exampleBtn.click();
  window.scrollTo({ top: form.offsetTop - 20, behavior: 'smooth' });
});

refreshHistoryBtn.addEventListener('click', () => {
  loadHistory();
});

// Initial history load (if any) on page open
loadHistory();


