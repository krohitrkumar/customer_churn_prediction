import { useState, useEffect } from 'react';
import { useSearchParams, useNavigate } from 'react-router-dom';
import { useCustomers } from '../hooks/useCustomers';
import { usePrediction } from '../hooks/usePredictions';
import { useToastState } from '../hooks/useToast';
import { extractError } from '../api/client';
import ChurnGauge from '../components/predictions/ChurnGauge';
import PlaybookCard from '../components/predictions/PlaybookCard';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';
import Button from '../components/ui/Button';
import Spinner from '../components/ui/Spinner';
import ToastContainer from '../components/ui/Toast';
import './PredictPage.css';

const CONTRACT_OPTIONS = [
  { value: 'month_to_month', label: 'Month-to-Month' },
  { value: 'one_year', label: 'One Year' },
  { value: 'two_year', label: 'Two Year' },
];
const PAYMENT_OPTIONS = [
  { value: 'card', label: 'Card (Credit/Debit)' },
  { value: 'wallet', label: 'Digital Wallet' },
  { value: 'bank', label: 'Bank Transfer' },
];
const REGION_OPTIONS = [
  { value: 'north_america', label: 'North America' },
  { value: 'europe', label: 'Europe' },
  { value: 'asia', label: 'Asia Pacific' },
  { value: 'latam', label: 'Latin America (LATAM)' },
  { value: 'africa', label: 'Africa' },
  { value: 'south_america', label: 'South America' },
];

const EMPTY_FORM = {
  tenure_months: '', support_calls: '', late_payments: '',
  satisfaction_score: '', contract_type: '', payment_method: '', region: '',
};

export default function PredictPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const preCustomerId = searchParams.get('customerId');

  const { customers, loading: customersLoading } = useCustomers();
  const { result, loading, predict, reset } = usePrediction();
  const { toasts, error: toastError, dismiss } = useToastState();

  const [form, setForm] = useState(EMPTY_FORM);
  const [errors, setErrors] = useState({});
  const [customerId, setCustomerId] = useState(preCustomerId || '');

  // Pre-fill form if customer selected
  useEffect(() => {
    if (!customerId || !customers.length) return;
    const c = customers.find((x) => String(x.id) === String(customerId));
    if (c) {
      setForm({
        tenure_months:      String(c.tenure_months),
        support_calls:      String(c.support_calls),
        late_payments:      String(c.late_payments),
        satisfaction_score: String(c.satisfaction_score),
        contract_type:      c.contract_type,
        payment_method:     c.payment_method,
        region:             c.region,
      });
    }
  }, [customerId, customers]);

  const customerOptions = [
    { value: '', label: 'No customer (quick predict)' },
    ...customers.map((c) => ({ value: String(c.id), label: `${c.first_name} ${c.last_name} — ${c.customer_code}` })),
  ];

  function validate() {
    const e = {};
    if (!form.tenure_months || isNaN(+form.tenure_months) || +form.tenure_months < 1) e.tenure_months = 'Required (1-100).';
    if (form.satisfaction_score === '' || isNaN(+form.satisfaction_score) || +form.satisfaction_score < 1 || +form.satisfaction_score > 10) e.satisfaction_score = 'Required (1-10).';
    if (!form.contract_type)  e.contract_type  = 'Select contract type.';
    if (!form.payment_method) e.payment_method = 'Select payment method.';
    if (!form.region)         e.region         = 'Select region.';
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    reset();
    if (!validate()) return;
    try {
      await predict({
        tenure_months:      parseInt(form.tenure_months, 10),
        support_calls:      parseInt(form.support_calls || '0', 10),
        late_payments:      parseInt(form.late_payments || '0', 10),
        satisfaction_score: parseFloat(form.satisfaction_score),
        contract_type:      form.contract_type,
        payment_method:     form.payment_method,
        region:             form.region,
        customer_id:        customerId ? parseInt(customerId, 10) : undefined,
      });
    } catch (err) {
      toastError('Prediction failed', extractError(err));
    }
  }

  function f(key, val) { setForm((p) => ({ ...p, [key]: val })); if (errors[key]) setErrors((e) => ({ ...e, [key]: '' })); }

  return (
    <div>
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
      <div className="page-header">
        <h1>AI Churn Predictor</h1>
        <p>Run a real-time churn risk prediction using the trained ML model</p>
      </div>

      <div className="predict-layout">
        {/* Input Form */}
        <div className="card predict-form-card">
          <h3 style={{ margin: '0 0 20px', fontSize: 'var(--text-md)' }}>Customer Parameters</h3>
          <form onSubmit={handleSubmit} noValidate>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {/* Optional customer link */}
              {!customersLoading && (
                <Select label="Link to existing customer (optional)" options={customerOptions}
                  value={customerId} onValueChange={(v) => { setCustomerId(v); if (!v) setForm(EMPTY_FORM); }} />
              )}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <Input label="Tenure (months)" type="number" min={1} max={100} placeholder="12" value={form.tenure_months} onChange={(e) => f('tenure_months', e.target.value)} error={errors.tenure_months} required />
                <Input label="Satisfaction Score (1-10)" type="number" min={1} max={10} step={0.1} placeholder="7.0" value={form.satisfaction_score} onChange={(e) => f('satisfaction_score', e.target.value)} error={errors.satisfaction_score} required />
                <Input label="Support Calls" type="number" min={0} max={50} placeholder="2" value={form.support_calls} onChange={(e) => f('support_calls', e.target.value)} />
                <Input label="Late Payments" type="number" min={0} max={30} placeholder="0" value={form.late_payments} onChange={(e) => f('late_payments', e.target.value)} />
              </div>
              <Select label="Contract Type" options={CONTRACT_OPTIONS} value={form.contract_type} onValueChange={(v) => f('contract_type', v)} error={errors.contract_type} required />
              <Select label="Payment Method" options={PAYMENT_OPTIONS} value={form.payment_method} onValueChange={(v) => f('payment_method', v)} error={errors.payment_method} required />
              <Select label="Region" options={REGION_OPTIONS} value={form.region} onValueChange={(v) => f('region', v)} error={errors.region} required />

              <Button type="submit" variant="primary" size="lg" fullWidth loading={loading}
                leftIcon={!loading && <svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M1 13L6 6l4 4 4-8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/><circle cx="14" cy="3" r="2" fill="currentColor"/></svg>}>
                {loading ? 'Running model…' : 'Run Prediction'}
              </Button>
            </div>
          </form>
        </div>

        {/* Result Panel */}
        <div className="predict-result">
          {loading && (
            <div className="predict-loading">
              <Spinner size="lg" label="Running ML model…" />
              <p style={{ color: 'var(--text-secondary)', marginTop: 16 }}>Running churn model…</p>
            </div>
          )}

          {!loading && !result && (
            <div className="predict-empty">
              <div style={{ fontSize: 48, marginBottom: 16 }}>🧠</div>
              <h3>Ready to predict</h3>
              <p>Fill in the customer parameters and run the model to see churn risk score and retention playbooks.</p>
            </div>
          )}

          {!loading && result && (
            <div className="anim-fade-up">
              {/* Gauge */}
              <div className="card predict-gauge-card">
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                  <ChurnGauge probability={result.churn_probability} size={220} />
                  <div style={{ textAlign: 'center' }}>
                    <p style={{ margin: 0, fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Churn Probability</p>
                    <p style={{ margin: '4px 0 0', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
                      Model prediction: <strong style={{ color: result.churn_prediction === 1 ? 'var(--brand-crimson)' : 'var(--brand-emerald)' }}>
                        {result.churn_prediction === 1 ? 'Will Churn' : 'Will Stay'}
                      </strong>
                    </p>
                    {customerId && (
                      <Button variant="ghost" size="xs" style={{ marginTop: 10 }} onClick={() => navigate(`/customers/${customerId}`)}>
                        View Customer Profile →
                      </Button>
                    )}
                  </div>
                </div>
              </div>

              {/* Playbooks */}
              {result.playbooks?.length > 0 && (
                <div style={{ marginTop: 16 }}>
                  <h3 style={{ margin: '0 0 12px', fontSize: 'var(--text-md)', fontWeight: 600 }}>
                    Retention Playbooks
                    <span style={{ marginLeft: 8, fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontWeight: 400 }}>
                      {result.playbooks.length} recommendations
                    </span>
                  </h3>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {result.playbooks.map((p, i) => (
                      <PlaybookCard key={i} index={i} icon={p.icon} category={p.category} action={p.action} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
