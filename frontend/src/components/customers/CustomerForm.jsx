import { useState } from 'react';
import Input from '../ui/Input';
import Select from '../ui/Select';
import Button from '../ui/Button';

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

const EMPTY = {
  customer_code: '',
  first_name: '',
  last_name: '',
  email: '',
  tenure_months: '',
  support_calls: '',
  late_payments: '',
  satisfaction_score: '',
  contract_type: '',
  payment_method: '',
  region: '',
};

export default function CustomerForm({ initial = null, onSubmit, onCancel, loading }) {
  const [form, setForm] = useState(
    initial
      ? {
          ...initial,
          tenure_months: String(initial.tenure_months),
          support_calls: String(initial.support_calls ?? 0),
          late_payments: String(initial.late_payments ?? 0),
          satisfaction_score: String(initial.satisfaction_score),
        }
      : EMPTY
  );
  const [errors, setErrors] = useState({});

  function validate() {
    const e = {};
    if (!form.customer_code.trim()) e.customer_code = 'Customer code required.';
    if (!form.first_name.trim()) e.first_name = 'First name required.';
    if (!form.last_name.trim()) e.last_name = 'Last name required.';
    if (form.tenure_months === '' || isNaN(+form.tenure_months) || +form.tenure_months < 0)
      e.tenure_months = 'Enter valid tenure months (0-100).';
    if (
      form.satisfaction_score === '' ||
      isNaN(+form.satisfaction_score) ||
      +form.satisfaction_score < 0 ||
      +form.satisfaction_score > 10
    )
      e.satisfaction_score = 'Score must be 0.0 – 10.0.';
    if (!form.contract_type) e.contract_type = 'Select contract type.';
    if (!form.payment_method) e.payment_method = 'Select payment method.';
    if (!form.region) e.region = 'Select region.';
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  function handleSubmit(e) {
    e.preventDefault();
    if (!validate()) return;
    onSubmit({
      customer_code: form.customer_code.trim(),
      first_name: form.first_name.trim(),
      last_name: form.last_name.trim(),
      email: form.email?.trim() || null,
      tenure_months: parseInt(form.tenure_months, 10),
      support_calls: parseInt(form.support_calls || '0', 10),
      late_payments: parseInt(form.late_payments || '0', 10),
      satisfaction_score: parseFloat(form.satisfaction_score),
      contract_type: form.contract_type,
      payment_method: form.payment_method,
      region: form.region,
    });
  }

  function f(key, val) {
    setForm((p) => ({ ...p, [key]: val }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: '' }));
  }

  return (
    <form onSubmit={handleSubmit} noValidate>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 14, marginBottom: 16 }}>
        <Input
          label="Customer Code"
          placeholder="CUST-1001"
          value={form.customer_code}
          onChange={(e) => f('customer_code', e.target.value)}
          error={errors.customer_code}
          required
          disabled={!!initial}
        />
        <Input
          label="Email address"
          type="email"
          placeholder="john@company.com"
          value={form.email}
          onChange={(e) => f('email', e.target.value)}
          error={errors.email}
        />
        <Input
          label="First Name"
          placeholder="John"
          value={form.first_name}
          onChange={(e) => f('first_name', e.target.value)}
          error={errors.first_name}
          required
        />
        <Input
          label="Last Name"
          placeholder="Doe"
          value={form.last_name}
          onChange={(e) => f('last_name', e.target.value)}
          error={errors.last_name}
          required
        />
        <Input
          label="Tenure (months)"
          type="number"
          min={0}
          max={100}
          placeholder="12"
          value={form.tenure_months}
          onChange={(e) => f('tenure_months', e.target.value)}
          error={errors.tenure_months}
          required
        />
        <Input
          label="Satisfaction Score (0.0 – 10.0)"
          type="number"
          min={0}
          max={10}
          step={0.1}
          placeholder="7.5"
          value={form.satisfaction_score}
          onChange={(e) => f('satisfaction_score', e.target.value)}
          error={errors.satisfaction_score}
          required
        />
        <Input
          label="Support Calls"
          type="number"
          min={0}
          max={50}
          placeholder="2"
          value={form.support_calls}
          onChange={(e) => f('support_calls', e.target.value)}
        />
        <Input
          label="Late Payments"
          type="number"
          min={0}
          max={30}
          placeholder="0"
          value={form.late_payments}
          onChange={(e) => f('late_payments', e.target.value)}
        />
        <Select
          label="Contract Type"
          options={CONTRACT_OPTIONS}
          value={form.contract_type}
          onValueChange={(v) => f('contract_type', v)}
          error={errors.contract_type}
          required
        />
        <Select
          label="Payment Method"
          options={PAYMENT_OPTIONS}
          value={form.payment_method}
          onValueChange={(v) => f('payment_method', v)}
          error={errors.payment_method}
          required
        />
        <Select
          label="Region"
          options={REGION_OPTIONS}
          value={form.region}
          onValueChange={(v) => f('region', v)}
          error={errors.region}
          required
        />
      </div>

      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10, marginTop: 12, borderTop: '1px solid var(--border-subtle)', paddingTop: 16 }}>
        <Button type="button" variant="secondary" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" variant="primary" loading={loading}>
          {initial ? 'Save Changes' : 'Create Customer'}
        </Button>
      </div>
    </form>
  );
}
