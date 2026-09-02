import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { registerUser, sendOtp } from '../api/auth';
import { extractError } from '../api/client';
import Input from '../components/ui/Input';
import Button from '../components/ui/Button';
import Logo from '../components/ui/Logo';
import './AuthPages.css';

function passwordStrength(pw) {
  let score = 0;
  if (pw.length >= 8) score++;
  if (/[A-Z]/.test(pw)) score++;
  if (/[a-z]/.test(pw)) score++;
  if (/\d/.test(pw)) score++;
  if (/[!@#$%^&*(),.?":{}|<>]/.test(pw)) score++;
  return score;
}

const STRENGTH_LABELS = ['', 'Weak', 'Fair', 'Good', 'Strong', 'Very strong'];
const STRENGTH_COLORS = ['', '#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#10b981'];

export default function RegisterPage() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    first_name: '',
    last_name: '',
    email: '',
    password: '',
    confirm: '',
    role: 'user',
  });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState('');
  const strength = passwordStrength(form.password);

  function validate() {
    const e = {};
    if (!form.first_name.trim()) e.first_name = 'First name required.';
    if (!form.last_name.trim())  e.last_name  = 'Last name required.';
    if (!form.email || !/\S+@\S+\.\S+/.test(form.email)) e.email = 'Valid email required.';
    if (strength < 4) e.password = 'Password needs uppercase, lowercase, number and special character (min 8 chars).';
    if (form.password !== form.confirm) e.confirm = 'Passwords do not match.';
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setServerError('');
    if (!validate()) return;
    setLoading(true);
    try {
      await registerUser({
        first_name: form.first_name.trim(),
        last_name: form.last_name.trim(),
        email: form.email,
        password: form.password,
        role: form.role,
      });
      await sendOtp(form.email);
      navigate(`/verify-otp?email=${encodeURIComponent(form.email)}&intent=register&pw=${encodeURIComponent(form.password)}`);
    } catch (err) {
      setServerError(extractError(err));
    } finally {
      setLoading(false);
    }
  }

  function field(key, val) {
    setForm((f) => ({ ...f, [key]: val }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: '' }));
  }

  return (
    <div className="auth-shell">
      <div className="auth-hero" aria-hidden="true">
        <div className="auth-hero-grid" />
        <div className="auth-hero-content">
          <Logo size="xl" showText />
          <p className="auth-hero-tagline">
            Predict churn before it happens.<br />
            Retain high-value customers with automated ML playbooks.
          </p>
          <div className="auth-hero-stats">
            {[
              { v: 'Free', l: 'Forever Access' },
              { v: 'Real-time', l: 'ML Scoring' },
              { v: 'Instant', l: 'Email Verification' },
            ].map(({ v, l }) => (
              <div key={l} className="auth-stat">
                <span className="auth-stat-value">{v}</span>
                <span className="auth-stat-label">{l}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="auth-panel">
        <div className="auth-form-wrapper anim-fade-up">
          <div className="auth-mobile-logo"><Logo size="md" showText /></div>
          <h1 className="auth-heading">Create account</h1>
          <p className="auth-subheading">Start your Retentrix intelligence workspace</p>
          {serverError && <div className="auth-alert auth-alert--error" role="alert">{serverError}</div>}
          <form onSubmit={handleSubmit} noValidate className="auth-form">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <Input
                label="First name"
                type="text"
                placeholder="Jane"
                value={form.first_name}
                onChange={(e) => field('first_name', e.target.value)}
                error={errors.first_name}
                required
              />
              <Input
                label="Last name"
                type="text"
                placeholder="Doe"
                value={form.last_name}
                onChange={(e) => field('last_name', e.target.value)}
                error={errors.last_name}
                required
              />
            </div>
            <Input
              label="Work email"
              type="email"
              placeholder="you@company.com"
              value={form.email}
              onChange={(e) => field('email', e.target.value)}
              error={errors.email}
              required
              leftIcon={
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <rect x="1" y="3" width="14" height="10" rx="2" stroke="currentColor" strokeWidth="1.4" />
                  <path d="M1 5.5l7 4.5 7-4.5" stroke="currentColor" strokeWidth="1.4" />
                </svg>
              }
            />
            <div>
              <Input
                label="Password"
                type="password"
                placeholder="Min. 8 chars, 1 uppercase, 1 special"
                value={form.password}
                onChange={(e) => field('password', e.target.value)}
                error={errors.password}
                required
                leftIcon={
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <rect x="3" y="7" width="10" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
                    <path d="M5 7V5a3 3 0 016 0v2" stroke="currentColor" strokeWidth="1.4" />
                  </svg>
                }
              />
              {form.password && (
                <>
                  <div className="pw-strength">
                    <div
                      className="pw-strength-bar"
                      style={{ width: `${(strength / 5) * 100}%`, background: STRENGTH_COLORS[strength] }}
                    />
                  </div>
                  <p className="pw-strength-text" style={{ color: STRENGTH_COLORS[strength] }}>
                    {STRENGTH_LABELS[strength]}
                  </p>
                </>
              )}
            </div>
            <Input
              label="Confirm password"
              type="password"
              placeholder="Repeat password"
              value={form.confirm}
              onChange={(e) => field('confirm', e.target.value)}
              error={errors.confirm}
              required
            />
            <Button type="submit" variant="primary" size="lg" fullWidth loading={loading}>
              Create account & verify email
            </Button>
          </form>
          <p className="auth-footer-text">
            Already have an account? <Link to="/login">Sign in</Link>
          </p>
        </div>
      </div>
    </div>
  );
}
