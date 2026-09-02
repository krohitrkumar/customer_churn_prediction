import { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { loginUser, sendOtp, verifyOtp, resetPasswordWithOtp, getMe } from '../api/auth';
import { extractError } from '../api/client';
import Input from '../components/ui/Input';
import Button from '../components/ui/Button';
import Logo from '../components/ui/Logo';
import Modal from '../components/ui/Modal';
import OtpInput from '../components/auth/OtpInput';
import './AuthPages.css';

export default function LoginPage() {
  const { saveSession } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = location.state?.from?.pathname ?? '/dashboard';

  const sessionExpired = new URLSearchParams(location.search).get('session') === 'expired';

  // Standard Login State
  const [form, setForm] = useState({ email: '', password: '' });
  const [errors, setErrors] = useState({});
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState('');

  // OTP Login Modal State
  const [otpModalOpen, setOtpModalOpen] = useState(false);
  const [otpEmail, setOtpEmail] = useState('');
  const [otpCode, setOtpCode] = useState('');
  const [otpStep, setOtpStep] = useState('email'); // 'email' | 'code'
  const [otpLoading, setOtpLoading] = useState(false);
  const [otpError, setOtpError] = useState('');
  const [otpSuccess, setOtpSuccess] = useState('');

  // Forgot Password Modal State
  const [forgotModalOpen, setForgotModalOpen] = useState(false);
  const [forgotEmail, setForgotEmail] = useState('');
  const [forgotOtp, setForgotOtp] = useState('');
  const [forgotNewPw, setForgotNewPw] = useState('');
  const [forgotStep, setForgotStep] = useState('email'); // 'email' | 'reset'
  const [forgotLoading, setForgotLoading] = useState(false);
  const [forgotError, setForgotError] = useState('');
  const [forgotSuccess, setForgotSuccess] = useState('');

  function validate() {
    const e = {};
    if (!form.email) e.email = 'Email is required.';
    else if (!/\S+@\S+\.\S+/.test(form.email)) e.email = 'Enter a valid email address.';
    if (!form.password) e.password = 'Password is required.';
    setErrors(e);
    return Object.keys(e).length === 0;
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setServerError('');
    if (!validate()) return;
    setLoading(true);
    try {
      const tokenData = await loginUser({ email: form.email, password: form.password });
      localStorage.setItem('retentrix_token', tokenData.access_token);
      const user = await getMe();
      saveSession(tokenData.access_token, user);
      navigate(from, { replace: true });
    } catch (err) {
      setServerError(extractError(err));
    } finally {
      setLoading(false);
    }
  }

  // OTP Sign-in: Step 1 (Send Code)
  async function handleSendOtpLogin(e) {
    e.preventDefault();
    setOtpError('');
    if (!otpEmail || !/\S+@\S+\.\S+/.test(otpEmail)) {
      setOtpError('Please enter a valid email address.');
      return;
    }
    setOtpLoading(true);
    try {
      await sendOtp(otpEmail.trim());
      setOtpSuccess(`Verification code sent to ${otpEmail}`);
      setOtpStep('code');
    } catch (err) {
      setOtpError(extractError(err));
    } finally {
      setOtpLoading(false);
    }
  }

  // OTP Sign-in: Step 2 (Verify & Sign In)
  async function handleVerifyOtpLogin(e) {
    e.preventDefault();
    setOtpError('');
    if (otpCode.length < 6) {
      setOtpError('Please enter the complete 6-digit code.');
      return;
    }
    setOtpLoading(true);
    try {
      const res = await verifyOtp(otpEmail.trim(), otpCode.trim());
      if (res.access_token) {
        localStorage.setItem('retentrix_token', res.access_token);
        const user = await getMe();
        saveSession(res.access_token, user);
        setOtpModalOpen(false);
        navigate(from, { replace: true });
      } else {
        setOtpError('Could not authenticate. Please try signing in with password.');
      }
    } catch (err) {
      setOtpError(extractError(err));
    } finally {
      setOtpLoading(false);
    }
  }

  // Forgot Password: Step 1 (Send Reset OTP)
  async function handleSendForgotOtp(e) {
    e.preventDefault();
    setForgotError('');
    if (!forgotEmail || !/\S+@\S+\.\S+/.test(forgotEmail)) {
      setForgotError('Please enter a valid email address.');
      return;
    }
    setForgotLoading(true);
    try {
      await sendOtp(forgotEmail.trim());
      setForgotSuccess(`Reset code sent to ${forgotEmail}`);
      setForgotStep('reset');
    } catch (err) {
      setForgotError(extractError(err));
    } finally {
      setForgotLoading(false);
    }
  }

  // Forgot Password: Step 2 (Reset with OTP + New Password)
  async function handleResetPasswordSubmit(e) {
    e.preventDefault();
    setForgotError('');
    if (forgotOtp.length < 6) {
      setForgotError('Please enter the 6-digit OTP code.');
      return;
    }
    if (!forgotNewPw || forgotNewPw.length < 8) {
      setForgotError('New password must be at least 8 characters long.');
      return;
    }
    setForgotLoading(true);
    try {
      await resetPasswordWithOtp(forgotEmail.trim(), forgotOtp.trim(), forgotNewPw);
      setForgotSuccess('Password reset successfully! You can now sign in with your new password.');
      setTimeout(() => {
        setForgotModalOpen(false);
        setForm((f) => ({ ...f, email: forgotEmail, password: '' }));
      }, 1500);
    } catch (err) {
      setForgotError(extractError(err));
    } finally {
      setForgotLoading(false);
    }
  }

  return (
    <div className="auth-shell">
      {/* Left hero banner */}
      <div className="auth-hero" aria-hidden="true">
        <div className="auth-hero-grid" />
        <div className="auth-hero-content">
          <Logo size="xl" showText />
          <p className="auth-hero-tagline">
            AI-powered customer churn intelligence.<br />
            Predict. Retain. Grow.
          </p>
          <div className="auth-hero-stats">
            {[
              { v: '94%', l: 'Prediction Accuracy' },
              { v: '3×', l: 'Retention Lift' },
              { v: '60s', l: 'Insight Latency' },
            ].map(({ v, l }) => (
              <div key={l} className="auth-stat">
                <span className="auth-stat-value">{v}</span>
                <span className="auth-stat-label">{l}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right sign-in form */}
      <div className="auth-panel">
        <div className="auth-form-wrapper anim-fade-up">
          <div className="auth-mobile-logo">
            <Logo size="md" showText />
          </div>

          <h1 className="auth-heading">Welcome back</h1>
          <p className="auth-subheading">Sign in to your Retentrix workspace</p>

          {sessionExpired && (
            <div className="auth-alert auth-alert--warning">
              ⚠️ Your session expired. Please sign in again.
            </div>
          )}
          {serverError && (
            <div className="auth-alert auth-alert--error" role="alert">
              {serverError}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="auth-form">
            <Input
              label="Email address"
              type="email"
              id="login-email"
              autoComplete="email"
              placeholder="you@company.com"
              value={form.email}
              onChange={(e) => setForm((f) => ({ ...f, email: e.target.value }))}
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
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 'var(--text-sm)', fontWeight: 500, color: 'var(--text-secondary)' }}>
                  Password <span style={{ color: 'var(--brand-crimson)' }}>*</span>
                </span>
                <button
                  type="button"
                  style={{ background: 'none', border: 'none', color: 'var(--brand-primary-h)', fontSize: 'var(--text-xs)', cursor: 'pointer', padding: 0 }}
                  onClick={() => {
                    setForgotEmail(form.email);
                    setForgotStep('email');
                    setForgotError('');
                    setForgotSuccess('');
                    setForgotModalOpen(true);
                  }}
                >
                  Forgot password?
                </button>
              </div>
              <Input
                type="password"
                id="login-password"
                autoComplete="current-password"
                placeholder="••••••••"
                value={form.password}
                onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
                error={errors.password}
                required
                leftIcon={
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                    <rect x="3" y="7" width="10" height="7" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
                    <path d="M5 7V5a3 3 0 016 0v2" stroke="currentColor" strokeWidth="1.4" />
                  </svg>
                }
              />
            </div>

            <Button type="submit" variant="primary" size="lg" fullWidth loading={loading} style={{ marginTop: 8 }}>
              Sign in
            </Button>
          </form>

          <div className="auth-divider"><span>or continue with</span></div>

          <button
            type="button"
            className="auth-otp-btn"
            onClick={() => {
              setOtpEmail(form.email);
              setOtpStep('email');
              setOtpError('');
              setOtpSuccess('');
              setOtpCode('');
              setOtpModalOpen(true);
            }}
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <rect x="1" y="3" width="16" height="12" rx="2" stroke="currentColor" strokeWidth="1.4" />
              <path d="M1 7h16" stroke="currentColor" strokeWidth="1.4" />
              <path d="M5 11h2M9 11h2M13 11h2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
            Sign in with email OTP
          </button>

          <p className="auth-footer-text">
            Don't have an account?{' '}
            <Link to="/register">Create one free</Link>
          </p>
        </div>
      </div>

      {/* ── Modal: Sign in with Email OTP ── */}
      <Modal
        open={otpModalOpen}
        onOpenChange={(o) => {
          if (!o) setOtpModalOpen(false);
        }}
        title="Sign in with Email OTP"
        description="Passwordless authentication via 6-digit one-time code."
        size="sm"
      >
        {otpError && <div className="auth-alert auth-alert--error" style={{ marginBottom: 14 }}>{otpError}</div>}
        {otpSuccess && <div className="auth-alert auth-alert--success" style={{ marginBottom: 14 }}>{otpSuccess}</div>}

        {otpStep === 'email' ? (
          <form onSubmit={handleSendOtpLogin} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Input
              label="Enter registered email"
              type="email"
              placeholder="you@company.com"
              value={otpEmail}
              onChange={(e) => setOtpEmail(e.target.value)}
              required
              autoFocus
            />
            <Button type="submit" variant="primary" fullWidth loading={otpLoading}>
              Send Verification Code
            </Button>
          </form>
        ) : (
          <form onSubmit={handleVerifyOtpLogin} style={{ display: 'flex', flexDirection: 'column', gap: 18, alignItems: 'center' }}>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', margin: 0 }}>
              Enter the 6-digit code sent to <strong>{otpEmail}</strong>
            </p>
            <OtpInput value={otpCode} onChange={setOtpCode} disabled={otpLoading} error={otpError} />
            <Button type="submit" variant="primary" fullWidth loading={otpLoading}>
              Verify & Sign In
            </Button>
            <button
              type="button"
              className="otp-resend-btn"
              onClick={() => {
                setOtpStep('email');
                setOtpCode('');
              }}
            >
              ← Use different email
            </button>
          </form>
        )}
      </Modal>

      {/* ── Modal: Forgot Password / Passwordless Reset ── */}
      <Modal
        open={forgotModalOpen}
        onOpenChange={(o) => {
          if (!o) setForgotModalOpen(false);
        }}
        title="Reset Password"
        description="Verify your email with an OTP code to set a new password."
        size="sm"
      >
        {forgotError && <div className="auth-alert auth-alert--error" style={{ marginBottom: 14 }}>{forgotError}</div>}
        {forgotSuccess && <div className="auth-alert auth-alert--success" style={{ marginBottom: 14 }}>{forgotSuccess}</div>}

        {forgotStep === 'email' ? (
          <form onSubmit={handleSendForgotOtp} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Input
              label="Account email address"
              type="email"
              placeholder="you@company.com"
              value={forgotEmail}
              onChange={(e) => setForgotEmail(e.target.value)}
              required
              autoFocus
            />
            <Button type="submit" variant="primary" fullWidth loading={forgotLoading}>
              Send Reset Code
            </Button>
          </form>
        ) : (
          <form onSubmit={handleResetPasswordSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <p style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', margin: 0, textAlign: 'center' }}>
              Code sent to <strong>{forgotEmail}</strong>
            </p>
            <OtpInput value={forgotOtp} onChange={setForgotOtp} disabled={forgotLoading} />
            <Input
              label="New Password"
              type="password"
              placeholder="Min. 8 characters (A-Z, a-z, 0-9, special)"
              value={forgotNewPw}
              onChange={(e) => setForgotNewPw(e.target.value)}
              required
            />
            <Button type="submit" variant="primary" fullWidth loading={forgotLoading} style={{ marginTop: 6 }}>
              Set New Password
            </Button>
          </form>
        )}
      </Modal>
    </div>
  );
}
