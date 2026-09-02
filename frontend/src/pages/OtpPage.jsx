import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { sendOtp, verifyOtp, loginUser, getMe } from '../api/auth';
import { extractError } from '../api/client';
import { useAuth } from '../context/AuthContext';
import OtpInput from '../components/auth/OtpInput';
import Input from '../components/ui/Input';
import Button from '../components/ui/Button';
import Logo from '../components/ui/Logo';
import './AuthPages.css';

const RESEND_COOLDOWN = 60;

export default function OtpPage() {
  const [params] = useSearchParams();
  const urlEmail = params.get('email') || '';
  const intent = params.get('intent') || 'login';
  const pw = params.get('pw') || '';
  const { saveSession } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState(urlEmail);
  const [emailSubmitted, setEmailSubmitted] = useState(!!urlEmail);
  const [otp, setOtp] = useState('');
  const [otpError, setOtpError] = useState('');
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState('');
  const [success, setSuccess] = useState('');
  const [countdown, setCountdown] = useState(RESEND_COOLDOWN);

  useEffect(() => {
    if (countdown <= 0) return;
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [countdown]);

  async function handleSendEmail(e) {
    e.preventDefault();
    setServerError('');
    setOtpError('');
    if (!email || !/\S+@\S+\.\S+/.test(email)) {
      setServerError('Please enter a valid email address.');
      return;
    }
    setLoading(true);
    try {
      await sendOtp(email.trim());
      setEmailSubmitted(true);
      setCountdown(RESEND_COOLDOWN);
      setSuccess(`Verification code dispatched to ${email}`);
    } catch (err) {
      setServerError(extractError(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleVerify(e) {
    e.preventDefault();
    setOtpError('');
    setServerError('');
    if (otp.replace(/\s/g, '').length < 6) {
      setOtpError('Please enter the complete 6-digit verification code.');
      return;
    }
    setLoading(true);
    try {
      const res = await verifyOtp(email.trim(), otp.trim());
      // Auto login if backend returned access token
      if (res.access_token) {
        localStorage.setItem('retentrix_token', res.access_token);
        const user = await getMe();
        saveSession(res.access_token, user);
        navigate('/dashboard', { replace: true });
        return;
      }
      // Registration flow with saved password
      if (intent === 'register' && pw) {
        const tokenData = await loginUser({ email: email.trim(), password: decodeURIComponent(pw) });
        localStorage.setItem('retentrix_token', tokenData.access_token);
        const user = await getMe();
        saveSession(tokenData.access_token, user);
        navigate('/dashboard', { replace: true });
        return;
      }
      if (intent === 'change-pw') {
        setSuccess('Email verified! Redirecting...');
        setTimeout(() => navigate(`/settings?verified=true&email=${encodeURIComponent(email)}`), 1000);
        return;
      }
      setSuccess('Email verified successfully! Redirecting to login...');
      setTimeout(() => navigate('/login'), 1200);
    } catch (err) {
      setOtpError(extractError(err));
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    setServerError('');
    setCountdown(RESEND_COOLDOWN);
    try {
      await sendOtp(email.trim());
      setSuccess('A fresh 6-digit code has been dispatched to your email.');
    } catch (err) {
      setServerError(extractError(err));
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
      <div className="auth-form-wrapper anim-scale-pop" style={{ maxWidth: 440, width: '100%' }}>
        <div style={{ textAlign: 'center', marginBottom: 28 }}>
          <Logo size="md" showText />
        </div>

        <div className="card-raised otp-step">
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: '50%',
              background: 'rgba(79,70,229,0.1)',
              border: '1px solid var(--border-accent)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
              <path d="M13 2L3 7v7c0 5.5 4.3 10.7 10 12 5.7-1.3 10-6.5 10-12V7L13 2z" stroke="#818cf8" strokeWidth="1.8" fill="rgba(79,70,229,0.12)" strokeLinejoin="round" />
              <path d="M9 13l3 3 5-5" stroke="#818cf8" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>

          <div>
            <h2 style={{ margin: 0, fontSize: 'var(--text-xl)', fontWeight: 700 }}>Security Verification</h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: 'var(--text-sm)', margin: '8px 0 0' }}>
              {emailSubmitted ? 'Enter the 6-digit one-time passcode sent to' : 'Enter your registered email to receive an OTP code'}
            </p>
            {emailSubmitted && <div className="otp-email-display">{email}</div>}
          </div>

          {serverError && <div className="auth-alert auth-alert--error" style={{ width: '100%', boxSizing: 'border-box' }}>{serverError}</div>}
          {success && <div className="auth-alert auth-alert--success" style={{ width: '100%', boxSizing: 'border-box' }}>{success}</div>}

          {!emailSubmitted ? (
            <form onSubmit={handleSendEmail} style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 14 }}>
              <Input
                label="Registered Email"
                type="email"
                placeholder="you@company.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                autoFocus
              />
              <Button type="submit" variant="primary" size="lg" fullWidth loading={loading}>
                Send Verification Code
              </Button>
            </form>
          ) : (
            <form onSubmit={handleVerify} style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: 16 }}>
              <OtpInput value={otp} onChange={setOtp} disabled={loading} error={otpError} />
              <Button type="submit" variant="primary" size="lg" fullWidth loading={loading} style={{ marginTop: 8 }}>
                Verify & Enter Workspace
              </Button>
            </form>
          )}

          {emailSubmitted && (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              {countdown > 0 ? (
                <p className="otp-countdown">Resend new code in {countdown}s</p>
              ) : (
                <button className="otp-resend-btn" onClick={handleResend} type="button">
                  Resend verification code
                </button>
              )}
              <button
                type="button"
                style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: 'var(--text-xs)', cursor: 'pointer', padding: 0 }}
                onClick={() => {
                  setEmailSubmitted(false);
                  setOtp('');
                }}
              >
                ← Change email address
              </button>
            </div>
          )}

          <Link to="/login" style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
            ← Back to sign in
          </Link>
        </div>
      </div>
    </div>
  );
}
