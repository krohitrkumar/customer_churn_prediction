import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { changePasswordWithOld, sendOtp, resetPasswordWithOtp } from '../api/auth';
import { extractError } from '../api/client';
import { useToastState } from '../hooks/useToast';
import Input from '../components/ui/Input';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import ToastContainer from '../components/ui/Toast';
import './SettingsPage.css';

function passwordStrength(pw) {
  let s = 0;
  if (pw.length >= 8) s++;
  if (/[A-Z]/.test(pw)) s++;
  if (/[a-z]/.test(pw)) s++;
  if (/\d/.test(pw)) s++;
  if (/[!@#$%^&*(),.?":{}|<>]/.test(pw)) s++;
  return s;
}
const STR_LABELS = ['', 'Weak', 'Fair', 'Good', 'Strong', 'Very strong'];
const STR_COLORS = ['', '#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#10b981'];

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const verifiedEmail = params.get('email');
  const { toasts, success, error: toastError, dismiss } = useToastState();

  // Change password via old password
  const [pwForm, setPwForm] = useState({ current: '', new: '', confirm: '' });
  const [pwErrors, setPwErrors] = useState({});
  const [pwLoading, setPwLoading] = useState(false);
  const strength = passwordStrength(pwForm.new);

  // OTP-based password flow
  const [otpLoading, setOtpLoading] = useState(false);

  function validatePw() {
    const e = {};
    if (!pwForm.current) e.current = 'Current password required.';
    if (strength < 4) e.new = 'Password needs uppercase, lowercase, number and special character.';
    if (pwForm.new !== pwForm.confirm) e.confirm = 'Passwords do not match.';
    setPwErrors(e);
    return Object.keys(e).length === 0;
  }

  async function handleChangePw(e) {
    e.preventDefault();
    if (!validatePw()) return;
    setPwLoading(true);
    try {
      await changePasswordWithOld(pwForm.current, pwForm.new);
      success('Password changed', 'Your password has been updated successfully.');
      setPwForm({ current: '', new: '', confirm: '' });
    } catch (err) {
      toastError('Failed', extractError(err));
    } finally {
      setPwLoading(false);
    }
  }

  async function handleOtpPwReset() {
    setOtpLoading(true);
    try {
      await sendOtp(user?.email);
      navigate(`/verify-otp?email=${encodeURIComponent(user?.email)}&intent=change-pw`);
    } catch (err) {
      toastError('OTP send failed', extractError(err));
    } finally {
      setOtpLoading(false);
    }
  }

  function pw(key, val) {
    setPwForm((p) => ({ ...p, [key]: val }));
    if (pwErrors[key]) setPwErrors((e) => ({ ...e, [key]: '' }));
  }

  const roleBadgeVariant = user?.role === 'admin' ? 'admin' : user?.role === 'csm' ? 'info' : 'default';

  return (
    <div>
      <ToastContainer toasts={toasts} onDismiss={dismiss} />
      <div className="page-header">
        <h1>Settings & Security</h1>
        <p>Manage your profile, password credentials, and security preferences</p>
      </div>

      <div className="settings-layout">
        {/* Profile Card */}
        <section className="card settings-section anim-fade-up">
          <div className="settings-section-header">
            <div className="settings-section-icon">👤</div>
            <div>
              <h2>Profile Details</h2>
              <p>Your authenticated workspace identity</p>
            </div>
          </div>
          <div className="settings-profile">
            <div className="settings-avatar">
              {user?.first_name?.[0]?.toUpperCase()}
              {user?.last_name?.[0]?.toUpperCase()}
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {user?.first_name} {user?.last_name}
                </span>
                <Badge variant={roleBadgeVariant}>{user?.role?.toUpperCase()}</Badge>
              </div>
              <p style={{ margin: '4px 0 0', fontSize: 'var(--text-sm)', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                {user?.email}
              </p>
              <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                Member since{' '}
                {user?.created_at
                  ? new Date(user.created_at).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
                  : 'Active session'}
              </p>
            </div>
          </div>
        </section>

        {/* Change Password — with current password */}
        <section className="card settings-section anim-fade-up delay-1">
          <div className="settings-section-header">
            <div className="settings-section-icon">🔐</div>
            <div>
              <h2>Method 1: Change Password (Using Current Password)</h2>
              <p>Update your password by confirming your current credentials</p>
            </div>
          </div>
          <form onSubmit={handleChangePw} noValidate style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Input
              label="Current password"
              type="password"
              autoComplete="current-password"
              placeholder="••••••••"
              value={pwForm.current}
              onChange={(e) => pw('current', e.target.value)}
              error={pwErrors.current}
              required
            />
            <div>
              <Input
                label="New password"
                type="password"
                autoComplete="new-password"
                placeholder="Min. 8 chars (uppercase, lowercase, digit, symbol)"
                value={pwForm.new}
                onChange={(e) => pw('new', e.target.value)}
                error={pwErrors.new}
                required
              />
              {pwForm.new && (
                <>
                  <div style={{ height: 4, borderRadius: 9999, background: 'var(--border-default)', overflow: 'hidden', marginTop: 6 }}>
                    <div
                      style={{
                        height: '100%',
                        borderRadius: 9999,
                        width: `${(strength / 5) * 100}%`,
                        background: STR_COLORS[strength],
                        transition: 'width 0.3s, background 0.3s',
                      }}
                    />
                  </div>
                  <p style={{ fontSize: 'var(--text-xs)', color: STR_COLORS[strength], marginTop: 4 }}>{STR_LABELS[strength]}</p>
                </>
              )}
            </div>
            <Input
              label="Confirm new password"
              type="password"
              autoComplete="new-password"
              placeholder="Repeat new password"
              value={pwForm.confirm}
              onChange={(e) => pw('confirm', e.target.value)}
              error={pwErrors.confirm}
              required
            />
            <Button type="submit" variant="primary" loading={pwLoading}>
              Update Password
            </Button>
          </form>
        </section>

        {/* Change Password via Email OTP */}
        <section className="card settings-section anim-fade-up delay-2">
          <div className="settings-section-header">
            <div className="settings-section-icon">📧</div>
            <div>
              <h2>Method 2: Passwordless Reset via Email OTP</h2>
              <p>Verify via Gmail one-time passcode — no old password required</p>
            </div>
          </div>
          <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', marginBottom: 16, lineHeight: 1.6 }}>
            Forgot your current password or prefer email verification? We'll dispatch a secure 6-digit verification code to{' '}
            <strong style={{ color: 'var(--text-primary)' }}>{user?.email}</strong>.
          </p>
          <Button
            variant="secondary"
            loading={otpLoading}
            onClick={handleOtpPwReset}
            leftIcon={
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <rect x="1" y="2" width="12" height="10" rx="2" stroke="currentColor" strokeWidth="1.4" />
                <path d="M1 5l6 4 6-4" stroke="currentColor" strokeWidth="1.4" />
              </svg>
            }
          >
            Send OTP Verification to Gmail
          </Button>
        </section>

        {/* Danger Zone */}
        <section className="card settings-section settings-danger anim-fade-up delay-3">
          <div className="settings-section-header">
            <div className="settings-section-icon">⚠️</div>
            <div>
              <h2>Active Session</h2>
              <p>Sign out of your active workspace</p>
            </div>
          </div>
          <Button
            variant="danger"
            onClick={() => {
              logout();
              navigate('/login');
            }}
            leftIcon={
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M5 12H2a1 1 0 01-1-1V3a1 1 0 011-1h3M9 10l3-3-3-3M12 7H5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            }
          >
            Sign out of Retentrix
          </Button>
        </section>
      </div>
    </div>
  );
}
