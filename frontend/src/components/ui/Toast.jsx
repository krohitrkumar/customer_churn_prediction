import { createPortal } from 'react-dom';
import './Toast.css';

const ICONS = {
  success: (<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="9" fill="#10b981" opacity="0.15"/><path d="M5 9l3 3 5-5" stroke="#10b981" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>),
  error:   (<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="9" fill="#ef4444" opacity="0.15"/><path d="M6 6l6 6M12 6l-6 6" stroke="#ef4444" strokeWidth="1.8" strokeLinecap="round"/></svg>),
  warning: (<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="9" fill="#f59e0b" opacity="0.15"/><path d="M9 5v5M9 13v.5" stroke="#f59e0b" strokeWidth="2" strokeLinecap="round"/></svg>),
  info:    (<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><circle cx="9" cy="9" r="9" fill="#3b82f6" opacity="0.15"/><path d="M9 8v5M9 5v.5" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round"/></svg>),
};

function ToastItem({ toast, onDismiss }) {
  return (
    <div className={`toast toast--${toast.type} ${toast.leaving ? 'toast--leaving' : 'toast--entering'}`} role="alert" aria-live="polite">
      <span className="toast-icon">{ICONS[toast.type]}</span>
      <div className="toast-body">
        {toast.title   && <strong className="toast-title">{toast.title}</strong>}
        {toast.message && <p className="toast-message">{toast.message}</p>}
      </div>
      <button className="toast-close" onClick={() => onDismiss(toast.id)} aria-label="Dismiss notification">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M2 2l10 10M12 2L2 12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>
      </button>
    </div>
  );
}

export default function ToastContainer({ toasts, onDismiss }) {
  return createPortal(
    <div className="toast-container" role="region" aria-label="Notifications" aria-live="polite">
      {toasts.map((t) => <ToastItem key={t.id} toast={t} onDismiss={onDismiss} />)}
    </div>,
    document.body
  );
}
