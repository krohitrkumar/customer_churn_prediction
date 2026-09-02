import { useState } from 'react';
import './Input.css';

export default function Input({
  label,
  error,
  hint,
  leftIcon,
  rightIcon,
  id,
  type = 'text',
  className = '',
  required,
  ...rest
}) {
  const [showPassword, setShowPassword] = useState(false);
  const inputId = id || `input-${Math.random().toString(36).slice(2, 7)}`;
  const isPassword = type === 'password';
  const effectiveType = isPassword ? (showPassword ? 'text' : 'password') : type;

  return (
    <div className={`field ${className}`}>
      {label && (
        <label className="field-label" htmlFor={inputId}>
          {label}
          {required && <span className="field-required" aria-hidden="true">*</span>}
        </label>
      )}
      <div className={`field-wrap ${leftIcon ? 'has-left-icon' : ''} ${rightIcon || isPassword ? 'has-right-icon' : ''} ${error ? 'field-wrap--error' : ''}`}>
        {leftIcon && <span className="field-icon field-icon--left" aria-hidden="true">{leftIcon}</span>}
        <input
          id={inputId}
          type={effectiveType}
          className="field-input"
          aria-invalid={!!error}
          aria-describedby={error ? `${inputId}-err` : hint ? `${inputId}-hint` : undefined}
          {...rest}
        />
        {isPassword ? (
          <button
            type="button"
            className="field-icon-btn field-icon--right"
            onClick={() => setShowPassword((s) => !s)}
            aria-label={showPassword ? 'Hide password' : 'Show password'}
            tabIndex={-1}
          >
            {showPassword ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                <line x1="1" y1="1" x2="23" y2="23" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            )}
          </button>
        ) : (
          rightIcon && <span className="field-icon field-icon--right" aria-hidden="true">{rightIcon}</span>
        )}
      </div>
      {error && <p id={`${inputId}-err`} className="field-error" role="alert">{error}</p>}
      {!error && hint && <p id={`${inputId}-hint`} className="field-hint">{hint}</p>}
    </div>
  );
}
