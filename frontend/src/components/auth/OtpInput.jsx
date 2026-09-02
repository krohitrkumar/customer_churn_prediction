import { useRef, useEffect } from 'react';
import './OtpInput.css';

const OTP_LENGTH = 6;

export default function OtpInput({ value = '', onChange, disabled = false, error }) {
  const refs = useRef([]);

  // Generate array of exactly 6 slots from value string
  const digits = Array.from({ length: OTP_LENGTH }, (_, i) => value[i] || '');

  // Auto-focus first input on mount
  useEffect(() => {
    if (!disabled && refs.current[0]) {
      refs.current[0].focus();
    }
  }, [disabled]);

  function update(index, char) {
    const nextArr = [...digits];
    nextArr[index] = char;
    const newVal = nextArr.join('').trim();
    onChange(newVal);
  }

  function handleKey(e, i) {
    if (e.key === 'Backspace') {
      if (digits[i]) {
        update(i, '');
      } else if (i > 0) {
        update(i - 1, '');
        refs.current[i - 1]?.focus();
      }
    } else if (e.key === 'ArrowLeft' && i > 0) {
      refs.current[i - 1]?.focus();
    } else if (e.key === 'ArrowRight' && i < OTP_LENGTH - 1) {
      refs.current[i + 1]?.focus();
    }
  }

  function handleChange(e, i) {
    const raw = e.target.value.replace(/\D/g, '');
    const char = raw.slice(-1);
    update(i, char);
    if (char && i < OTP_LENGTH - 1) {
      refs.current[i + 1]?.focus();
    }
  }

  function handlePaste(e) {
    e.preventDefault();
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, OTP_LENGTH);
    if (pasted) {
      onChange(pasted);
      const targetIdx = Math.min(pasted.length, OTP_LENGTH - 1);
      refs.current[targetIdx]?.focus();
    }
  }

  return (
    <div className={`otp-root ${error ? 'otp-root--error' : ''}`}>
      <div className="otp-boxes" role="group" aria-label="6-digit verification code">
        {digits.map((digit, i) => (
          <input
            key={i}
            ref={(el) => { refs.current[i] = el; }}
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            maxLength={1}
            value={digit}
            disabled={disabled}
            aria-label={`Digit ${i + 1} of ${OTP_LENGTH}`}
            className={`otp-box ${digit ? 'otp-box--filled' : ''}`}
            onChange={(e) => handleChange(e, i)}
            onKeyDown={(e) => handleKey(e, i)}
            onPaste={handlePaste}
            onFocus={(e) => e.target.select()}
            autoComplete={i === 0 ? 'one-time-code' : 'off'}
          />
        ))}
      </div>
      {error && <p className="otp-error" role="alert">{error}</p>}
    </div>
  );
}
