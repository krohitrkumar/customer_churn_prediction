import './Button.css';

/**
 * Retentrix Button — fully custom, no library base
 * Props: variant, size, loading, leftIcon, rightIcon, fullWidth, ...rest
 */
export default function Button({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  leftIcon,
  rightIcon,
  fullWidth = false,
  className = '',
  disabled,
  ...rest
}) {
  return (
    <button
      className={[
        'btn',
        `btn--${variant}`,
        `btn--${size}`,
        fullWidth ? 'btn--full' : '',
        loading   ? 'btn--loading' : '',
        className,
      ].filter(Boolean).join(' ')}
      disabled={disabled || loading}
      aria-busy={loading}
      {...rest}
    >
      {loading ? (
        <span className="btn-spinner" aria-hidden="true" />
      ) : leftIcon ? (
        <span className="btn-icon btn-icon--left" aria-hidden="true">{leftIcon}</span>
      ) : null}
      <span className="btn-label">{children}</span>
      {!loading && rightIcon && (
        <span className="btn-icon btn-icon--right" aria-hidden="true">{rightIcon}</span>
      )}
    </button>
  );
}
