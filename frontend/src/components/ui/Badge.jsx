import './Badge.css';
export default function Badge({ children, variant = 'default', size = 'md', dot = false }) {
  return (
    <span className={`badge badge--${variant} badge--${size}`}>
      {dot && <span className="badge-dot" aria-hidden="true" />}
      {children}
    </span>
  );
}
