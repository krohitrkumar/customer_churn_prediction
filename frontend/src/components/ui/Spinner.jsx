import './Spinner.css';
export default function Spinner({ size = 'md', label = 'Loading...' }) {
  const sizes = { xs: 16, sm: 20, md: 28, lg: 40, xl: 56 };
  const px = sizes[size] ?? 28;
  const stroke = px < 24 ? 2 : 2.5;
  const r = (px / 2) - stroke - 1;
  const circ = 2 * Math.PI * r;
  return (
    <span className="spinner" role="status" aria-label={label} style={{ width: px, height: px }}>
      <svg viewBox={`0 0 ${px} ${px}`} fill="none" style={{ width: px, height: px }}>
        <circle cx={px/2} cy={px/2} r={r} stroke="var(--border-default)" strokeWidth={stroke} />
        <circle cx={px/2} cy={px/2} r={r} stroke="var(--brand-primary)" strokeWidth={stroke}
          strokeLinecap="round" strokeDasharray={`${circ * 0.75} ${circ * 0.25}`} className="spinner-arc" />
      </svg>
      <span className="sr-only">{label}</span>
    </span>
  );
}
