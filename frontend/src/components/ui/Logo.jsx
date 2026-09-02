import './Logo.css';
export default function Logo({ size = 'md', showText = true }) {
  const sizes = { xs: 24, sm: 28, md: 32, lg: 40, xl: 56 };
  const px = sizes[size] ?? 32;
  return (
    <span className={`logo logo--${size}`} aria-label="Retentrix">
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 40 40" width={px} height={px}
        fill="none" aria-hidden="true" draggable="false" className="logo-mark">
        <rect width="40" height="40" rx="10" fill="url(#lg)"/>
        <circle cx="12" cy="28" r="2" fill="rgba(255,255,255,0.2)"/>
        <circle cx="20" cy="28" r="2" fill="rgba(255,255,255,0.2)"/>
        <circle cx="28" cy="28" r="2" fill="rgba(255,255,255,0.2)"/>
        <polyline points="10,26 16,16 22,21 28,11" stroke="white" strokeWidth="2.5"
          strokeLinecap="round" strokeLinejoin="round" fill="none" opacity="0.9"/>
        <circle cx="28" cy="11" r="3" fill="#10b981"/>
        <defs>
          <linearGradient id="lg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
            <stop offset="0%" stopColor="#3730a3"/>
            <stop offset="100%" stopColor="#4f46e5"/>
          </linearGradient>
        </defs>
      </svg>
      {showText && <span className="logo-wordmark">Retentrix</span>}
    </span>
  );
}
