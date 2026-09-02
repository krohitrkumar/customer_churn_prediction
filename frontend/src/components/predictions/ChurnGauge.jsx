import './ChurnGauge.css';

/**
 * Pure SVG arc gauge — handles both 0.0-1.0 and 0-100 probability values safely.
 */
export default function ChurnGauge({ probability = 0, size = 200 }) {
  const raw = typeof probability === 'number' ? probability : parseFloat(probability) || 0;
  // If > 1.0 (e.g. 20.79), normalize to 0.0 - 1.0 (0.2079)
  const pct = raw > 1.0 ? Math.min(100, Math.max(0, raw)) / 100 : Math.max(0, Math.min(1, raw));
  const percent = Math.round(pct * 100);

  const cx = size / 2;
  const cy = size / 2 + 10;
  const R = size / 2 - 22;

  // Arc covers 220° (from 200° to 340° going clockwise)
  const startAngle = 200;
  const sweepDeg = 220;
  const circumference = 2 * Math.PI * R;
  const arcLen = (sweepDeg / 360) * circumference;
  const offset = arcLen - pct * arcLen;

  const color = pct < 0.35 ? '#10b981' : pct < 0.7 ? '#f59e0b' : '#ef4444';
  const label = pct < 0.35 ? 'Low Risk' : pct < 0.7 ? 'Moderate' : 'Critical';

  function polarToCartesian(centerX, centerY, r, deg) {
    const rad = ((deg - 90) * Math.PI) / 180;
    return { x: centerX + r * Math.cos(rad), y: centerY + r * Math.sin(rad) };
  }

  function arcPath(centerX, centerY, r, startDeg, endDeg) {
    const s = polarToCartesian(centerX, centerY, r, startDeg);
    const e = polarToCartesian(centerX, centerY, r, endDeg);
    const large = endDeg - startDeg > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
  }

  const endAngle = startAngle + sweepDeg;
  const trackPath = arcPath(cx, cy, R, startAngle, endAngle);
  const fillPath = arcPath(cx, cy, R, startAngle, startAngle + pct * sweepDeg);

  return (
    <div className="gauge-wrapper" style={{ width: size, height: size }}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={`Churn risk: ${percent}% — ${label}`}
      >
        {/* Track */}
        <path d={trackPath} fill="none" stroke="var(--border-default)" strokeWidth="14" strokeLinecap="round" />

        {/* Fill arc */}
        <path
          d={fillPath}
          fill="none"
          stroke={color}
          strokeWidth="14"
          strokeLinecap="round"
          className="gauge-arc"
          style={{
            filter: `drop-shadow(0 0 8px ${color}55)`,
            '--gauge-offset': offset,
            strokeDasharray: arcLen,
            strokeDashoffset: offset,
          }}
        />

        {/* Center percent */}
        <text
          x={cx}
          y={cy - 8}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={size * 0.18}
          fontWeight="800"
          fill={color}
          fontFamily="var(--font-sans)"
        >
          {percent}%
        </text>
        {/* Label */}
        <text
          x={cx}
          y={cy + size * 0.13}
          textAnchor="middle"
          dominantBaseline="middle"
          fontSize={size * 0.075}
          fontWeight="500"
          fill="var(--text-muted)"
          fontFamily="var(--font-sans)"
        >
          {label}
        </text>
      </svg>
    </div>
  );
}
