import { useState } from 'react';
import './RiskPieChart.css';

/**
 * Modern Interactive SVG Donut Chart with glowing slices, hover inspection & responsive space utilization.
 */
export default function RiskPieChart({ data = [], size = 220 }) {
  const [activeIdx, setActiveIdx] = useState(null);

  // Filter items with value > 0
  const validData = data.filter((d) => d.value > 0);
  const total = validData.reduce((s, d) => s + d.value, 0);

  if (!total) {
    return (
      <div className="pie-empty-state">
        <span style={{ fontSize: 32 }}>📊</span>
        <p>No customer risk data available to visualize.</p>
      </div>
    );
  }

  const cx = size / 2;
  const cy = size / 2;
  const strokeWidth = 28;
  const R = size / 2 - strokeWidth;
  const innerR = R - strokeWidth / 2;
  const outerR = R + strokeWidth / 2;

  // Calculate slice geometry
  let cumulativeAngle = 0;
  const slices = validData.map((d, index) => {
    const fraction = d.value / total;
    const angle = fraction * 360;
    const startAngle = cumulativeAngle;
    const endAngle = cumulativeAngle + angle;
    cumulativeAngle += angle;

    const pct = Math.round(fraction * 100);
    return {
      ...d,
      index,
      pct,
      startAngle,
      endAngle,
      fraction,
    };
  });

  function polarToCart(centerX, centerY, radius, deg) {
    const rad = ((deg - 90) * Math.PI) / 180;
    return {
      x: centerX + radius * Math.cos(rad),
      y: centerY + radius * Math.sin(rad),
    };
  }

  function describeDonutSlice(startAngle, endAngle, isHovered) {
    // If only 1 slice takes 100% (360 deg), adjust slightly to prevent degenerate arc
    const adjustedEnd = endAngle - startAngle >= 359.9 ? startAngle + 359.99 : endAngle;
    
    // Add subtle expansion when hovered
    const oR = isHovered ? outerR + 4 : outerR;
    const iR = isHovered ? innerR - 2 : innerR;

    // Small angular gap for separation
    const gap = slices.length > 1 ? 1.2 : 0;
    const sA = startAngle + gap;
    const eA = adjustedEnd - gap;

    const os = polarToCart(cx, cy, oR, sA);
    const oe = polarToCart(cx, cy, oR, eA);
    const is = polarToCart(cx, cy, iR, eA);
    const ie = polarToCart(cx, cy, iR, sA);

    const largeArc = eA - sA > 180 ? 1 : 0;

    return `M ${os.x} ${os.y} A ${oR} ${oR} 0 ${largeArc} 1 ${oe.x} ${oe.y} L ${is.x} ${is.y} A ${iR} ${iR} 0 ${largeArc} 0 ${ie.x} ${ie.y} Z`;
  }

  const activeItem = activeIdx !== null ? slices[activeIdx] : null;

  return (
    <div className="donut-container">
      {/* SVG Donut Graphic */}
      <div className="donut-chart-box">
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="donut-svg"
          role="img"
          aria-label="Interactive customer risk distribution chart"
        >
          <defs>
            {slices.map((s, i) => (
              <linearGradient key={`grad-${i}`} id={`donut-grad-${i}`} x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor={s.color} stopOpacity="1" />
                <stop offset="100%" stopColor={s.color} stopOpacity="0.8" />
              </linearGradient>
            ))}
            <filter id="donut-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="4" stdDeviation="6" floodColor="rgba(0,0,0,0.3)" />
            </filter>
          </defs>

          {/* Slices */}
          <g filter="url(#donut-glow)">
            {slices.map((s, i) => {
              const isHovered = activeIdx === i;
              return (
                <path
                  key={i}
                  d={describeDonutSlice(s.startAngle, s.endAngle, isHovered)}
                  fill={`url(#donut-grad-${i})`}
                  className={`donut-slice ${isHovered ? 'donut-slice--active' : ''}`}
                  onMouseEnter={() => setActiveIdx(i)}
                  onMouseLeave={() => setActiveIdx(null)}
                  style={{
                    transformOrigin: `${cx}px ${cy}px`,
                    cursor: 'pointer',
                  }}
                >
                  <title>{`${s.label}: ${s.value} (${s.pct}%)`}</title>
                </path>
              );
            })}
          </g>

          {/* Center Metric Text */}
          <g className="donut-center-group" pointerEvents="none">
            {activeItem ? (
              <>
                <text
                  x={cx}
                  y={cy - 12}
                  textAnchor="middle"
                  className="donut-center-num"
                  fill={activeItem.color}
                >
                  {activeItem.value}
                </text>
                <text
                  x={cx}
                  y={cy + 8}
                  textAnchor="middle"
                  className="donut-center-label"
                >
                  {activeItem.label}
                </text>
                <text
                  x={cx}
                  y={cy + 24}
                  textAnchor="middle"
                  className="donut-center-sub"
                >
                  {activeItem.pct}% of total
                </text>
              </>
            ) : (
              <>
                <text
                  x={cx}
                  y={cy - 6}
                  textAnchor="middle"
                  className="donut-center-num"
                >
                  {total}
                </text>
                <text
                  x={cx}
                  y={cy + 14}
                  textAnchor="middle"
                  className="donut-center-label"
                >
                  Accounts
                </text>
                <text
                  x={cx}
                  y={cy + 28}
                  textAnchor="middle"
                  className="donut-center-sub"
                >
                  Hover to inspect
                </text>
              </>
            )}
          </g>
        </svg>
      </div>

      {/* Modern High-Impact Legend Cards */}
      <div className="donut-legend-list">
        {slices.map((s, i) => {
          const isHovered = activeIdx === i;
          return (
            <div
              key={i}
              className={`donut-legend-row ${isHovered ? 'donut-legend-row--active' : ''}`}
              onMouseEnter={() => setActiveIdx(i)}
              onMouseLeave={() => setActiveIdx(null)}
              role="button"
              tabIndex={0}
            >
              <div className="donut-legend-header">
                <div className="donut-legend-title">
                  <span
                    className="donut-legend-dot"
                    style={{
                      background: s.color,
                      boxShadow: isHovered ? `0 0 10px ${s.color}` : 'none',
                    }}
                  />
                  <span>{s.label}</span>
                </div>
                <div className="donut-legend-stats">
                  <span className="donut-legend-count">{s.value}</span>
                  <span className="donut-legend-pct font-mono">({s.pct}%)</span>
                </div>
              </div>

              {/* Progress bar line */}
              <div className="donut-bar-track">
                <div
                  className="donut-bar-fill"
                  style={{
                    width: `${s.pct}%`,
                    background: s.color,
                    boxShadow: isHovered ? `0 0 8px ${s.color}99` : 'none',
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
