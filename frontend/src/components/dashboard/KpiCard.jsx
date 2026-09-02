import './KpiCard.css';
import Skeleton from '../ui/Skeleton';

export default function KpiCard({ label, value, sub, icon, color = 'primary', loading, trend }) {
  if (loading) {
    return (
      <div className="kpi-card">
        <Skeleton height="12px" width="60%" />
        <Skeleton height="36px" width="40%" style={{ marginTop: 8 }} />
        <Skeleton height="10px" width="50%" style={{ marginTop: 6 }} />
      </div>
    );
  }

  return (
    <div className={`kpi-card kpi-card--${color} anim-fade-up`}>
      <div className="kpi-header">
        <span className="kpi-label">{label}</span>
        {icon && <span className="kpi-icon" aria-hidden="true">{icon}</span>}
      </div>
      <div className="kpi-value">{value ?? '—'}</div>
      <div className="kpi-footer">
        {trend !== undefined && (
          <span className={`kpi-trend ${trend >= 0 ? 'kpi-trend--up' : 'kpi-trend--down'}`}>
            {trend >= 0 ? '▲' : '▼'} {Math.abs(trend)}%
          </span>
        )}
        {sub && <span className="kpi-sub">{sub}</span>}
      </div>
    </div>
  );
}
