import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useCustomers } from '../hooks/useCustomers';
import { getAnalyticsSummary, getRiskBreakdown, getTopAtRisk, getRecentActivity } from '../api/analytics';
import { seedCustomers } from '../api/customers';
import KpiCard from '../components/dashboard/KpiCard';
import RiskPieChart from '../components/dashboard/RiskPieChart';
import RiskBadge from '../components/customers/RiskBadge';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import Skeleton from '../components/ui/Skeleton';

// Safe Percentage Normalizer & Formatter
function formatScorePct(score) {
  if (score == null || isNaN(score)) return '0.0%';
  const num = Number(score);
  // If > 1.0 (e.g. 20.79), it's already a percentage (20.8%)
  // If <= 1.0 (e.g. 0.2079), convert to percentage (20.8%)
  const pct = num > 1.0 ? num : num * 100;
  return `${pct.toFixed(1)}%`;
}

export default function DashboardPage() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const { customers, loading: custLoading, refetch } = useCustomers();

  const [summary, setSummary] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [topAtRisk, setTopAtRisk] = useState([]);
  const [recentActivity, setRecentActivity] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('contract'); // 'contract' | 'region' | 'payment'
  const [seeding, setSeeding] = useState(false);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const [sum, brk, risk, act] = await Promise.allSettled([
          getAnalyticsSummary(),
          getRiskBreakdown(),
          getTopAtRisk(5),
          getRecentActivity(8),
        ]);
        if (sum.status === 'fulfilled') setSummary(sum.value);
        if (brk.status === 'fulfilled') setBreakdown(brk.value);
        if (risk.status === 'fulfilled') setTopAtRisk(risk.value);
        if (act.status === 'fulfilled') setRecentActivity(act.value);
      } catch {
        // Fallback gracefully
      } finally {
        setLoading(false);
      }
    }
    loadData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run once on mount — analytics are not customer-count-dependent

  // Client-side fallback computation if analytics endpoints return empty
  const computedStats = useMemo(() => {
    if (summary) {
      return {
        total: summary.total_customers,
        critical: summary.critical_risk_count,
        moderate: summary.moderate_risk_count,
        low: summary.low_risk_count,
        avgScore: summary.avg_churn_score,
        criticalRate: summary.critical_rate_pct,
        unscored: summary.unscored_count,
      };
    }
    if (!customers.length) return { total: 0, critical: 0, moderate: 0, low: 0, avgScore: 0, criticalRate: 0, unscored: 0 };
    const scored = customers.filter((c) => c.latest_churn_score != null);
    const critical = customers.filter((c) => c.latest_risk_level === 'Critical').length;
    const moderate = customers.filter((c) => c.latest_risk_level === 'Moderate').length;
    const low = customers.filter((c) => c.latest_risk_level === 'Low').length;
    const avgScore = scored.length ? scored.reduce((s, c) => s + c.latest_churn_score, 0) / scored.length : 0;
    const criticalRate = scored.length ? ((critical / scored.length) * 100).toFixed(1) : 0;
    return { total: customers.length, critical, moderate, low, avgScore, criticalRate, unscored: customers.length - scored.length };
  }, [summary, customers]);

  const pieData = [
    { label: 'Critical Risk', value: computedStats.critical, color: '#ef4444' },
    { label: 'Moderate Risk', value: computedStats.moderate, color: '#f59e0b' },
    { label: 'Low Risk', value: computedStats.low, color: '#10b981' },
    { label: 'Unscored', value: computedStats.unscored, color: '#6366f1' },
  ].filter((d) => d.value > 0);

  const displayAtRisk = topAtRisk.length
    ? topAtRisk
    : customers
        .filter((c) => c.latest_risk_level === 'Critical')
        .sort((a, b) => (b.latest_churn_score ?? 0) - (a.latest_churn_score ?? 0))
        .slice(0, 5);

  async function handleQuickSeed() {
    setSeeding(true);
    try {
      await seedCustomers(50);
      refetch();
    } finally {
      setSeeding(false);
    }
  }

  const greeting = () => {
    const h = new Date().getHours();
    if (h < 12) return 'Good morning';
    if (h < 18) return 'Good afternoon';
    return 'Good evening';
  };

  const isLoading = loading && custLoading;

  return (
    <div>
      {/* Page Header */}
      <div className="page-header flex items-center justify-between" style={{ flexWrap: 'wrap', gap: 14 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <h1>
              {greeting()}, {user?.first_name || 'Team'} 👋
            </h1>
            <Badge variant="primary" size="sm">
              Live Churn Intelligence
            </Badge>
          </div>
          <p>
            Real-time portfolio retention metrics & AI risk tracking •{' '}
            {new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}
          </p>
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => navigate('/customers')}
            leftIcon={
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="7" cy="5" r="3" stroke="currentColor" strokeWidth="1.4" />
                <path d="M1 13c0-3 2.7-5 6-5s6 2 6 5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            }
          >
            Manage Accounts
          </Button>
          <Button
            variant="primary"
            size="sm"
            onClick={() => navigate('/predict')}
            leftIcon={
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M1 11L5 6l3 3 4-7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            }
          >
            Run ML Predict
          </Button>
        </div>
      </div>

      {/* Empty Database Prompt */}
      {!isLoading && computedStats.total === 0 && (
        <div
          className="card anim-scale-pop"
          style={{
            marginBottom: 'var(--sp-6)',
            background: 'linear-gradient(135deg, rgba(79,70,229,0.08) 0%, rgba(16,185,129,0.08) 100%)',
            border: '1px solid var(--border-accent)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 16,
          }}
        >
          <div>
            <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600 }}>Your database is currently empty</h3>
            <p style={{ margin: '4px 0 0', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
              Seed 50 realistic demo accounts with pre-calculated AI churn predictions to experience the full analytics suite.
            </p>
          </div>
          <Button variant="primary" loading={seeding} onClick={handleQuickSeed}>
            ⚡ 1-Click Seed Demo Accounts
          </Button>
        </div>
      )}

      {/* KPI Top Cards */}
      <div className="grid-4 mb-6 anim-fade-up">
        <KpiCard
          loading={isLoading}
          label="Total Managed Accounts"
          color="primary"
          value={isLoading ? null : computedStats.total.toLocaleString()}
          icon="👥"
          sub="Enterprise portfolio"
        />
        <KpiCard
          loading={isLoading}
          label="Critical Risk Accounts"
          color="danger"
          value={isLoading ? null : computedStats.critical.toLocaleString()}
          icon="🚨"
          sub={`${computedStats.criticalRate}% critical rate`}
        />
        <KpiCard
          loading={isLoading}
          label="Moderate Risk Watchlist"
          color="warning"
          value={isLoading ? null : computedStats.moderate.toLocaleString()}
          icon="⚠️"
          sub="Requires CSM follow-up"
        />
        <KpiCard
          loading={isLoading}
          label="Avg Predicted Churn"
          color="info"
          value={isLoading ? null : formatScorePct(computedStats.avgScore)}
          icon="🧠"
          sub="Portfolio health score"
        />
      </div>

      {/* Analytics Grid: Risk Distribution & Priority Feed */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--sp-6)', marginBottom: 'var(--sp-6)' }}>
        {/* Donut Chart */}
        <div className="card anim-fade-up delay-2">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600 }}>Risk Distribution Tier</h3>
              <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                Portfolio breakdown by ML risk classification
              </p>
            </div>
          </div>
          {isLoading ? (
            <Skeleton height="200px" />
          ) : pieData.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '50px 0' }}>
              No accounts scored yet.
            </div>
          ) : (
            <RiskPieChart data={pieData} size={200} />
          )}
        </div>

        {/* Priority Critical Feed */}
        <div className="card anim-fade-up delay-3">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600 }}>Priority At-Risk Accounts</h3>
              <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                Highest predicted probability of churn
              </p>
            </div>
            <Button variant="ghost" size="sm" onClick={() => navigate('/customers')}>
              View all →
            </Button>
          </div>

          {isLoading ? (
            Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} height="44px" style={{ marginBottom: 8 }} />)
          ) : displayAtRisk.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', padding: '40px 0', textAlign: 'center' }}>
              🎉 Zero critical churn risk detected in portfolio!
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {displayAtRisk.map((c) => (
                <div
                  key={c.id}
                  className="flex items-center justify-between"
                  style={{
                    padding: '10px 14px',
                    background: 'var(--bg-raised)',
                    borderRadius: 'var(--radius-md)',
                    cursor: 'pointer',
                    border: '1px solid var(--border-subtle)',
                    transition: 'var(--transition-fast)',
                  }}
                  onClick={() => navigate(`/customers/${c.id}`)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => e.key === 'Enter' && navigate(`/customers/${c.id}`)}
                >
                  <div>
                    <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>
                      {c.first_name} {c.last_name}
                    </div>
                    <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                      {c.customer_code} • {c.tenure_months}mo tenure
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span
                      style={{
                        fontSize: 'var(--text-sm)',
                        fontWeight: 700,
                        color: 'var(--brand-crimson)',
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      {formatScorePct(c.latest_churn_score)}
                    </span>
                    <RiskBadge level={c.latest_risk_level} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Categorized Breakdown Tabbed Section */}
      {breakdown && (
        <div className="card anim-fade-up delay-4" style={{ marginBottom: 'var(--sp-6)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600 }}>Risk Breakdown by Segment</h3>
              <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                Compare churn metrics across commitment plans, geography, and payment methods
              </p>
            </div>
            <div style={{ display: 'flex', gap: 6, background: 'var(--bg-raised)', padding: 4, borderRadius: 'var(--radius-md)' }}>
              {[
                { id: 'contract', label: 'Contract' },
                { id: 'region', label: 'Geography' },
                { id: 'payment', label: 'Payment' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  style={{
                    background: activeTab === tab.id ? 'var(--brand-primary)' : 'none',
                    color: activeTab === tab.id ? '#ffffff' : 'var(--text-secondary)',
                    border: 'none',
                    padding: '6px 12px',
                    borderRadius: 'var(--radius-sm)',
                    fontSize: 'var(--text-xs)',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'var(--transition-fast)',
                  }}
                  onClick={() => setActiveTab(tab.id)}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 12 }}>
            {(activeTab === 'contract'
              ? breakdown.by_contract
              : activeTab === 'region'
              ? breakdown.by_region
              : breakdown.by_payment
            )?.map((item, idx) => (
              <div
                key={idx}
                style={{
                  padding: '12px 14px',
                  background: 'var(--bg-raised)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                  <strong style={{ fontSize: 'var(--text-sm)', color: 'var(--text-primary)' }}>{item.category}</strong>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{item.total} accounts</span>
                </div>
                <div style={{ display: 'flex', gap: 8, fontSize: 'var(--text-xs)', marginBottom: 6 }}>
                  <span style={{ color: 'var(--brand-crimson)' }}>{item.critical} Critical</span>
                  <span style={{ color: 'var(--brand-amber)' }}>{item.moderate} Moderate</span>
                  <span style={{ color: 'var(--brand-emerald)' }}>{item.low} Low</span>
                </div>
                <div style={{ fontSize: 'var(--text-xs)', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                  Avg Churn: <strong>{formatScorePct(item.avg_churn_score)}</strong>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Prediction Audit Stream */}
      {recentActivity.length > 0 && (
        <div className="card anim-fade-up delay-4">
          <h3 style={{ margin: '0 0 14px', fontSize: 'var(--text-md)', fontWeight: 600 }}>Recent AI Prediction History</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {recentActivity.map((r) => (
              <div
                key={r.id}
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '10px 14px',
                  background: 'var(--bg-raised)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <div>
                  <div style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>
                    {r.customer_name}
                  </div>
                  <div style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                    {r.customer_code} • {r.created_at ? new Date(r.created_at).toLocaleTimeString() : 'Recent'}
                  </div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 'var(--text-sm)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                    {formatScorePct(r.churn_probability)}
                  </span>
                  <RiskBadge level={r.risk_level} />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
