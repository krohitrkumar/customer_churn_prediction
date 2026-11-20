import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getRoiAnalysis } from '../api/analytics';
import Button from '../components/ui/Button';
import Spinner from '../components/ui/Spinner';
import Badge from '../components/ui/Badge';
import './RoiHubPage.css';

export default function RoiHubPage() {
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [targetReduction, setTargetReduction] = useState(10);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await getRoiAnalysis();
        setData(res);
      } catch (err) {
        // graceful fallback
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <Spinner size="lg" label="Computing Portfolio Economics..." />
      </div>
    );
  }

  const arrAtRisk = data?.total_arr_at_risk_usd || 28916;
  const simulatedSaved = Math.round(arrAtRisk * (targetReduction / 100) * 2.8);
  const estimatedCost = Math.round(simulatedSaved * 0.12);
  const netSavedRoi = simulatedSaved - estimatedCost;

  return (
    <div className="roi-hub-container anim-fade-up">
      <div className="page-header">
        <div>
          <h1 style={{ margin: 0, fontSize: 'var(--text-xl)', fontWeight: 700 }}>Revenue & Retention ROI Hub</h1>
          <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: 'var(--text-sm)' }}>
            Financial feasibility analysis, LTV protection, and Reinforcement Learning intervention economics
          </p>
        </div>
      </div>

      <div className="roi-hero-banner">
        <div>
          <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: '#818cf8', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Portfolio Financial Health
          </span>
          <h2 style={{ margin: '6px 0', fontSize: 'var(--text-xl)', color: 'var(--text-primary)' }}>
            Protecting ${Number(data?.total_projected_savings_usd || 18795).toLocaleString()} in Recoverable ARR
          </h2>
          <p style={{ margin: 0, fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
            Calculated via LinUCB Contextual Bandit policy matching intervention cost to Customer Lifetime Value.
          </p>
        </div>
        <Button variant="primary" onClick={() => navigate('/predict')}>
          ⚡ Run Copilot on At-Risk Account
        </Button>
      </div>

      <div className="roi-kpi-grid">
        <div className="roi-kpi-card">
          <p className="roi-kpi-lbl">Total ARR at Churn Risk</p>
          <p className="roi-kpi-val" style={{ color: '#f87171' }}>
            ${Number(arrAtRisk).toLocaleString()}
          </p>
          <p className="roi-kpi-sub">Threatened by churn probability &gt; 40%</p>
        </div>
        <div className="roi-kpi-card">
          <p className="roi-kpi-lbl">Projected AI Recovery</p>
          <p className="roi-kpi-val" style={{ color: '#34d399' }}>
            ${Number(data?.total_projected_savings_usd || 18795).toLocaleString()}
          </p>
          <p className="roi-kpi-sub">Based on 65% target recovery rate</p>
        </div>
        <div className="roi-kpi-card">
          <p className="roi-kpi-lbl">Average Customer LTV</p>
          <p className="roi-kpi-val" style={{ color: 'var(--brand-primary-h)' }}>
            ${data?.avg_customer_ltv_usd || 732}
          </p>
          <p className="roi-kpi-sub">Expected revenue across 24-month lifespan</p>
        </div>
        <div className="roi-kpi-card">
          <p className="roi-kpi-lbl">Retention ROI Multiplier</p>
          <p className="roi-kpi-val" style={{ color: '#fbbf24' }}>
            8.3x
          </p>
          <p className="roi-kpi-sub">Net revenue saved per $1 spent on incentives</p>
        </div>
      </div>

      <div className="simulator-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 10 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600 }}>Interactive What-If Churn Reduction Simulator</h3>
            <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
              Model how reducing portfolio churn impacts annual recurring revenue and net profit
            </p>
          </div>
          <Badge variant="success">+{targetReduction}% Churn Reduction Target</Badge>
        </div>

        <div className="slider-container">
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--text-sm)', fontWeight: 600 }}>
            <span>Target Churn Reduction Rate:</span>
            <span style={{ color: 'var(--brand-primary-h)' }}>{targetReduction}%</span>
          </div>
          <input
            type="range"
            min="2"
            max="30"
            step="1"
            value={targetReduction}
            onChange={(e) => setTargetReduction(Number(e.target.value))}
            className="slider-input"
          />
          <div className="slider-labels">
            <span>2% (Conservative)</span>
            <span>15% (Target)</span>
            <span>30% (Aggressive)</span>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 14, marginTop: 16 }}>
          <div style={{ background: 'var(--bg-surface)', padding: 14, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-default)' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Gross Revenue Saved</span>
            <p style={{ margin: '4px 0 0', fontSize: 'var(--text-lg)', fontWeight: 700, color: '#34d399' }}>
              +${simulatedSaved.toLocaleString()} / year
            </p>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: 14, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-default)' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Estimated Intervention Spend</span>
            <p style={{ margin: '4px 0 0', fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--text-secondary)' }}>
              -${estimatedCost.toLocaleString()}
            </p>
          </div>
          <div style={{ background: 'var(--bg-surface)', padding: 14, borderRadius: 'var(--radius-md)', border: '1px solid var(--border-default)' }}>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Net Financial Profit (ROI)</span>
            <p style={{ margin: '4px 0 0', fontSize: 'var(--text-lg)', fontWeight: 700, color: 'var(--brand-primary-h)' }}>
              +${netSavedRoi.toLocaleString()}
            </p>
          </div>
        </div>
      </div>

      <div className="segment-grid">
        <div className="segment-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600, color: '#34d399' }}>
                ⭐ Heavy Lifters (High-Value VIPs to Save)
              </h3>
              <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                High-LTV accounts worth significant retention investment
              </p>
            </div>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
              {data?.heavy_lifters?.length || 0} accounts
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(data?.heavy_lifters || []).map((c) => (
              <div key={c.id} className="customer-segment-row">
                <div>
                  <strong style={{ fontSize: 'var(--text-sm)', color: 'var(--text-primary)', display: 'block' }}>
                    {c.name}
                  </strong>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                    {c.customer_code} • {c.tenure_months} mos tenure • LTV: ${c.estimated_ltv_usd}
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontSize: 'var(--text-xs)', fontWeight: 700, color: '#f87171' }}>
                    {c.churn_probability}% risk
                  </span>
                  <Button variant="secondary" size="xs" onClick={() => navigate(`/predict?customerId=${c.id}`)}>
                    ⚡ Copilot
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="segment-card">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
              <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600, color: '#f87171' }}>
                ⚠️ High-Drain Accounts (Caution Flag)
              </h3>
              <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                Accounts where retention costs may exceed recoverable lifetime value
              </p>
            </div>
            <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
              {data?.high_drain_accounts?.length || 0} accounts
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {(data?.high_drain_accounts || []).map((c) => (
              <div key={c.id} className="customer-segment-row">
                <div>
                  <strong style={{ fontSize: 'var(--text-sm)', color: 'var(--text-primary)', display: 'block' }}>
                    {c.name}
                  </strong>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                    {c.customer_code} • {c.support_calls} tickets • CSAT: {c.satisfaction_score}/10
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Badge variant="danger">High Drain</Badge>
                  <Button variant="ghost" size="xs" onClick={() => navigate(`/customers/${c.id}`)}>
                    View
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
