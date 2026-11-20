import { useState } from 'react';
import Button from '../ui/Button';
import { sendRetentionEmail, sendRLFeedback } from '../../api/predictions';
import './AiCopilotCard.css';

export default function AiCopilotCard({ shapData, copilotData, rlData, customerEmail, customerName }) {
  const [copied, setCopied] = useState(false);
  const [sendingEmail, setSendingEmail] = useState(false);
  const [emailStatus, setEmailStatus] = useState(null);
  const [feedbackGiven, setFeedbackGiven] = useState(null);

  if (!shapData && !copilotData && !rlData) {
    return null;
  }

  function handleCopyEmail() {
    if (!copilotData?.email_body) return;
    const fullText = `Subject: ${copilotData.email_subject || 'Important update'}\n\n${copilotData.email_body}`;
    navigator.clipboard.writeText(fullText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  async function handleSendDirectEmail() {
    const targetEmail = customerEmail || prompt('Enter recipient email address:');
    if (!targetEmail) return;

    setSendingEmail(true);
    setEmailStatus(null);
    try {
      await sendRetentionEmail({
        recipient_email: targetEmail,
        subject: copilotData.email_subject || 'Retentrix Account Update',
        body: copilotData.email_body
      });
      setEmailStatus(`✓ Delivered to ${targetEmail}`);
    } catch (err) {
      setEmailStatus('❌ Failed to send. Check SMTP settings.');
    } finally {
      setSendingEmail(false);
    }
  }

  async function handleRLFeedback(reward) {
    if (!rlData?.selected_action) return;
    try {
      await sendRLFeedback({
        arm_id: rlData.selected_action.arm_id,
        observed_reward: reward,
        churn_probability: rlData.context?.churn_risk_pct || 50.0,
        tenure_months: rlData.context?.tenure_months || 12
      });
      setFeedbackGiven(reward > 0 ? 'Retained (+Reward Recorded)' : 'Lost (-Penalty Recorded)');
    } catch (e) {
      setFeedbackGiven('Feedback saved');
    }
  }

  const riskDrivers = shapData?.top_risk_drivers || [];
  const protectiveFactors = shapData?.top_protective_factors || [];
  const selectedArm = rlData?.selected_action;

  const rootCauses = Array.isArray(copilotData?.root_cause_bullets)
    ? copilotData.root_cause_bullets
    : (copilotData?.root_cause_analysis ? [copilotData.root_cause_analysis] : []);

  const strategies = Array.isArray(copilotData?.strategy_steps)
    ? copilotData.strategy_steps
    : (copilotData?.retention_strategy ? [copilotData.retention_strategy] : []);

  return (
    <div className="copilot-container anim-fade-up">
      {shapData && (
        <div className="copilot-section-card">
          <div className="copilot-header">
            <h4 className="copilot-title">
              <span style={{ fontSize: 18 }}>📊</span>
              Feature Explainability (SHAP Analysis)
            </h4>
            <span className="copilot-badge copilot-badge-brand">Risk Attribution</span>
          </div>
          <div className="shap-bars">
            {riskDrivers.map((item, idx) => {
              const widthPct = Math.min(100, Math.max(20, Math.abs(item.impact) * 40));
              return (
                <div key={`risk-${idx}`} className="shap-row">
                  <div className="shap-labels">
                    <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{item.feature}</span>
                    <span style={{ color: '#f87171', fontWeight: 600 }}>+{Math.abs(item.impact).toFixed(2)} Risk Driver</span>
                  </div>
                  <div className="shap-bar-track">
                    <div className="shap-bar-fill-risk" style={{ width: `${widthPct}%` }} />
                  </div>
                </div>
              );
            })}
            {protectiveFactors.map((item, idx) => {
              const widthPct = Math.min(100, Math.max(20, Math.abs(item.impact) * 40));
              return (
                <div key={`safe-${idx}`} className="shap-row">
                  <div className="shap-labels">
                    <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{item.feature}</span>
                    <span style={{ color: '#34d399', fontWeight: 600 }}>-{Math.abs(item.impact).toFixed(2)} Protective</span>
                  </div>
                  <div className="shap-bar-track">
                    <div className="shap-bar-fill-safe" style={{ width: `${widthPct}%` }} />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {copilotData && (
        <div className="copilot-section-card">
          <div className="copilot-header">
            <h4 className="copilot-title">
              <span style={{ fontSize: 18 }}>⚡</span>
              Retentrix Copilot (AI Retention Strategy)
            </h4>
            <span className={`copilot-badge ${copilotData.urgency === 'Critical' ? 'copilot-badge-danger' : 'copilot-badge-warning'}`}>
              {copilotData.urgency || 'Action Required'}
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <p style={{ margin: '0 0 8px', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Key Churn Drivers & Friction Points
              </p>
              <ul className="copilot-bullet-list">
                {rootCauses.map((bullet, i) => (
                  <li key={i} className="copilot-bullet-item">
                    <span className="copilot-bullet-dot">•</span>
                    <span>{bullet}</span>
                  </li>
                ))}
              </ul>
            </div>

            <div>
              <p style={{ margin: '0 0 8px', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                Prescribed Retention Action Steps
              </p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {strategies.map((step, i) => (
                  <div key={i} className="copilot-numbered-item">
                    <span className="copilot-number-badge">{i + 1}</span>
                    <span>{step}</span>
                  </div>
                ))}
              </div>
            </div>

            {copilotData.recommended_incentive && (
              <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-default)', padding: '12px 14px', borderRadius: 'var(--radius-md)' }}>
                <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                  Recommended Loyalty Incentive
                </span>
                <p style={{ margin: '4px 0 0', color: 'var(--brand-primary-h)', fontWeight: 600, fontSize: 'var(--text-sm)' }}>
                  🎁 {copilotData.recommended_incentive}
                </p>
              </div>
            )}

            {copilotData.economic_viability && (
              <div className="copilot-viability-box">
                <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--brand-primary-h)', textTransform: 'uppercase' }}>
                  Account Economic Feasibility
                </span>
                <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                  {copilotData.economic_viability}
                </p>
              </div>
            )}
          </div>

          {copilotData.email_body && (
            <div className="copilot-email-box">
              <div className="copilot-email-header">
                <div>
                  <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)', textTransform: 'uppercase', display: 'block' }}>
                    Personalized Outreach Draft {customerName ? `for ${customerName}` : ''}
                  </span>
                  <span style={{ fontSize: 'var(--text-sm)', fontWeight: 600, color: 'var(--text-primary)' }}>
                    ✉️ {copilotData.email_subject}
                  </span>
                </div>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  {emailStatus && (
                    <span style={{ fontSize: 'var(--text-xs)', color: emailStatus.startsWith('✓') ? '#34d399' : '#f87171' }}>
                      {emailStatus}
                    </span>
                  )}
                  <Button variant="secondary" size="xs" onClick={handleCopyEmail}>
                    {copied ? '✓ Copied' : 'Copy'}
                  </Button>
                  <Button variant="primary" size="xs" loading={sendingEmail} onClick={handleSendDirectEmail}>
                    🚀 Send Direct Email
                  </Button>
                </div>
              </div>
              <div className="copilot-email-body">{copilotData.email_body}</div>
            </div>
          )}
        </div>
      )}

      {selectedArm && (
        <div className="copilot-section-card">
          <div className="copilot-header">
            <h4 className="copilot-title">
              <span style={{ fontSize: 18 }}>🎯</span>
              Reinforcement Learning Policy (LinUCB Bandit)
            </h4>
            <span className="copilot-badge copilot-badge-success">Optimal Policy Arm</span>
          </div>

          <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border-default)', padding: '14px', borderRadius: 'var(--radius-md)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 'var(--text-md)', fontWeight: 700, color: 'var(--text-primary)' }}>
                {selectedArm.name}
              </span>
              <span className="copilot-badge copilot-badge-brand">{selectedArm.category}</span>
            </div>
            <p style={{ margin: '6px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {selectedArm.description}
            </p>
          </div>

          <div className="rl-metrics-grid">
            <div className="rl-metric-card">
              <p className="rl-metric-lbl">Expected Net ROI</p>
              <p className="rl-metric-val" style={{ color: selectedArm.net_roi_usd >= 0 ? '#34d399' : '#f87171' }}>
                {selectedArm.net_roi_usd >= 0 ? `+$${selectedArm.net_roi_usd}` : `-$${Math.abs(selectedArm.net_roi_usd)}`}
              </p>
            </div>
            <div className="rl-metric-card">
              <p className="rl-metric-lbl">Saved Revenue</p>
              <p className="rl-metric-val">${selectedArm.expected_saved_revenue}</p>
            </div>
            <div className="rl-metric-card">
              <p className="rl-metric-lbl">Incentive Cost</p>
              <p className="rl-metric-val" style={{ color: 'var(--text-secondary)' }}>${selectedArm.cost_usd}</p>
            </div>
            <div className="rl-metric-card">
              <p className="rl-metric-lbl">UCB Score</p>
              <p className="rl-metric-val">{selectedArm.ucb_score}</p>
            </div>
          </div>

          <div className="rl-feedback-bar">
            <div>
              <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-primary)', display: 'block' }}>
                Online Reinforcement Learning Feedback
              </span>
              <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                Did this intervention retain the customer? Click to update LinUCB weights in real-time.
              </span>
            </div>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              {feedbackGiven ? (
                <span style={{ fontSize: 'var(--text-xs)', fontWeight: 600, color: '#34d399' }}>✓ {feedbackGiven}</span>
              ) : (
                <>
                  <Button variant="secondary" size="xs" onClick={() => handleRLFeedback(1.0)}>
                    ✓ Customer Retained (+1)
                  </Button>
                  <Button variant="ghost" size="xs" onClick={() => handleRLFeedback(-1.0)}>
                    ✗ Customer Churned (-1)
                  </Button>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
