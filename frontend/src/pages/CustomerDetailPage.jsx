import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getCustomer, deleteCustomer, updateCustomer } from '../api/customers';
import { useHistory } from '../hooks/usePredictions';
import { useAuth } from '../context/AuthContext';
import { useToastState } from '../hooks/useToast';
import { extractError } from '../api/client';
import RiskBadge from '../components/customers/RiskBadge';
import CustomerForm from '../components/customers/CustomerForm';
import ChurnGauge from '../components/predictions/ChurnGauge';
import PlaybookCard from '../components/predictions/PlaybookCard';
import Modal from '../components/ui/Modal';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import Spinner from '../components/ui/Spinner';
import ToastContainer from '../components/ui/Toast';
import './CustomerDetailPage.css';

function formatScorePct(score) {
  if (score == null || isNaN(score)) return '0.0%';
  const num = Number(score);
  const pct = num > 1.0 ? num : num * 100;
  return `${pct.toFixed(1)}%`;
}

export default function CustomerDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { canWrite, isAdmin } = useAuth();
  const { toasts, success, error: toastError, dismiss } = useToastState();

  const [customer, setCustomer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState(null); // 'edit' | 'delete' | null
  const [saving, setSaving] = useState(false);

  const { history, loading: historyLoading, load: loadHistory } = useHistory(id);

  useEffect(() => {
    async function loadData() {
      setLoading(true);
      try {
        const data = await getCustomer(id);
        setCustomer(data);
        loadHistory();
      } catch (err) {
        toastError('Not found', extractError(err));
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [id]);

  async function handleUpdate(formData) {
    setSaving(true);
    try {
      const updated = await updateCustomer(id, formData);
      setCustomer(updated);
      success('Updated', 'Customer details saved.');
      setModal(null);
    } catch (err) {
      toastError('Update failed', extractError(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    setSaving(true);
    try {
      await deleteCustomer(id);
      success('Deleted', 'Customer deleted successfully.');
      navigate('/customers');
    } catch (err) {
      toastError('Delete failed', extractError(err));
      setSaving(false);
    }
  }

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <Spinner size="lg" label="Loading customer details..." />
      </div>
    );
  }

  if (!customer) {
    return (
      <div style={{ textAlign: 'center', padding: '60px 20px' }}>
        <h2>Customer Not Found</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: 20 }}>This customer record does not exist or has been removed.</p>
        <Button variant="secondary" onClick={() => navigate('/customers')}>Return to Customers</Button>
      </div>
    );
  }

  return (
    <div>
      <ToastContainer toasts={toasts} onDismiss={dismiss} />

      {/* Back button & Title */}
      <div style={{ marginBottom: 20 }}>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate('/customers')}
          style={{ marginBottom: 12 }}
          leftIcon={
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M9 11L5 7l4-4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          }
        >
          Back to Customers
        </Button>

        <div className="page-header flex items-center justify-between">
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              <h1>{customer.first_name} {customer.last_name}</h1>
              <span className="font-mono text-sm" style={{ color: 'var(--text-muted)' }}>{customer.customer_code}</span>
              <RiskBadge level={customer.latest_risk_level} />
            </div>
            <p>{customer.email || 'No email registered'}</p>
          </div>

          <div style={{ display: 'flex', gap: 10 }}>
            <Button
              variant="primary"
              size="sm"
              onClick={() => navigate(`/predict?customerId=${customer.id}`)}
              leftIcon={
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path d="M1 11L5 6l3 3 4-7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              }
            >
              Run AI Prediction
            </Button>
            {canWrite && <Button variant="secondary" size="sm" onClick={() => setModal('edit')}>Edit Profile</Button>}
            {isAdmin && <Button variant="danger" size="sm" onClick={() => setModal('delete')}>Delete</Button>}
          </div>
        </div>
      </div>

      {/* Grid: Details + Churn score card */}
      <div className="customer-detail-grid">
        {/* Left: Attributes */}
        <div className="card anim-fade-up">
          <h3 style={{ margin: '0 0 16px', fontSize: 'var(--text-md)', fontWeight: 600 }}>Account & Engagement Parameters</h3>
          <div className="attribute-grid">
            <div className="attribute-box">
              <span className="attr-label">Tenure</span>
              <span className="attr-val">{customer.tenure_months} months</span>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Satisfaction Score</span>
              <span className="attr-val">{customer.satisfaction_score} / 10</span>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Support Calls</span>
              <span className="attr-val">{customer.support_calls} tickets</span>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Late Payments</span>
              <span className="attr-val">{customer.late_payments} recorded</span>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Contract Type</span>
              <span className="attr-val"><Badge variant="default">{customer.contract_type?.replace(/_/g, ' ')}</Badge></span>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Payment Method</span>
              <span className="attr-val"><Badge variant="default">{customer.payment_method?.toUpperCase()}</Badge></span>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Region</span>
              <span className="attr-val"><Badge variant="default">{customer.region?.replace(/_/g, ' ')}</Badge></span>
            </div>
            <div className="attribute-box">
              <span className="attr-label">Created At</span>
              <span className="attr-val text-xs text-muted">
                {customer.created_at ? new Date(customer.created_at).toLocaleDateString() : '—'}
              </span>
            </div>
          </div>
        </div>

        {/* Right: Latest Churn Score */}
        <div className="card anim-fade-up delay-1" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', textAlign: 'center' }}>
          <h3 style={{ margin: '0 0 14px', fontSize: 'var(--text-md)', fontWeight: 600 }}>Latest Predicted Churn Probability</h3>
          {customer.latest_churn_score != null ? (
            <div>
              <ChurnGauge probability={customer.latest_churn_score} size={200} />
              <div style={{ marginTop: 12 }}>
                <RiskBadge level={customer.latest_risk_level} />
              </div>
            </div>
          ) : (
            <div style={{ color: 'var(--text-muted)', padding: '30px 10px' }}>
              <div style={{ fontSize: 36, marginBottom: 10 }}>📊</div>
              <p style={{ margin: 0 }}>This customer has not been evaluated by the AI model yet.</p>
              <Button
                variant="primary"
                size="sm"
                style={{ marginTop: 16 }}
                onClick={() => navigate(`/predict?customerId=${customer.id}`)}
              >
                Run First Prediction
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Historical Predictions */}
      <div className="card anim-fade-up delay-2" style={{ marginTop: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: 'var(--text-md)', fontWeight: 600 }}>Prediction Audit Trail & Historical Logs</h3>
            <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>Historical runs stored in PostgreSQL</p>
          </div>
        </div>

        {historyLoading ? (
          <Spinner size="md" />
        ) : history.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', fontSize: 'var(--text-sm)', margin: 0 }}>No prediction history runs on record for this customer.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {history.map((record) => (
              <div key={record.id} className="history-record-card">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8, flexWrap: 'wrap', gap: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <span style={{ fontSize: 'var(--text-lg)', fontWeight: 700, fontFamily: 'var(--font-mono)', color: (record.churn_probability > 1 ? record.churn_probability : record.churn_probability * 100) > 70 ? 'var(--brand-crimson)' : (record.churn_probability > 1 ? record.churn_probability : record.churn_probability * 100) > 35 ? 'var(--brand-amber)' : 'var(--brand-emerald)' }}>
                      {formatScorePct(record.churn_probability)}
                    </span>
                    <RiskBadge level={record.risk_level} />
                  </div>
                  <span className="text-xs text-muted">
                    {record.created_at ? new Date(record.created_at).toLocaleString() : '—'}
                  </span>
                </div>

                {record.playbook_recommendations?.length > 0 && (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 8, marginTop: 8 }}>
                    {record.playbook_recommendations.map((p, idx) => (
                      <PlaybookCard key={idx} index={idx} icon={p.icon} category={p.category} action={p.action} />
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Edit Modal */}
      <Modal
        open={modal === 'edit'}
        onOpenChange={(o) => { if (!o) setModal(null); }}
        title={`Edit ${customer.first_name} ${customer.last_name}`}
        size="lg"
      >
        <CustomerForm
          initial={customer}
          onSubmit={handleUpdate}
          onCancel={() => setModal(null)}
          loading={saving}
        />
      </Modal>

      {/* Delete Modal */}
      <Modal
        open={modal === 'delete'}
        onOpenChange={(o) => { if (!o) setModal(null); }}
        title="Delete Customer Record"
        description={`Are you sure you want to remove ${customer.first_name} ${customer.last_name}?`}
        size="sm"
        footer={
          <>
            <Button variant="secondary" onClick={() => setModal(null)}>Cancel</Button>
            <Button variant="danger" loading={saving} onClick={handleDelete}>Delete permanently</Button>
          </>
        }
      >
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--text-secondary)', margin: 0 }}>
          This will delete all customer attributes and associated prediction history logs.
        </p>
      </Modal>
    </div>
  );
}
