import { useState, useMemo, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { useCustomers } from '../hooks/useCustomers';
import { useToastState } from '../hooks/useToast';
import { createCustomer, updateCustomer, deleteCustomer, uploadCustomerFile, seedCustomers } from '../api/customers';
import { extractError } from '../api/client';
import RiskBadge from '../components/customers/RiskBadge';
import CustomerForm from '../components/customers/CustomerForm';
import Modal from '../components/ui/Modal';
import Button from '../components/ui/Button';
import Badge from '../components/ui/Badge';
import { SkeletonRow } from '../components/ui/Skeleton';
import ToastContainer from '../components/ui/Toast';
import './CustomersPage.css';

const SORT_KEYS = {
  name: (c) => `${c.first_name} ${c.last_name}`,
  code: (c) => c.customer_code,
  risk: (c) => ({ Critical: 2, Moderate: 1, Low: 0 })[c.latest_risk_level] ?? -1,
  score: (c) => c.latest_churn_score ?? -1,
  tenure: (c) => c.tenure_months,
};

function formatScorePct(score) {
  if (score == null || isNaN(score)) return '—';
  const num = Number(score);
  const pct = num > 1.0 ? num : num * 100;
  return `${pct.toFixed(1)}%`;
}

const RISK_TABS = [
  { id: '', label: 'All Accounts' },
  { id: 'Critical', label: 'Critical Risk' },
  { id: 'Moderate', label: 'Moderate Risk' },
  { id: 'Low', label: 'Low Risk' },
  { id: 'unscored', label: 'Unscored' },
];

export default function CustomersPage() {
  const { canWrite, isAdmin } = useAuth();
  const navigate = useNavigate();
  const { customers, loading, refetch } = useCustomers();
  const { toasts, success, error: toastError, dismiss } = useToastState();

  const [search, setSearch] = useState('');
  const [riskFilter, setRisk] = useState('');
  const [sortKey, setSortKey] = useState('name');
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 25;

  const [modal, setModal] = useState(null); // null | 'create' | 'edit' | 'delete' | 'upload' | 'seed'
  const [selected, setSelected] = useState(null);
  const [saving, setSaving] = useState(false);

  // Upload modal state
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadDragOver, setUploadDragOver] = useState(false);
  const [uploadError, setUploadError] = useState(null);     // column validation error
  const [uploadProgress, setUploadProgress] = useState(0);  // 0–100 upload byte progress
  const fileInputRef = useRef(null);

  // Filter + sort
  const filtered = useMemo(() => {
    let list = customers;
    if (search.trim()) {
      const q = search.toLowerCase();
      list = list.filter(
        (c) =>
          `${c.first_name} ${c.last_name}`.toLowerCase().includes(q) ||
          c.customer_code.toLowerCase().includes(q) ||
          (c.email ?? '').toLowerCase().includes(q)
      );
    }
    if (riskFilter === 'unscored') list = list.filter((c) => !c.latest_risk_level);
    else if (riskFilter) list = list.filter((c) => c.latest_risk_level === riskFilter);

    const fn = SORT_KEYS[sortKey] ?? SORT_KEYS.name;
    list = [...list].sort((a, b) => {
      const av = fn(a),
        bv = fn(b);
      if (av == null) return 1;
      if (bv == null) return -1;
      return sortDir === 'asc' ? (av < bv ? -1 : av > bv ? 1 : 0) : av > bv ? -1 : av < bv ? 1 : 0;
    });
    return list;
  }, [customers, search, riskFilter, sortKey, sortDir]);

  const counts = useMemo(() => {
    return {
      all: customers.length,
      critical: customers.filter((c) => c.latest_risk_level === 'Critical').length,
      moderate: customers.filter((c) => c.latest_risk_level === 'Moderate').length,
      low: customers.filter((c) => c.latest_risk_level === 'Low').length,
      unscored: customers.filter((c) => !c.latest_risk_level).length,
    };
  }, [customers]);

  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  const paginated = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  function toggleSort(key) {
    if (sortKey === key) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else {
      setSortKey(key);
      setSortDir('asc');
    }
    setPage(1);
  }

  function SortIcon({ k }) {
    if (sortKey !== k) return <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>↕</span>;
    return <span style={{ color: 'var(--brand-primary-h)', marginLeft: 4 }}>{sortDir === 'asc' ? '↑' : '↓'}</span>;
  }

  async function handleSave(formData) {
    setSaving(true);
    try {
      if (modal === 'create') {
        await createCustomer(formData);
        success('Customer created', `${formData.first_name} ${formData.last_name} registered.`);
      } else {
        await updateCustomer(selected.id, formData);
        success('Customer updated', 'Changes saved successfully.');
      }
      setModal(null);
      setSelected(null);
      refetch();
    } catch (err) {
      toastError('Save failed', extractError(err));
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    setSaving(true);
    try {
      await deleteCustomer(selected.id);
      success('Deleted', `${selected.first_name} ${selected.last_name} removed.`);
      setModal(null);
      setSelected(null);
      refetch();
    } catch (err) {
      toastError('Delete failed', extractError(err));
    } finally {
      setSaving(false);
    }
  }

  // Client-side: read the first row of CSV/Excel to check required columns BEFORE uploading
  async function validateFileColumns(file) {
    const REQUIRED = ['customer_code', 'first_name', 'last_name', 'tenure_months', 'satisfaction_score'];
    const name = file.name.toLowerCase();
    const buf = await file.arrayBuffer();

    let headers = [];
    if (name.endsWith('.csv')) {
      // Read first line only
      const text = new TextDecoder().decode(buf);
      const firstLine = text.split('\n')[0] || '';
      headers = firstLine.split(',').map((h) => h.trim().replace(/"/g, '').toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_'));
    } else {
      // Excel: use SheetJS if available, else skip validation
      if (window.XLSX) {
        const wb = window.XLSX.read(buf, { type: 'array' });
        const ws = wb.Sheets[wb.SheetNames[0]];
        const rows = window.XLSX.utils.sheet_to_json(ws, { header: 1 });
        if (rows[0]) {
          headers = rows[0].map((h) => String(h).trim().toLowerCase().replace(/\s+/g, '_').replace(/-/g, '_'));
        }
      } else {
        return null; // Cannot validate — skip, let backend handle it
      }
    }
    const missing = REQUIRED.filter((r) => !headers.includes(r));
    return missing.length ? missing : null;
  }

  // Handle Bulk File Upload
  async function handleUploadSubmit(e) {
    e.preventDefault();
    if (!uploadFile) {
      toastError('Upload error', 'Please select a CSV or Excel spreadsheet.');
      return;
    }

    // Step 1: Client-side column check
    setUploadError(null);
    let missingCols = null;
    try {
      missingCols = await validateFileColumns(uploadFile);
    } catch {
      // If parsing fails don't block — let backend validate
    }
    if (missingCols) {
      setUploadError(`Missing required columns: ${missingCols.join(', ')}`);
      return; // Stop — don't send the file
    }

    // Step 2: Upload with progress tracking
    setUploading(true);
    setUploadProgress(0);
    try {
      const res = await uploadCustomerFile(uploadFile, (pct) => setUploadProgress(pct));
      const { imported = 0, skipped = 0, total_rows: total = 0 } = res;

      // Smart success message based on result
      let msg;
      if (imported === 0 && skipped > 0) {
        msg = `All ${total} customers already existed. 0 imported.`;
      } else if (skipped > 0) {
        msg = `${imported} of ${total} rows imported. ${skipped} already existed.`;
      } else {
        msg = `Upload complete — ${imported} new accounts imported with AI predictions.`;
      }

      success('Upload complete', msg);
      setModal(null);
      setUploadFile(null);
      setUploadProgress(0);
      refetch();
    } catch (err) {
      toastError('Upload failed', extractError(err));
    } finally {
      setUploading(false);
    }
  }

  // Handle 1-Click Seed Generator
  async function handleSeedDataset() {
    setSaving(true);
    try {
      const res = await seedCustomers(50);
      success('Dataset seeded', res.message || 'Seeded 50 demo customer accounts.');
      setModal(null);
      refetch();
    } catch (err) {
      toastError('Seed failed', extractError(err));
    } finally {
      setSaving(false);
    }
  }

  // Export to CSV
  function handleExportCSV() {
    if (!filtered.length) {
      toastError('Export Error', 'No customer records to export.');
      return;
    }
    const headers = [
      'customer_code',
      'first_name',
      'last_name',
      'email',
      'tenure_months',
      'support_calls',
      'late_payments',
      'satisfaction_score',
      'contract_type',
      'payment_method',
      'region',
      'churn_score',
      'risk_level',
    ];
    const rows = filtered.map((c) => [
      c.customer_code,
      c.first_name,
      c.last_name,
      c.email || '',
      c.tenure_months,
      c.support_calls || 0,
      c.late_payments || 0,
      c.satisfaction_score,
      c.contract_type,
      c.payment_method,
      c.region,
      c.latest_churn_score != null ? (c.latest_churn_score * 100).toFixed(2) + '%' : '',
      c.latest_risk_level || 'Unscored',
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const link = document.createElement('a');
    link.setAttribute('href', encodeURI(csvContent));
    link.setAttribute('download', `retentrix_customers_${new Date().toISOString().slice(0, 10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    success('Exported', 'Customer data exported to CSV.');
  }

  return (
    <div>
      <ToastContainer toasts={toasts} onDismiss={dismiss} />

      {/* Page Header */}
      <div className="page-header flex items-center justify-between" style={{ flexWrap: 'wrap', gap: 14 }}>
        <div>
          <h1>Customer Intelligence</h1>
          <p>{loading ? 'Loading database...' : `${filtered.length} of ${customers.length} total enterprise accounts`}</p>
        </div>

        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <Button
            variant="secondary"
            size="sm"
            onClick={handleExportCSV}
            leftIcon={
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <path d="M7 1v8M4 6l3 3 3-3M1 10v2a1 1 0 001 1h10a1 1 0 001-1v-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            }
          >
            Export CSV
          </Button>

          {canWrite && (
            <>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setModal('upload')}
                leftIcon={
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M7 9V1M4 4l3-3 3 3M1 10v2a1 1 0 001 1h10a1 1 0 001-1v-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                }
              >
                Upload CSV / Excel
              </Button>

              <Button
                variant="secondary"
                size="sm"
                onClick={() => setModal('seed')}
                leftIcon={
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M2 7a5 5 0 019-3M12 7a5 5 0 01-9 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    <circle cx="7" cy="7" r="1.5" fill="currentColor" />
                  </svg>
                }
              >
                Seed 50 Demo
              </Button>

              <Button
                variant="primary"
                size="sm"
                onClick={() => setModal('create')}
                leftIcon={
                  <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                    <path d="M7 1v12M1 7h12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  </svg>
                }
              >
                Add Customer
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Risk Filter Tabs */}
      <div className="risk-tabs-bar">
        {RISK_TABS.map((tab) => {
          const count =
            tab.id === ''
              ? counts.all
              : tab.id === 'Critical'
              ? counts.critical
              : tab.id === 'Moderate'
              ? counts.moderate
              : tab.id === 'Low'
              ? counts.low
              : counts.unscored;

          const isActive = riskFilter === tab.id;
          return (
            <button
              key={tab.id}
              className={`risk-tab-btn ${isActive ? 'risk-tab-btn--active' : ''}`}
              onClick={() => {
                setRisk(tab.id);
                setPage(1);
              }}
            >
              <span>{tab.label}</span>
              <span className="risk-tab-count">{count}</span>
            </button>
          );
        })}
      </div>

      {/* Search Bar */}
      <div className="customers-filters">
        <div className="search-bar">
          <svg width="15" height="15" viewBox="0 0 15 15" fill="none" className="search-icon">
            <circle cx="6.5" cy="6.5" r="5" stroke="currentColor" strokeWidth="1.4" />
            <path d="M10.5 10.5l3 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
          </svg>
          <input
            type="search"
            placeholder="Search by customer name, code, email..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="search-input"
            aria-label="Search customers"
          />
        </div>
      </div>

      {/* Data Table */}
      <div className="data-table-wrapper">
        <table className="data-table" aria-label="Customers table">
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort('code')}>
                Code <SortIcon k="code" />
              </th>
              <th className="sortable" onClick={() => toggleSort('name')}>
                Customer <SortIcon k="name" />
              </th>
              <th>Email</th>
              <th className="sortable" onClick={() => toggleSort('tenure')}>
                Tenure <SortIcon k="tenure" />
              </th>
              <th>Contract</th>
              <th className="sortable" onClick={() => toggleSort('score')}>
                Churn Score <SortIcon k="score" />
              </th>
              <th className="sortable" onClick={() => toggleSort('risk')}>
                Risk <SortIcon k="risk" />
              </th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 8 }).map((_, i) => <SkeletonRow key={i} />)
            ) : paginated.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', padding: '50px 20px', color: 'var(--text-muted)' }}>
                  <div style={{ fontSize: 32, marginBottom: 8 }}>👥</div>
                  <p style={{ margin: 0, fontWeight: 500 }}>
                    {search || riskFilter ? 'No customers match your active filter.' : 'No customer accounts yet.'}
                  </p>
                  {canWrite && !customers.length && (
                    <div style={{ display: 'flex', gap: 10, justifyContent: 'center', marginTop: 14 }}>
                      <Button variant="primary" size="sm" onClick={() => setModal('create')}>
                        Add First Customer
                      </Button>
                      <Button variant="secondary" size="sm" onClick={() => setModal('seed')}>
                        Seed 50 Demo Accounts
                      </Button>
                    </div>
                  )}
                </td>
              </tr>
            ) : (
              paginated.map((c) => (
                <tr key={c.id}>
                  <td>
                    <span className="font-mono text-sm" style={{ color: 'var(--text-muted)' }}>
                      {c.customer_code}
                    </span>
                  </td>
                  <td>
                    <button className="customer-name-btn" onClick={() => navigate(`/customers/${c.id}`)}>
                      {c.first_name} {c.last_name}
                    </button>
                  </td>
                  <td className="text-secondary text-sm">{c.email || '—'}</td>
                  <td className="text-secondary text-sm">{c.tenure_months} mo</td>
                  <td>
                    <Badge variant="default">{c.contract_type?.replace(/_/g, ' ')}</Badge>
                  </td>
                  <td>
                    {c.latest_churn_score != null ? (
                      <span
                        className="font-mono"
                        style={{
                          fontWeight: 700,
                          color:
                            (c.latest_churn_score > 1 ? c.latest_churn_score : c.latest_churn_score * 100) > 70
                              ? 'var(--brand-crimson)'
                              : (c.latest_churn_score > 1 ? c.latest_churn_score : c.latest_churn_score * 100) > 35
                              ? 'var(--brand-amber)'
                              : 'var(--brand-emerald)',
                        }}
                      >
                        {formatScorePct(c.latest_churn_score)}
                      </span>
                    ) : (
                      <span className="text-muted">—</span>
                    )}
                  </td>
                  <td>
                    <RiskBadge level={c.latest_risk_level} />
                  </td>
                  <td>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <Button size="xs" variant="ghost" onClick={() => navigate(`/predict?customerId=${c.id}`)}>
                        Predict
                      </Button>
                      {canWrite && (
                        <Button
                          size="xs"
                          variant="secondary"
                          onClick={() => {
                            setSelected(c);
                            setModal('edit');
                          }}
                        >
                          Edit
                        </Button>
                      )}
                      {isAdmin && (
                        <Button
                          size="xs"
                          variant="danger"
                          onClick={() => {
                            setSelected(c);
                            setModal('delete');
                          }}
                        >
                          Del
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="pagination">
          <Button variant="ghost" size="sm" disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
            ← Prev
          </Button>
          <span className="text-sm text-secondary">
            Page {page} of {totalPages}
          </span>
          <Button variant="ghost" size="sm" disabled={page === totalPages} onClick={() => setPage((p) => p + 1)}>
            Next →
          </Button>
        </div>
      )}

      {/* ── Modal: Create / Edit Customer ── */}
      <Modal
        open={modal === 'create' || modal === 'edit'}
        onOpenChange={(o) => {
          if (!o) {
            setModal(null);
            setSelected(null);
          }
        }}
        title={modal === 'create' ? 'Add New Customer' : `Edit — ${selected?.first_name} ${selected?.last_name}`}
        size="lg"
      >
        <CustomerForm
          initial={modal === 'edit' ? selected : null}
          onSubmit={handleSave}
          onCancel={() => {
            setModal(null);
            setSelected(null);
          }}
          loading={saving}
        />
      </Modal>

      {/* ── Modal: Bulk CSV / Excel Upload ── */}
      <Modal
        open={modal === 'upload'}
        onOpenChange={(o) => {
          if (!o) {
            setModal(null);
            setUploadFile(null);
            setUploadError(null);
            setUploadProgress(0);
          }
        }}
        title="Upload Customer Spreadsheet"
        description="Upload a CSV or Excel (.xlsx) file to batch import and run AI churn predictions."
        size="md"
      >
        <form onSubmit={handleUploadSubmit}>
          <div
            style={{
              border: `2px dashed ${uploadError ? '#ef4444' : uploadDragOver ? 'var(--brand-primary)' : 'var(--border-default)'}`,
              borderRadius: 'var(--radius-lg)',
              padding: '36px 20px',
              textAlign: 'center',
              cursor: 'pointer',
              background: uploadDragOver ? 'var(--bg-hover)' : 'var(--bg-raised)',
              transition: 'var(--transition-fast)',
              marginBottom: 12,
            }}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setUploadDragOver(true);
            }}
            onDragLeave={() => setUploadDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setUploadDragOver(false);
              const f = e.dataTransfer.files?.[0];
              if (f) { setUploadFile(f); setUploadError(null); }
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv,.xlsx,.xls"
              style={{ display: 'none' }}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) { setUploadFile(f); setUploadError(null); }
              }}
            />
            <div style={{ fontSize: 36, marginBottom: 8 }}>📁</div>
            {uploadFile ? (
              <div>
                <p style={{ margin: 0, fontWeight: 600, color: 'var(--brand-primary-h)' }}>{uploadFile.name}</p>
                <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                  {(uploadFile.size / 1024).toFixed(1)} KB — Click to change
                </p>
              </div>
            ) : (
              <div>
                <p style={{ margin: 0, fontWeight: 600, color: 'var(--text-primary)' }}>
                  Drag and drop your .csv or .xlsx file here
                </p>
                <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
                  or browse from your device
                </p>
              </div>
            )}
          </div>

          {/* Column validation error */}
          {uploadError && (
            <div style={{
              background: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.3)',
              borderRadius: 'var(--radius-md)',
              padding: '10px 14px',
              marginBottom: 12,
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
            }}>
              <span style={{ fontSize: 16, flexShrink: 0 }}>⚠️</span>
              <div>
                <p style={{ margin: 0, fontWeight: 600, fontSize: 'var(--text-sm)', color: '#ef4444' }}>Column Error</p>
                <p style={{ margin: '2px 0 0', fontSize: 'var(--text-xs)', color: '#ef4444', opacity: 0.85 }}>{uploadError}</p>
              </div>
            </div>
          )}

          {/* Upload progress bar (shows while uploading bytes) */}
          {uploading && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-secondary)' }}>
                  {uploadProgress < 100 ? 'Uploading file...' : '⚙️ Running AI predictions (this may take a few minutes)...'}
                </span>
                <span style={{ fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{uploadProgress}%</span>
              </div>
              <div style={{ height: 6, borderRadius: 3, background: 'var(--border-default)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${uploadProgress}%`,
                  background: 'linear-gradient(90deg, var(--brand-primary), var(--brand-primary-h))',
                  borderRadius: 3,
                  transition: 'width 0.3s ease',
                }} />
              </div>
            </div>
          )}

          <div style={{ background: 'var(--bg-raised)', padding: '10px 14px', borderRadius: 'var(--radius-md)', marginBottom: 16 }}>
            <p style={{ margin: '0 0 4px', fontSize: 'var(--text-xs)', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Required Columns:
            </p>
            <p style={{ margin: 0, fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              customer_code, first_name, last_name, tenure_months, satisfaction_score
            </p>
            <p style={{ margin: '4px 0 0', fontSize: 'var(--text-xs)', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
              Optional: email, support_calls, late_payments, contract_type, payment_method, region
            </p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 10 }}>
            <Button
              type="button"
              variant="secondary"
              onClick={() => {
                setModal(null);
                setUploadFile(null);
                setUploadError(null);
                setUploadProgress(0);
              }}
              disabled={uploading}
            >
              Cancel
            </Button>
            <Button type="submit" variant="primary" loading={uploading} disabled={!uploadFile || uploading}>
              Upload & Run Batch ML
            </Button>
          </div>
        </form>
      </Modal>

      {/* ── Modal: Seed Dataset Confirmation ── */}
      <Modal
        open={modal === 'seed'}
        onOpenChange={(o) => {
          if (!o) setModal(null);
        }}
        title="Seed Demo Customer Dataset"
        description="This will automatically generate 50 realistic enterprise accounts with AI churn predictions."
        size="sm"
        footer={
          <>
            <Button variant="secondary" onClick={() => setModal(null)}>
              Cancel
            </Button>
            <Button variant="primary" loading={saving} onClick={handleSeedDataset}>
              Seed 50 Accounts
            </Button>
          </>
        }
      >
        <div style={{ padding: '8px 0', fontSize: 'var(--text-sm)', color: 'var(--text-secondary)' }}>
          Populates your database with realistic contract types, regions, payment methods, and live churn risk scores.
        </div>
      </Modal>

      {/* ── Modal: Delete Confirm ── */}
      <Modal
        open={modal === 'delete'}
        onOpenChange={(o) => {
          if (!o) {
            setModal(null);
            setSelected(null);
          }
        }}
        title="Delete Customer"
        description={`Are you sure you want to permanently delete ${selected?.first_name} ${selected?.last_name}?`}
        size="sm"
        footer={
          <>
            <Button
              variant="secondary"
              onClick={() => {
                setModal(null);
                setSelected(null);
              }}
            >
              Cancel
            </Button>
            <Button variant="danger" loading={saving} onClick={handleDelete}>
              Delete permanently
            </Button>
          </>
        }
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '4px 0' }}>
          <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'rgba(239,68,68,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 6h12M8 6V4h4v2M9 10v4M11 10v4M5 6l1 11h8l1-11" stroke="#ef4444" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <p style={{ margin: 0, fontWeight: 600, color: 'var(--text-primary)' }}>
              {selected?.first_name} {selected?.last_name}
            </p>
            <p style={{ margin: 0, fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>{selected?.customer_code}</p>
          </div>
        </div>
      </Modal>
    </div>
  );
}
