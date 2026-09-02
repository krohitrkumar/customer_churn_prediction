import axios from 'axios';

function getBaseUrl() {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (!envUrl) return '/api';
  
  let clean = String(envUrl).trim().replace(/^["']|["']$/g, '');
  if (!clean) return '/api';
  if (clean.startsWith('/')) return clean;
  if (!/^https?:\/\//i.test(clean)) {
    clean = `https://${clean}`;
  }
  return clean.replace(/\/+$/, '');
}

const BASE_URL = getBaseUrl();

const client = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 15000, // 15s default for regular API calls
});

// Separate client instance for file uploads — longer timeout (5 min)
export const uploadClient = axios.create({
  baseURL: BASE_URL,
  timeout: 300000, // 5 minutes for bulk ML processing
});

// ── Request interceptors: attach JWT ──
function attachJwt(config) {
  const token = localStorage.getItem('retentrix_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  localStorage.setItem('retentrix_last_active', Date.now().toString());
  return config;
}

client.interceptors.request.use(attachJwt, (error) => Promise.reject(error));
uploadClient.interceptors.request.use(attachJwt, (error) => Promise.reject(error));

// ── Response interceptor: handle 401 ──
function handle401(error) {
  if (error.response?.status === 401) {
    localStorage.removeItem('retentrix_token');
    localStorage.removeItem('retentrix_user');
    localStorage.removeItem('retentrix_last_active');
    if (!window.location.pathname.startsWith('/login')) {
      window.location.href = '/login?session=expired';
    }
  }
  return Promise.reject(error);
}

client.interceptors.response.use((r) => r, handle401);
uploadClient.interceptors.response.use((r) => r, handle401);

// ── Helper to extract user-readable error message ──
export function extractError(error) {
  const data = error?.response?.data;
  if (typeof data === 'string') return data;
  if (data?.detail) {
    if (Array.isArray(data.detail)) {
      return data.detail.map((e) => e.msg || JSON.stringify(e)).join(', ');
    }
    return data.detail;
  }
  if (data?.message) return data.message;
  if (error?.code === 'ECONNABORTED') return 'Request timed out. The server may be processing a large dataset — please try again.';
  if (error?.message) return error.message;
  return 'An unexpected error occurred.';
}

export default client;
