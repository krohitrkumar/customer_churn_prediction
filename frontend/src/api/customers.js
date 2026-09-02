import client, { uploadClient } from './client';

export async function getCustomers({ skip = 0, limit = 500 } = {}) {
  const res = await client.get('/customers/', { params: { skip, limit } });
  return res.data;
}

export async function getCustomer(id) {
  const res = await client.get(`/customers/${id}`);
  return res.data;
}

export async function createCustomer(data) {
  const res = await client.post('/customers/', data);
  return res.data;
}

export async function updateCustomer(id, data) {
  const res = await client.put(`/customers/${id}`, data);
  return res.data;
}

export async function deleteCustomer(id) {
  const res = await client.delete(`/customers/${id}`);
  return res.data;
}

// 1-Click Database Seed
export async function seedCustomers(count = 50) {
  const res = await client.post('/customers/seed', null, {
    params: { count },
    timeout: 120000, // 2 min for seed (50 × ML predictions)
  });
  return res.data;
}

// Bulk CSV / Excel Upload — uses uploadClient (5-min timeout)
export async function uploadCustomerFile(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await uploadClient.post('/customers/upload_file', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress
      ? (e) => {
          const pct = Math.round((e.loaded * 100) / (e.total || e.loaded));
          onProgress(pct);
        }
      : undefined,
  });
  return res.data;
}
