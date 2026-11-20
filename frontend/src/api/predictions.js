import client from './client';

export async function predictSingle(data) {
  const res = await client.post('/predict/single', data);
  return res.data;
}

export async function runCopilot(data) {
  const res = await client.post('/predict/copilot', data);
  return res.data;
}

export async function getCustomerCopilot(customerId) {
  const res = await client.get(`/predict/copilot/${customerId}`);
  return res.data;
}

export async function sendRetentionEmail(payload) {
  const res = await client.post('/predict/send-email', payload);
  return res.data;
}

export async function sendRLFeedback(payload) {
  const res = await client.post('/predict/rl-feedback', payload);
  return res.data;
}

export async function getPredictionHistory(customerId) {
  const res = await client.get(`/predict/history/${customerId}`);
  return res.data;
}
