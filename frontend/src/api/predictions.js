import client from './client';

// Run churn prediction for a single customer
export async function predictSingle(data) {
  // data shape: { tenure_months, support_calls, late_payments, satisfaction_score,
  //               contract_type, payment_method, region, customer_id? }
  const res = await client.post('/predict/single', data);
  return res.data;
}

export async function getPredictionHistory(customerId) {
  const res = await client.get(`/predict/history/${customerId}`);
  return res.data;
}
