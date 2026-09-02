import client from './client';

// 1. Get Executive Summary KPIs
export async function getAnalyticsSummary() {
  const res = await client.get('/analytics/summary');
  return res.data;
}

// 2. Get Categorized Risk Breakdown (Contract, Region, Payment)
export async function getRiskBreakdown() {
  const res = await client.get('/analytics/risk_breakdown');
  return res.data;
}

// 3. Get Top At-Risk Accounts
export async function getTopAtRisk(limit = 10) {
  const res = await client.get('/analytics/top_at_risk', { params: { limit } });
  return res.data;
}

// 4. Get Recent Prediction Logs
export async function getRecentActivity(limit = 15) {
  const res = await client.get('/analytics/recent_activity', { params: { limit } });
  return res.data;
}
