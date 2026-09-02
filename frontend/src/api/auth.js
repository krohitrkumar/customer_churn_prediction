import client from './client';

// Register new user
export async function registerUser(data) {
  const res = await client.post('/auth/register', data);
  return res.data;
}

// Login — MUST use form-data (application/x-www-form-urlencoded)
export async function loginUser({ email, password }) {
  const params = new URLSearchParams();
  params.append('username', email);
  params.append('password', password);
  const res = await client.post('/auth/login', params, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });
  return res.data;
}

// Get current user profile
export async function getMe() {
  const res = await client.get('/auth/me');
  return res.data;
}

// Send OTP to email
export async function sendOtp(email) {
  const res = await client.post('/auth/send_otp', { email });
  return res.data;
}

// Verify OTP code
export async function verifyOtp(email, otp_code) {
  const res = await client.post('/auth/verify_otp', { email, otp_code });
  return res.data;
}

// Change password using old password
export async function changePasswordWithOld(currentPassword, newPassword) {
  const res = await client.post('/auth/change_password', {
    current_password: currentPassword,
    new_password: newPassword,
  });
  return res.data;
}

// Reset password using email OTP
export async function resetPasswordWithOtp(email, otp_code, newPassword) {
  const res = await client.post('/auth/reset_password', {
    email,
    otp_code,
    new_password: newPassword,
  });
  return res.data;
}
