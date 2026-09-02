# backend/services/email_service.py
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database.config import settings

class EmailService:
    @staticmethod
    def send_otp_email(to_email: str, otp_code: str) -> bool:
        """Sends a high-end, responsive HTML email containing the 6-digit OTP code."""
        # 🟢 Priority 1: Resend HTTP API (HTTPS Port 443 — NEVER blocked on cloud hosting)
        if getattr(settings, "RESEND_API_KEY", ""):
            try:
                res = requests.post(
                    "https://api.resend.com/emails",
                    headers={
                        "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "from": f"{settings.SMTP_FROM_NAME} <onboarding@resend.dev>",
                        "to": [to_email],
                        "subject": f"{otp_code} is your Retentrix verification code",
                        "html": f"<div style='font-family:sans-serif;padding:24px;background:#0d1117;color:#fff;border-radius:8px;'><h2 style='color:#6366f1;'>Retentrix Verification Code</h2><p>Your 6-digit code is:</p><h1 style='letter-spacing:6px;color:#10b981;font-size:32px;'>{otp_code}</h1><p>Valid for 10 minutes.</p></div>"
                    },
                    timeout=5
                )
                if res.status_code in [200, 201]:
                    print(f"✅ OTP email delivered via Resend API to {to_email}")
                    return True
                else:
                    print(f"⚠️ Resend API returned {res.status_code}: {res.text}")
            except Exception as e:
                print(f"⚠️ Resend API error: {e}")

        # 🟢 Priority 2: Fallback for local development when SMTP credentials are not yet configured
        if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
            print(f"\n==================================================================")
            print(f"📧 [DEV MODE OTP] To: {to_email} | 6-Digit Code: {otp_code}")
            print(f"==================================================================\n")
            return True

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{otp_code} is your Retentrix verification code"
            msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_USER}>"
            msg["To"] = to_email

            # 1. Plain-text version (for low-bandwidth / legacy email clients)
            plain_text = f"""
Hello,

Your 6-digit verification code for Retentrix is: {otp_code}

This code will expire in 10 minutes.

Security Notice:
- Do not share this code with anyone.
- Retentrix support staff will never ask for your verification code.

If you did not request this verification code, please ignore this email or contact our support at support@retentrix.ai.

Best regards,
The Retentrix AI Security Team
https://retentrix.ai
            """

            # 2. Premium Responsive HTML version
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Retentrix Verification Code</title>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        background-color: #f1f5f9;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                        color: #1e293b;
                        -webkit-font-smoothing: antialiased;
                    }}
                    .wrapper {{
                        width: 100%;
                        background-color: #f1f5f9;
                        padding: 40px 15px;
                        box-sizing: border-box;
                    }}
                    .container {{
                        max-width: 540px;
                        margin: 0 auto;
                        background-color: #ffffff;
                        border-radius: 16px;
                        overflow: hidden;
                        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
                        border: 1px solid #e2e8f0;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
                        padding: 36px 30px;
                        text-align: center;
                    }}
                    .logo-badge {{
                        display: inline-block;
                        background: rgba(255, 255, 255, 0.12);
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        padding: 6px 16px;
                        border-radius: 9999px;
                        color: #60a5fa;
                        font-size: 11px;
                        font-weight: 700;
                        letter-spacing: 1.5px;
                        text-transform: uppercase;
                        margin-bottom: 12px;
                    }}
                    .header h1 {{
                        margin: 0;
                        color: #ffffff;
                        font-size: 24px;
                        font-weight: 700;
                        letter-spacing: -0.5px;
                    }}
                    .header p {{
                        margin: 6px 0 0 0;
                        color: #94a3b8;
                        font-size: 13px;
                    }}
                    .content {{
                        padding: 36px 32px 28px 32px;
                    }}
                    .greeting {{
                        font-size: 18px;
                        font-weight: 600;
                        color: #0f172a;
                        margin-top: 0;
                        margin-bottom: 10px;
                    }}
                    .description {{
                        font-size: 14px;
                        line-height: 1.6;
                        color: #475569;
                        margin-bottom: 28px;
                    }}
                    .otp-card {{
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                        border: 2px dashed #cbd5e1;
                        border-radius: 12px;
                        padding: 24px;
                        text-align: center;
                        margin: 25px 0;
                    }}
                    .otp-label {{
                        font-size: 11px;
                        font-weight: 700;
                        color: #64748b;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        margin-bottom: 8px;
                    }}
                    .otp-code {{
                        font-family: 'Courier New', Courier, monospace;
                        font-size: 38px;
                        font-weight: 800;
                        letter-spacing: 8px;
                        color: #2563eb;
                        margin: 4px 0 0 0;
                        user-select: all;
                    }}
                    .info-box {{
                        background-color: #fffbeb;
                        border-left: 4px solid #f59e0b;
                        border-radius: 6px;
                        padding: 14px 16px;
                        margin: 24px 0;
                    }}
                    .info-box p {{
                        margin: 0;
                        font-size: 13px;
                        color: #92400e;
                        line-height: 1.5;
                    }}
                    .info-box b {{
                        color: #78350f;
                    }}
                    .security-tips {{
                        background-color: #f8fafc;
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        padding: 16px 18px;
                        margin: 20px 0;
                    }}
                    .security-tips h4 {{
                        margin: 0 0 8px 0;
                        font-size: 13px;
                        color: #334155;
                        font-weight: 600;
                    }}
                    .security-tips ul {{
                        margin: 0;
                        padding-left: 18px;
                        color: #64748b;
                        font-size: 12px;
                        line-height: 1.6;
                    }}
                    .support-card {{
                        border-top: 1px solid #e2e8f0;
                        padding-top: 20px;
                        margin-top: 24px;
                    }}
                    .support-text {{
                        font-size: 12px;
                        color: #64748b;
                        line-height: 1.5;
                        margin: 0;
                    }}
                    .support-text a {{
                        color: #2563eb;
                        text-decoration: none;
                        font-weight: 600;
                    }}
                    .footer {{
                        background-color: #f8fafc;
                        padding: 24px 30px;
                        text-align: center;
                        border-top: 1px solid #e2e8f0;
                    }}
                    .footer p {{
                        margin: 4px 0;
                        font-size: 11px;
                        color: #94a3b8;
                    }}
                    .footer a {{
                        color: #64748b;
                        text-decoration: underline;
                    }}
                </style>
            </head>
            <body>
                <div class="wrapper">
                    <div class="container">
                        <!-- Brand Header -->
                        <div class="header">
                            <div class="logo-badge">Retentrix AI Engine</div>
                            <h1>Authentication Security</h1>
                            <p>AI Customer Churn Intelligence & Retention Platform</p>
                        </div>

                        <!-- Main Body Content -->
                        <div class="content">
                            <h2 class="greeting">Verify Your Account</h2>
                            <p class="description">
                                You recently initiated an authentication request for your Retentrix account. 
                                Please enter the 6-digit verification code below to securely verify your identity:
                            </p>

                            <!-- OTP Box -->
                            <div class="otp-card">
                                <div class="otp-label">One-Time Verification Code</div>
                                <div class="otp-code">{otp_code}</div>
                            </div>

                            <!-- Expiry Timer Notice -->
                            <div class="info-box">
                                <p>⏳ <b>Time Sensitive:</b> This verification code will expire in <b>10 minutes</b>. Once expired, you will need to request a new code.</p>
                            </div>

                            <!-- Security Notice -->
                            <div class="security-tips">
                                <h4>🔒 Security Guidelines:</h4>
                                <ul>
                                    <li>Never share this code with anyone, including Retentrix staff.</li>
                                    <li>We will never call or message you asking for your verification code.</li>
                                    <li>If you did not request this code, your credentials may be safe, but you should review your account security.</li>
                                </ul>
                            </div>

                            <!-- Support Helpdesk Details -->
                            <div class="support-card">
                                <p class="support-text">
                                    Need help or experiencing login issues? Our 24/7 Customer Success Team is here to assist you at 
                                    <a href="mailto:support@retentrix.ai">support@retentrix.ai</a>.
                                </p>
                            </div>
                        </div>

                        <!-- Corporate Footer -->
                        <div class="footer">
                            <p>© 2026 Retentrix Intelligence Systems Inc. All rights reserved.</p>
                            <p>Enterprise Machine Learning Platform for Predictive Customer Retention.</p>
                            <p>This is an automated security transmission. Please do not reply directly to this email.</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """

            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            # Attempt 1: Direct SSL (Port 465) - fast and universally allowed on cloud platforms
            try:
                with smtplib.SMTP_SSL(settings.SMTP_HOST, 465, timeout=3) as server:
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
                print(f"✅ OTP email successfully delivered via SSL (Port 465) to {to_email}")
                return True
            except Exception as ssl_err:
                print(f"⚠️ Port 465 SSL failed ({ssl_err}), trying Port 587 TLS...")

            # Attempt 2: STARTTLS (Port 587) fallback
            try:
                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT or 587, timeout=3) as server:
                    server.starttls()
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
                print(f"✅ OTP email successfully delivered via TLS (Port 587) to {to_email}")
                return True
            except Exception as tls_err:
                print(f"❌ Port 587 TLS failed ({tls_err})")

            return False

        except Exception as e:
            print(f"❌ Failed to send email: {e}")
            return False

email_service = EmailService()