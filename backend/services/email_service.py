import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from database.config import settings

class EmailService:
    @staticmethod
    def send_otp_email(to_email: str, otp_code: str) -> bool:
        plain_text = f"""Hello,

Your 6-digit verification code for Retentrix is: {otp_code}

This code will expire in 10 minutes.

Security Notice:
- Do not share this code with anyone.
- Retentrix support staff will never ask for your verification code.
- If you did not request this verification code, please ignore this email.

Best regards,
The Retentrix AI Security Team
Contact on GitHub: https://github.com/krohitrkumar
"""

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retentrix Verification Code</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #1e293b;">
    <div style="width: 100%; background-color: #f1f5f9; padding: 40px 15px; box-sizing: border-box;">
        <div style="max-width: 540px; margin: 0 auto; background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05); border: 1px solid #e2e8f0;">
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%); padding: 36px 30px; text-align: center;">
                <div style="display: inline-block; background: rgba(255, 255, 255, 0.12); border: 1px solid rgba(255, 255, 255, 0.2); padding: 6px 16px; border-radius: 9999px; color: #60a5fa; font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 12px;">Retentrix AI Engine</div>
                <h1 style="margin: 0; color: #ffffff; font-size: 24px; font-weight: 700;">Authentication Security</h1>
                <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 13px;">AI Customer Churn Intelligence & Retention Platform</p>
            </div>
            <div style="padding: 36px 32px 28px 32px;">
                <h2 style="font-size: 18px; font-weight: 600; color: #0f172a; margin-top: 0; margin-bottom: 10px;">Verify Your Account</h2>
                <p style="font-size: 14px; line-height: 1.6; color: #475569; margin-bottom: 24px;">
                    You recently initiated an authentication request for your Retentrix account. 
                    Please enter the 6-digit verification code below to securely verify your identity:
                </p>
                <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border: 2px dashed #cbd5e1; border-radius: 12px; padding: 22px; text-align: center; margin: 20px 0;">
                    <div style="font-size: 11px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">One-Time Verification Code</div>
                    <div style="text-align: center; margin: 0; padding: 0;">
                        <span style="font-family: 'Courier New', Courier, monospace; font-size: 26px; font-weight: 800; letter-spacing: 4px; color: #2563eb; white-space: nowrap !important; display: inline-block; padding: 6px 18px; background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px;">{otp_code}</span>
                    </div>
                </div>
                <div style="background-color: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 14px 16px; margin: 22px 0;">
                    <p style="margin: 0; font-size: 13px; color: #92400e; line-height: 1.5;">⏳ <b style="color: #78350f;">Time Sensitive:</b> This verification code will expire in <b style="color: #78350f;">10 minutes</b>. Once expired, you will need to request a new code.</p>
                </div>
                <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 18px; margin: 20px 0;">
                    <h4 style="margin: 0 0 8px 0; font-size: 13px; color: #334155; font-weight: 600;">🔒 Security Guidelines:</h4>
                    <ul style="margin: 0; padding-left: 18px; color: #64748b; font-size: 12px; line-height: 1.6;">
                        <li>Never share this code with anyone, including Retentrix staff.</li>
                        <li>We will never call or message you asking for your verification code.</li>
                        <li>If you did not request this code, your credentials are safe, but you should review your account security.</li>
                    </ul>
                </div>
                <div style="border-top: 1px solid #e2e8f0; padding-top: 18px; margin-top: 22px;">
                    <p style="font-size: 12px; color: #64748b; line-height: 1.5; margin: 0;">
                        Need help or experiencing issues? Contact on GitHub at 
                        <a href="https://github.com/krohitrkumar" style="color: #2563eb; text-decoration: none; font-weight: 600;">github.com/krohitrkumar</a>.
                    </p>
                </div>
            </div>
            <div style="background-color: #f8fafc; padding: 22px 30px; text-align: center; border-top: 1px solid #e2e8f0;">
                <p style="margin: 4px 0; font-size: 11px; color: #94a3b8;">© 2026 Retentrix Intelligence Systems Inc. All rights reserved.</p>
                <p style="margin: 4px 0; font-size: 11px; color: #94a3b8;">Enterprise Machine Learning Platform for Predictive Customer Retention.</p>
            </div>
        </div>
    </div>
</body>
</html>"""

        # 1. Primary: Direct Gmail SMTP via Port 465 SSL (Sends to ANY email address in the world)
        if settings.SMTP_USER and settings.SMTP_PASSWORD:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"{otp_code} is your Retentrix verification code"
                msg["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_USER}>"
                msg["To"] = to_email

                msg.attach(MIMEText(plain_text, "plain"))
                msg.attach(MIMEText(html_content, "html"))

                try:
                    with smtplib.SMTP_SSL(settings.SMTP_HOST, 465, timeout=5) as server:
                        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                        server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
                    return True
                except Exception:
                    pass

                try:
                    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT or 587, timeout=5) as server:
                        server.starttls()
                        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                        server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
                    return True
                except Exception:
                    pass
            except Exception:
                pass

        # 2. Secondary: Brevo API (if configured)
        if getattr(settings, "BREVO_API_KEY", ""):
            try:
                sender_email = settings.SMTP_USER or "retentrix.security@gmail.com"
                res = requests.post(
                    "https://api.brevo.com/v3/smtp/email",
                    headers={
                        "api-key": settings.BREVO_API_KEY,
                        "Content-Type": "application/json"
                    },
                    json={
                        "sender": {"name": settings.SMTP_FROM_NAME, "email": sender_email},
                        "to": [{"email": to_email}],
                        "subject": f"{otp_code} is your Retentrix verification code",
                        "htmlContent": html_content
                    },
                    timeout=3
                )
                if res.status_code in [200, 201]:
                    return True
            except Exception:
                pass

        # 3. Tertiary: Resend API (if configured)
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
                        "html": html_content
                    },
                    timeout=3
                )
                if res.status_code in [200, 201]:
                    return True
            except Exception:
                pass

        print(f"[AUTH_OTP_DISPATCH] Verification code for {to_email}: {otp_code}")
        return True

    @staticmethod
    def send_retention_email(to_email: str, subject: str, body: str, sender_name: str = None, reply_to: str = None) -> bool:
        if not to_email or not settings.SMTP_USER or not settings.SMTP_PASSWORD:
            return False

        try:
            msg = MIMEMultipart()
            display_from = f"{sender_name} via {settings.SMTP_FROM_NAME}" if sender_name else settings.SMTP_FROM_NAME
            msg["From"] = f"{display_from} <{settings.SMTP_USER}>"
            msg["To"] = to_email
            msg["Subject"] = subject
            if reply_to:
                msg["Reply-To"] = reply_to
            msg.attach(MIMEText(body, "plain"))

            try:
                with smtplib.SMTP_SSL(settings.SMTP_HOST, 465, timeout=4) as server:
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
                return True
            except Exception:
                pass

            try:
                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT or 587, timeout=4) as server:
                    server.starttls()
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
                return True
            except Exception:
                return False

        except Exception:
            return False

email_service = EmailService()
