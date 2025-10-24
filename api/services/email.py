"""Email sending service with SMTP support."""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def _get_smtp_config():
    """Get SMTP configuration from environment variables."""
    return {
        "host": os.getenv("SMTP_HOST", ""),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "username": os.getenv("SMTP_USERNAME", ""),
        "password": os.getenv("SMTP_PASSWORD", ""),
        "from_email": os.getenv("SMTP_FROM_EMAIL", os.getenv("SMTP_USERNAME", "noreply@anagnosis.app")),
        "from_name": os.getenv("SMTP_FROM_NAME", "Anagnosis"),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"},
    }

def is_email_configured() -> bool:
    """Check if email/SMTP is properly configured."""
    config = _get_smtp_config()
    return bool(config["host"] and config["username"] and config["password"])

def send_email(to_email: str, subject: str, html_body: str, text_body: Optional[str] = None) -> bool:
    """
    Send an email via SMTP.
    
    Returns True if sent successfully, False otherwise.
    """
    if not is_email_configured():
        logger.warning("Email not configured. Skipping email send.")
        # In dev mode, just log the email content
        logger.info(f"[DEV] Would send email to {to_email}:\nSubject: {subject}\n{text_body or html_body}")
        return False
    
    config = _get_smtp_config()
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{config['from_name']} <{config['from_email']}>"
        msg["To"] = to_email
        
        if text_body:
            msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        
        with smtplib.SMTP(config["host"], config["port"]) as server:
            if config["use_tls"]:
                server.starttls()
            server.login(config["username"], config["password"])
            server.send_message(msg)
        
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {to_email}: {e}")
        return False

def send_verification_email(to_email: str, token: str, base_url: str) -> bool:
    """Send email verification link."""
    verification_url = f"{base_url}/verify-email?token={token}"
    
    subject = "Verify your Anagnosis account"
    
    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
          <h2 style="color: #5b7dff;">Welcome to Anagnosis!</h2>
          <p>Thank you for signing up. Please verify your email address by clicking the link below:</p>
          <p style="margin: 30px 0;">
            <a href="{verification_url}" 
               style="background-color: #5b7dff; color: white; padding: 12px 30px; 
                      text-decoration: none; border-radius: 5px; display: inline-block;">
              Verify Email Address
            </a>
          </p>
          <p style="color: #666; font-size: 14px;">
            Or copy and paste this link into your browser:<br>
            <a href="{verification_url}">{verification_url}</a>
          </p>
          <p style="color: #666; font-size: 14px; margin-top: 30px;">
            This link will expire in 24 hours. If you didn't create an account, you can safely ignore this email.
          </p>
        </div>
      </body>
    </html>
    """
    
    text_body = f"""
Welcome to Anagnosis!

Thank you for signing up. Please verify your email address by visiting:

{verification_url}

This link will expire in 24 hours. If you didn't create an account, you can safely ignore this email.
    """
    
    return send_email(to_email, subject, html_body, text_body)
