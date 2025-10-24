# SMTP Email Setup Guide

This guide will help you configure email sending for user verification emails.

## Quick Setup Options

### Option 1: Gmail (Easiest for testing)

1. **Enable 2-Factor Authentication** on your Gmail account
   - Go to https://myaccount.google.com/security
   - Enable 2-Step Verification

2. **Create an App Password**
   - Go to https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Name it "Anagnosis"
   - Copy the 16-character password

3. **Add to your `.env` file:**
   ```bash
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-16-char-app-password
   SMTP_FROM_EMAIL=your-email@gmail.com
   SMTP_FROM_NAME=Anagnosis
   SMTP_USE_TLS=true
   BASE_URL=http://localhost:8080  # or your production URL
   ```

### Option 2: SendGrid (Best for production)

1. **Sign up** at https://sendgrid.com (free tier: 100 emails/day)

2. **Create an API key**
   - Go to Settings → API Keys
   - Create API Key with "Mail Send" permissions
   - Copy the API key

3. **Add to your `.env` file:**
   ```bash
   SMTP_HOST=smtp.sendgrid.net
   SMTP_PORT=587
   SMTP_USERNAME=apikey
   SMTP_PASSWORD=your-sendgrid-api-key
   SMTP_FROM_EMAIL=noreply@yourdomain.com
   SMTP_FROM_NAME=Anagnosis
   SMTP_USE_TLS=true
   BASE_URL=https://yourdomain.com
   ```

### Option 3: Mailgun (Alternative for production)

1. **Sign up** at https://www.mailgun.com (free tier: 5,000 emails/month for 3 months)

2. **Get SMTP credentials**
   - Go to Sending → Domain Settings → SMTP credentials
   - Note your SMTP login and password

3. **Add to your `.env` file:**
   ```bash
   SMTP_HOST=smtp.mailgun.org
   SMTP_PORT=587
   SMTP_USERNAME=postmaster@your-mailgun-domain.mailgun.org
   SMTP_PASSWORD=your-mailgun-smtp-password
   SMTP_FROM_EMAIL=noreply@your-mailgun-domain.mailgun.org
   SMTP_FROM_NAME=Anagnosis
   SMTP_USE_TLS=true
   BASE_URL=https://yourdomain.com
   ```

### Option 4: AWS SES (Production at scale)

1. **Set up AWS SES** in your AWS account
2. **Verify your domain** or email address
3. **Get SMTP credentials** from SES console

4. **Add to your `.env` file:**
   ```bash
   SMTP_HOST=email-smtp.us-east-1.amazonaws.com  # your region
   SMTP_PORT=587
   SMTP_USERNAME=your-ses-smtp-username
   SMTP_PASSWORD=your-ses-smtp-password
   SMTP_FROM_EMAIL=noreply@yourdomain.com
   SMTP_FROM_NAME=Anagnosis
   SMTP_USE_TLS=true
   BASE_URL=https://yourdomain.com
   ```

## Testing Your Setup

### Local Testing (http://localhost:8080)

1. Add SMTP settings to `.env`
2. Restart the server:
   ```bash
   pkill -f "uvicorn serve:app"
   FORCE_SQLITE=1 uvicorn serve:app --host 0.0.0.0 --port 8080
   ```
3. Sign up with a real email address
4. Check your inbox for the verification email

### Production Testing (with Docker)

1. Add SMTP settings to `.env` on the server
2. Restart containers:
   ```bash
   docker compose down
   docker compose up -d
   ```
3. Sign up and check email

## Troubleshooting

### Emails not sending

Check the server logs:
```bash
# Local
# Check terminal output for errors

# Docker
docker compose logs -f api | grep -i email
```

Common issues:
- **"Authentication failed"**: Check username/password
- **"Connection refused"**: Check SMTP_HOST and SMTP_PORT
- **"TLS error"**: Try setting SMTP_USE_TLS=false (port 25 or 465)

### Emails going to spam

For production:
1. Set up SPF records for your domain
2. Set up DKIM signing (provider-specific)
3. Use a reputable SMTP service (SendGrid, Mailgun, SES)
4. Use a verified domain for SMTP_FROM_EMAIL

### Still not working?

Temporarily disable verification requirement:
```bash
# Add to .env
REQUIRE_EMAIL_VERIFICATION=false
```

Users can sign up and log in without verification (not recommended for production).

## Environment Variables Reference

```bash
# Required for email sending
SMTP_HOST=smtp.gmail.com              # SMTP server hostname
SMTP_PORT=587                          # Port (587 for TLS, 465 for SSL, 25 for plain)
SMTP_USERNAME=your-email@gmail.com    # SMTP username
SMTP_PASSWORD=your-password           # SMTP password or app password
SMTP_USE_TLS=true                     # Use TLS encryption (recommended)

# Optional (with sensible defaults)
SMTP_FROM_EMAIL=noreply@yourdomain.com  # "From" email (defaults to SMTP_USERNAME)
SMTP_FROM_NAME=Anagnosis                # "From" name (defaults to "Anagnosis")
BASE_URL=https://yourdomain.com         # Base URL for verification links

# Email verification behavior
REQUIRE_EMAIL_VERIFICATION=true         # Require verification before login (default: true)
```

## Current Behavior

**Without SMTP configured:**
- Verification links are logged to server console
- Look for lines like: `[DEV] Email verification link for user@example.com: ...`
- You can manually copy/paste the link

**With SMTP configured:**
- Verification emails are sent automatically
- Beautiful HTML email with branded template
- Link expires in 24 hours
