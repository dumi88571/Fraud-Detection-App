# FraudGuard AI

A web app that checks if a bank transaction looks fraudulent or not. You can test one transaction at a time, upload a whole file of transactions, and view charts showing how many were flagged.

---

## The Problem

When someone makes a payment, there's no easy way to know if it's real or fraud — at least not quickly. Checking transactions by hand takes time and people miss things. This app helps by scoring each transaction automatically and telling you whether it's low, medium, or high risk.

It's useful for:
- Fraud analysts who want a second opinion on a suspicious transaction
- Developers building payment tools who need a scoring system to plug into
- Anyone who wants to understand how fraud detection actually works

---

## Why It's Helpful

**It's fast.** You can upload hundreds of transactions in a CSV file and get results in seconds. Doing that by hand would take hours.

**It's consistent.** Every transaction goes through the same checks — the score doesn't change based on who's looking at it or how tired they are.

**It explains itself.** When a transaction is flagged, the app tells you why — things like "high device risk" or "transaction happened at 2am." You're not just getting a red flag with no reason.

**It keeps a record.** Every transaction you score gets saved, so you can go back and look at history.

**Your data stays with you.** Everything runs on your own machine or server. Nothing gets sent to a third party.

---

## Risk Levels

- **LOW** — Looks fine. Nothing unusual.
- **MEDIUM** — Something is a bit off. Worth a closer look.
- **HIGH** — Strong signs of fraud. Act on this one.

---

## What's Inside

| Page | What it does |
|---|---|
| Dashboard | Shows a quick summary of all transactions scored so far |
| Predict | Score one transaction by filling in a form |
| Batch Predict | Upload a CSV file and score many transactions at once |
| Flagged | Shows only the HIGH and MEDIUM risk transactions |
| Analytics | Charts showing fraud trends over time |
| Model Info | Shows how the model works and its performance numbers |

---

## How to Run It

You need Python 3.11 or newer.

```bash
# Install the required packages
pip install -r requirements.txt

# Start the app
python app.py
```

Then open your browser and go to `http://localhost:5000`.

---

## Deploying to Render

1. Push your code to a GitHub repo
2. Go to [render.com](https://render.com) and create a new **Web Service**
3. Connect it to your GitHub repo
4. Use these settings:

| Setting | Value |
|---|---|
| Build Command | `pip install -r requirements.txt` |
| Start Command | `gunicorn app:app` |
| Python Version | 3.11 |

5. Hit **Deploy** and wait for it to go live

> The database is created automatically when the app starts, so you don't need to set that up separately. On Render's free plan, the database resets when the app restarts — that's normal for free hosting.

---

## API

If you want to connect another system to this app, you can send a transaction as JSON and get a score back.

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 42,
    "amount": 899.99,
    "transaction_type": "online",
    "merchant_category": "electronics",
    "country": "US",
    "hour": 2,
    "device_risk_score": 0.85,
    "ip_risk_score": 0.72
  }'
```

Response:
```json
{
  "prediction": 1,
  "fraud_probability": 0.83,
  "risk_level": "HIGH",
  "confidence": 0.83,
  "transaction_id": "some-id",
  "timestamp": "2025-04-30T10:00:00"
}
```

---

## Common Issues

**App won't start** — Run `pip install -r requirements.txt` first, then try again.

**Charts are empty** — The analytics page shows sample data until you've scored some real transactions. Run a few predictions and then check back.

**Render build fails** — Make sure your `Procfile` says `web: gunicorn app:app` and that `gunicorn` is listed in `requirements.txt`.

---

## License

MIT. Use it however you like.
