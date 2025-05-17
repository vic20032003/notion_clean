import requests

url = "http://localhost:8080/task"
payload = {
    "title": "🔥 FINAL TEST — Strategic Vision Launch",
    "notes": "Testing final Notion integration after property update.",
    "date": "2025-06-01"
}

print("⏳ Sending request to FastAPI...")
res = requests.post(url, json=payload)

print("✅ Status:", res.status_code)
try:
    print("📦 Response:", res.json())
except Exception as e:
    print("⚠️ Could not parse JSON:", e)
    print("Raw response:", res.text)