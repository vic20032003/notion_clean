import requests

token = ""  # Use your real token here
db_id = ""

headers = {
    "Authorization": f"Bearer {token}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

url = f"https://api.notion.com/v1/databases/{db_id}/query"

response = requests.post(url, headers=headers)
print(response.status_code)
print(response.text)