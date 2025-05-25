import secrets

new_api_key = secrets.token_urlsafe(32)
print(new_api_key)