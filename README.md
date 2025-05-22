# Echo Notion–Telegram Integration Service

This FastAPI service connects a Telegram bot to your Notion workspace. It parses natural‑language commands (e.g. set reminders, create tasks/events/notes) via OpenAI, and then writes pages directly into your Notion database.

## Environment Variables

You can configure the service locally with a `.env` file (using `python-dotenv`) or via your hosting platform (e.g. Render) by setting the same keys.

| Variable             | Description                                                                                          |
|----------------------|------------------------------------------------------------------------------------------------------|
| `NOTION_TOKEN`       | Your Notion integration secret (scope: read/write pages and databases).                                |
| `NOTION_DATABASE_ID` | The ID of your primary Notion database (the part before `?v=` in the database URL).                   |
| `NOTION_CONTACTS_ID` | *(optional)* ID of a secondary contacts database (if you use contact‑management features).           |
| `NOTION_FEEDBACK_ID` | *(optional)* ID of a feedback database (defaults to `NOTION_DATABASE_ID` if unset).                  |
| `OPENAI_API_KEY`     | Your OpenAI API key (used for intent parsing and message replies).                                   |
| `TELEGRAM_TOKEN`     | Your Telegram Bot token (set up via BotFather).                                                      |

## Notion Database Schema

Your main database should at minimum include these properties:

- **Title** (Title)
- **Type** (Select) – options include `User Message`, `Task`, `Event`, `Note`, `Reminder`, `Feedback`
- **Tags** (Multi‑select)
- **Date** (Date)
- **Chat ID** (Rich text)

Make sure you've added your integration to the database and granted it the proper scopes.

## Local Development

1. Copy or create a `.env` in the project root:
   ```ini
   NOTION_TOKEN=ntn_xxxYOUR_TOKENxxx
   NOTION_DATABASE_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
   OPENAI_API_KEY=sk-…YOUR_OPENAI_KEY…
   TELEGRAM_TOKEN=123456:ABC‑…YOUR_TELEGRAM_BOT_TOKEN…
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn webhook_server:app --reload
   ```
4. (Local testing) Use a tunneling tool (e.g. ngrok) to expose `/telegram` and set your bot webhook:
   ```bash
   ngrok http 8000
   curl "https://api.telegram.org/bot$TELEGRAM_TOKEN/setWebhook?url=https://<your-tunnel>/telegram"
   ```

## Deployment on Render

1. In your Render dashboard, go to **Environment → Environment Variables** and add the same keys/values.
2. Deploy or redeploy your service.
3. Set your Telegram webhook to `https://<your-render-app>/telegram` (see above).

## Debugging Notion Writes

The `add_to_notion`, `update_notion_page`, and `archive_notion_page` helpers now log any HTTP errors from Notion to the console. Check your service logs (locally or in Render) for messages like:

```text
🔴 Notion add page failed (status 400): {error details}
```

If pages still don’t appear, verify:

- Your database schema matches the required properties.
- The `NOTION_DATABASE_ID` is correct (copy it from the URL before `?v=`).
- Your integration has been shared with the database and has read/write permissions.
