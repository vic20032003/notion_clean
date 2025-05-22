import unittest
import sqlite3
from datetime import datetime
import os
import requests
from dotenv import load_dotenv

# Load Notion credentials from environment
load_dotenv()
notion_token = os.getenv("NOTION_TOKEN")
database_id = os.getenv("NOTION_DATABASE_ID")

class TestChatMemoryDB(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        self.cursor = self.conn.cursor()
        # Create test tables
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                chat_id TEXT,
                sender TEXT,
                text TEXT,
                timestamp TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS privacy_optout (
                chat_id TEXT PRIMARY KEY,
                timestamp TEXT
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                chat_id TEXT,
                message TEXT,
                rating TEXT,
                timestamp TEXT
            )
        ''')

    def test_store_message(self):
        chat_id = "test123"
        sender = "tester"
        text = "hello world"
        self.cursor.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (chat_id, sender, text, datetime.now().isoformat())
        )
        self.cursor.execute("SELECT * FROM messages WHERE chat_id = ?", (chat_id,))
        result = self.cursor.fetchone()
        self.assertEqual(result[0], chat_id)
        self.assertEqual(result[1], sender)
        self.assertEqual(result[2], text)

    def test_get_recent_messages(self):
        messages = [
            ("test123", "user1", "first message"),
            ("test123", "user2", "second message"),
            ("test123", "user1", "third message")
        ]
        for msg in messages:
            self.cursor.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                (*msg, datetime.now().isoformat())
            )
        self.cursor.execute("SELECT * FROM messages WHERE chat_id = ?", ("test123",))
        results = self.cursor.fetchall()
        self.assertEqual(len(results), 3)

    def test_filtered_messages(self):
        filtered_words = {"spam", "scam", "buy now"}
        test_messages = [
            ("Hello world", False),
            ("This is spam message", True),
            ("buy now please", True),
            ("Regular message", False)
        ]
        for message, should_filter in test_messages:
            is_filtered = any(word in message.lower() for word in filtered_words)
            self.assertEqual(is_filtered, should_filter)

    def test_privacy_optout(self):
        chat_id = "test456"
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO privacy_optout VALUES (?, ?)",
            (chat_id, timestamp)
        )
        self.cursor.execute("SELECT * FROM privacy_optout WHERE chat_id = ?", (chat_id,))
        result = self.cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], chat_id)

    def test_store_feedback(self):
        chat_id = "test789"
        message = "Great response!"
        rating = "positive"
        self.cursor.execute(
            "INSERT INTO feedback VALUES (?, ?, ?, ?)",
            (chat_id, message, rating, datetime.now().isoformat())
        )
        self.cursor.execute("SELECT * FROM feedback WHERE chat_id = ?", (chat_id,))
        result = self.cursor.fetchone()
        self.assertEqual(result[0], chat_id)
        self.assertEqual(result[1], message)
        self.assertEqual(result[2], rating)

    def test_search_messages(self):
        messages = [
            ("test123", "user1", "hello world"),
            ("test123", "user2", "testing search"),
            ("test123", "user1", "important message"),
            ("test456", "user3", "different chat")
        ]
        for msg in messages:
            self.cursor.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                (*msg, datetime.now().isoformat())
            )
        self.cursor.execute(
            "SELECT * FROM messages WHERE text LIKE ?",
            ('%search%',)
        )
        search_results = self.cursor.fetchall()
        self.assertEqual(len(search_results), 1)
        self.assertEqual(search_results[0][2], "testing search")
        self.cursor.execute(
            "SELECT * FROM messages WHERE chat_id = ?",
            ("test456",)
        )
        chat_results = self.cursor.fetchall()
        self.assertEqual(len(chat_results), 1)
        self.assertEqual(chat_results[0][2], "different chat")

    def test_message_timestamps(self):
        messages = [
            ("test123", "user1", "first message"),
            ("test123", "user1", "second message")
        ]
        for msg in messages:
            self.cursor.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                (*msg, datetime.now().isoformat())
            )
        self.cursor.execute(
            "SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC",
            ("test123",)
        )
        results = self.cursor.fetchall()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][2], "first message")
        self.assertEqual(results[1][2], "second message")

    def test_delete_message(self):
        chat_id = "test123"
        sender = "user1"
        text = "message to delete"
        self.cursor.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (chat_id, sender, text, datetime.now().isoformat())
        )
        self.cursor.execute("SELECT * FROM messages WHERE chat_id = ?", (chat_id,))
        result = self.cursor.fetchone()
        self.assertIsNotNone(result)
        self.cursor.execute(
            "DELETE FROM messages WHERE chat_id = ? AND text = ?",
            (chat_id, text)
        )
        self.cursor.execute("SELECT * FROM messages WHERE chat_id = ?", (chat_id,))
        result = self.cursor.fetchone()
        self.assertIsNone(result)

    def test_batch_delete_messages(self):
        messages = [
            ("test123", "user1", "delete me 1"),
            ("test123", "user2", "keep me"),
            ("test123", "user1", "delete me 2"),
            ("test456", "user3", "keep me too")
        ]
        for msg in messages:
            self.cursor.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?)",
                (*msg, datetime.now().isoformat())
            )
        self.cursor.execute(
            "DELETE FROM messages WHERE text LIKE ?",
            ('%delete me%',)
        )
        self.cursor.execute("SELECT text FROM messages ORDER BY text")
        results = self.cursor.fetchall()
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "keep me")
        self.assertEqual(results[1][0], "keep me too")

    def test_notion_connection(self):
        # Use Notion credentials from environment
        if not notion_token or not database_id:
            self.skipTest("Notion credentials not found in environment")
        headers = {
            "Authorization": f"Bearer {notion_token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28"
        }
        try:
            response = requests.get(
                f"https://api.notion.com/v1/databases/{database_id}",
                headers=headers
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIsNotNone(data)
            self.assertIn('object', data)
            self.assertEqual(data['object'], 'database')
        except requests.exceptions.RequestException as e:
            self.fail(f"API request failed: {str(e)}")

    def tearDown(self):
        self.conn.close()

if __name__ == '__main__':
    unittest.main()
