from dotenv import load_dotenv
import os
import re
import google.generativeai as genai
from colorama import Fore, Back, Style

class GeminiAPI:
    def __init__(self):
        load_dotenv()
        self.check_key()
        genai.configure(api_key=os.getenv("API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.chat = self.model.start_chat(
            history=[
                {"role": "user", "parts": "You are an AI assistant for a habit tracking app. Provide helpful insights about user's habits, suggest improvements, and analyze patterns in a friendly, encouraging tone."},
                {"role": "model", "parts": "I'll provide helpful insights about your habits in a friendly and encouraging way. I'll analyze your patterns and suggest practical improvements to help you succeed."},
            ]
        )

    def check_key(self):
        try:
            with open(".env", "r") as k:
                keys = k.read().strip()
        except FileNotFoundError:
            print("\n.env file not found! Don't worry, just making one for you :)")
            self.setup_key()
        else:
            if self.is_key(keys):
                pass
            else:
                print("You have an invalid key!!! or You don't have a key in .env")
                self.setup_key()

    def is_key(self, key):
        return True if re.match("^API_KEY=[a-zA-Z0-9]{39}$", key) else False

    def setup_key(self):
        with open(".env", "w") as k:
            print(Fore.BLACK + Back.GREEN + "\nGet an API key From here: https://ai.google.dev/" + Style.RESET_ALL)
            key = input(Fore.GREEN + "\nPaste your Gemini API key here: " + Style.RESET_ALL)
            while True:
                if not self.is_key(f"API_KEY={key}"):
                    print(Fore.RED + "\nError: Invalid key, try again." + Style.RESET_ALL)
                    key = input(Fore.GREEN + "\nPaste your Gemini API key here: " + Style.RESET_ALL)
                else:
                    break
            k.write(f"API_KEY={key}")

    def generate_insight(self, habit_data):
        """Generate insights about habits using Gemini API"""
        prompt = f"Based on this habit data: {habit_data}, provide a brief, helpful insight about the user's habits. Focus on patterns, suggest one improvement, and be encouraging. Keep it under 100 words."
        response = self.chat.send_message(prompt)
        return response.text