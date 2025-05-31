from dotenv import load_dotenv,dotenv_values
import os

from pathlib import Path


if os.path.exists("gv.env"):
    load_dotenv('./gv.env')
else:
    documents_path = Path.home() / ".env"
    load_dotenv(os.path.join(documents_path, 'gv.env'))

LANGCHAIN_API_KEY = os.environ.get('LANGCHAIN_API_KEY')
#LANGCHAIN_API_KEY_DEV = os.environ.get('LANGCHAIN_API_KEY_DEV')
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
