import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Slack
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET", "")

# LLM
MODEL_NAME = os.environ.get("MODEL_NAME", "openai:gpt-4o")

# Database
DATABASE_PATH = os.environ.get(
    "DATABASE_PATH",
    str(Path(__file__).parent.parent / "data" / "synthetic_startup.sqlite"),
)