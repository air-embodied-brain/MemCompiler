import os
from dotenv import load_dotenv
load_dotenv()

# Only set if not None (for Azure compatibility)
if os.getenv("OPENAI_API_BASE") is not None:
    os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
if os.getenv("OPENAI_API_KEY") is not None:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")