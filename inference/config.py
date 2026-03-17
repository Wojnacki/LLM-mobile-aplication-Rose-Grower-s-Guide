import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the Anthropic API key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

if ANTHROPIC_API_KEY is None:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables. Please set it in the .env file.")

# You can add other configuration here
# For example:
# MODEL_PATH = os.getenv('MODEL_PATH', 'default/path')
