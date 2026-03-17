import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
API_KEY = os.getenv('API_KEY')

if API_KEY is None:
    raise ValueError("API_KEY not found in environment variables. Please set it in the .env file.")

# You can add other configuration here
# For example:
# MODEL_PATH = os.getenv('MODEL_PATH', 'default/path')
