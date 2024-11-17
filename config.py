import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    KAZLLM_API_KEY = "v7m59Y7H.r7BN7286uwx7br05Ok9yMqC02vuSWQPt"
    KAZLLM_BASE_URL = "https://apikazllm.nu.edu.kz"
    KAZLLM_HEADERS = {
        "Authorization": f"Api-Key {KAZLLM_API_KEY}",
        "accept": "application/json",
        "Content-Type": "application/json"
    }

