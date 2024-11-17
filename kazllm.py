from weave import Model
import weave
from typing import Any, Optional
from dataclasses import dataclass
import requests

class KazLLM:
    def __init__(self):
        self.base_url = "https://apikazllm.nu.edu.kz"
        self.headers = {
            "Authorization": "Api-Key v7m59Y7H.r7BN7286uwx7br05Ok9yMqC02vuSWQPt",
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        self.assistant_id = self.get_or_create_assistant()

    def get_or_create_assistant(self):
        """Retrieve a KazLLM assistant or create one if not found."""
        try:
            response = requests.get(
                f"{self.base_url}/assistant/",
                headers=self.headers
            )
            if response.status_code != 200:
                raise Exception(f"Failed to retrieve assistants: {response.text}")
            
            assistants = response.json()
            for assistant in assistants:
                if assistant.get("model") == "KazLLM":
                    print(f"Found existing KazLLM assistant: {assistant['id']}")
                    return assistant["id"]

            create_payload = {
                "name": "KazLLM RAG",
                "description": "Assistant for answering questions with context",
                "temperature": 0.7,
                "max_tokens": 200,
                "model": "KazLLM",
                "system_instructions": "Answer questions based on the provided context. Answer only using the context information.",
                "context": ""
            }
            
            response = requests.post(
                f"{self.base_url}/assistant/",
                json=create_payload,
                headers=self.headers
            )
            if response.status_code != 201:
                raise Exception(f"Failed to create assistant: {response.text}")
            
            assistant_id = response.json()["id"]
            print(f"Created new KazLLM assistant: {assistant_id}")
            return assistant_id
        except Exception as e:
            print(f"Error in get_or_create_assistant: {str(e)}")
            raise

    def generate_response(self, context, question):
        """Generate response using KazLLM with context."""
        try:
            instruction = "Answer only based on the provided context. Do not include Жауап, ответ, answer. Just provide the answer that is it."
            payload = {
                "text_prompt": f"{instruction}\n\nQuestion: {question}",
                "context": context,
                "file_prompt": None
            }
            
            print(f"\nSending request to KazLLM:")
            print(f"Assistant ID: {self.assistant_id}")
            print(f"Question: {question}")
            print(f"Context length: {len(context)} characters")
            
            response = requests.post(
                f"{self.base_url}/assistant/{self.assistant_id}/interactions/",
                json=payload,
                headers=self.headers
            )
            
            print(f"Response status code: {response.status_code}")
            if response.status_code != 201:
                raise Exception(f"Failed to generate response: {response.text}")
            
            response_data = response.json()
            if "vllm_response" in response_data and "content" in response_data["vllm_response"]:
                return response_data["vllm_response"]["content"]
            else:
                raise Exception(f"Unexpected response format: {response_data}")
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            raise

@dataclass
class KazRAGModel(Model):
    vector_store: Any  # Type hint for vector store
    llm: KazLLM
    system_message: str = "Сіз қазақ тілінде сұрақтарға жауап беретін көмекшісіз."
    
    def __init__(self, vector_store, llm: KazLLM, system_message: Optional[str] = None):
        self.vector_store = vector_store
        self.llm = llm
        if system_message:
            self.system_message = system_message
        super().__init__()

    @weave.op()
    def predict(self, question: str) -> dict:
        """
        Predict answer using RAG approach with KazLLM
        
        Args:
            question: Question in Kazakh
            
        Returns:
            dict: Contains answer and context used for generation
        """
        try:
            # Get relevant context from vector store
            relevant_chunks = self.vector_store.query(question)
            context = " ".join(relevant_chunks)
            
            # Generate response using KazLLM
            answer = self.llm.generate_response(context, question)
            
            return {
                'answer': answer,
                'context': context
            }
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'context': ""
            }

# Example usage
def create_rag_model(vector_store):
    """Helper function to create a RAG model instance"""
    llm = KazLLM()
    model = KazRAGModel(
        vector_store=vector_store,
        llm=llm,
        system_message="Сіз қазақ тілінде сұрақтарға жауап беретін көмекшісіз."
    )
    return model