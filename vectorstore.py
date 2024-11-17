from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from config import Config
import os

class VectorStore:
    def __init__(self, index_name="kazllm"): 
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        self.index = self.pc.Index(index_name)
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=80,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def process_and_store(self, text):
        chunks = self.text_splitter.split_text(text)
        
        embeddings = self.embeddings.embed_documents(chunks)
        
        # Prepare vectors
        vectors = [
            {
                "id": f"chunk_{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
        return len(vectors)

    def query(self, question, top_k=3):
        # Create embedding for the question
        question_embedding = self.embeddings.embed_query(question)
        
        # Query Pinecone
        results = self.index.query(
            vector=question_embedding,
            top_k=top_k,
            include_values=True,
            include_metadata=True
        )
        
        return [match.metadata["text"] for match in results.matches]
