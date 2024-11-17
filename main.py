from vectorstore import VectorStore
from eval import evaluate_rag_system
from dataclasses import dataclass
from typing import Dict
from kazllm import KazLLM

@dataclass
class RAGModel:
    vector_store: VectorStore
    llm: KazLLM
    
    def predict(self, question: str) -> Dict:
        """Predict answer using RAG approach"""
        context = " ".join(self.vector_store.query(question))
        answer = self.llm.generate_response(context, question)
        return {
            'answer': answer,
            'context': context
        }
    
    

def main():
    try:
        # Initialize components
        vector_store = VectorStore()
        
        # Process and store the transcription
        transcription = """
        Жасанды интеллект (ЖИ) бүгінгі күннің ең маңызды технологиялық жетістіктерінің бірі болып табылады. 
        Ол біздің күнделікті өмірімізде, жұмыста, медицинада, білім беру саласында, және өнеркәсіпте кеңінен қолданылуда. 
        Үлкен тілдік модельдер (ҮТМ), атап айтқанда, ЖИ-дің айқын дамуын көрсететін салалардың бірі болып табылады. 
        Бұл модельдер мәтіндерді түсіну, өңдеу, және тіпті жаңа ақпаратты құру мүмкіндіктерін кеңейтті. 
        Қазақстан үшін де ҮТМ маңызды рөл атқарады, себебі қазақ тілінің дамуына, цифрлық кеңістікте орын алуына және жаңа мүмкіндіктер ашуға үлес қосады.
        """
        vector_store.process_and_store(transcription)
        
        # Create RAG model
        rag_model = RAGModel(vector_store=vector_store, llm=KazLLM())
        
        # Questions for evaluation
        eval_questions = [
            "Жасанды интеллект қандай салаларда қолданылады?",
            "ҮТМ дегеніміз не?",
            "ЖИ Қазақстан үшін қандай маңызы бар?",
            "Жасанды интеллект қандай мүмкіндіктер береді?"
        ]
        
        # Run comparative evaluation
        print("=== Running Comparative Evaluation ===")
        results, report = evaluate_rag_system(rag_model, eval_questions)
        
        # Print evaluation report
        print(report)
        
        return results

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    main()