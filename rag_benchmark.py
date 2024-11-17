# from vectorstore import VectorStore
# from kazllm import KazLLM
# from sklearn.metrics.pairwise import cosine_similarity
# from rouge_score import rouge_scorer
# import numpy as np
# from transformers import AutoTokenizer, AutoModel
# import torch
# import time

# class RAGBenchmark:
#     def __init__(self):
#         self.vector_store = VectorStore()
#         self.llm = KazLLM()
#         self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
#         # Initialize BERT model for semantic similarity
#         self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
#         self.model = AutoModel.from_pretrained("bert-base-multilingual-cased")
        
#     def get_bert_embedding(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
#     def calculate_metrics(self, generated_answer, reference_answer):
#         # Calculate ROUGE scores
#         rouge_scores = self.scorer.score(generated_answer, reference_answer)
        
#         # Calculate semantic similarity using BERT
#         gen_embedding = self.get_bert_embedding(generated_answer)
#         ref_embedding = self.get_bert_embedding(reference_answer)
#         semantic_sim = cosine_similarity([gen_embedding], [ref_embedding])[0][0]
        
#         return {
#             'rouge1': rouge_scores['rouge1'].fmeasure,
#             'rouge2': rouge_scores['rouge2'].fmeasure,
#             'rougeL': rouge_scores['rougeL'].fmeasure,
#             'semantic_similarity': semantic_sim
#         }
    
#     def run_benchmark(self, text, qa_pairs):
#         # Store the text
#         print("Processing and storing text...")
#         self.vector_store.process_and_store(text)
#         time.sleep(2)  # Wait for indexing
        
#         results = []
        
#         for i, (question, reference_answer) in enumerate(qa_pairs, 1):
#             print(f"\nProcessing Q&A pair {i}...")
            
#             # Get relevant chunks
#             relevant_chunks = self.vector_store.query(question)
#             context = " ".join(relevant_chunks)
            
#             # Generate answer
#             generated_answer = self.llm.generate_response(context, question)
            
#             # Calculate metrics
#             metrics = self.calculate_metrics(generated_answer, reference_answer)
            
#             result = {
#                 'question': question,
#                 'reference_answer': reference_answer,
#                 'generated_answer': generated_answer,
#                 'metrics': metrics
#             }
#             results.append(result)
        
#         return results

# # Example usage
# def main():
#     # Sample text
#     text = """
#     Жасанды интеллект (ЖИ) бүгінгі күннің ең маңызды технологиялық жетістіктерінің бірі болып табылады. Ол біздің күнделікті өмірімізде, жұмыста, медицинада, білім беру саласында, және өнеркәсіпте кеңінен қолданылуда. Үлкен тілдік модельдер (ҮТМ), атап айтқанда, ЖИ-дің айқын дамуын көрсететін салалардың бірі болып табылады. Бұл модельдер мәтіндерді түсіну, өңдеу, және тіпті жаңа ақпаратты құру мүмкіндіктерін кеңейтті. Қазақстан үшін де ҮТМ маңызды рөл атқарады, себебі қазақ тілінің дамуына, цифрлық кеңістікте орын алуына және жаңа мүмкіндіктер ашуға үлес қосады.
#     """
    
#     # Predefined question-answer pairs
#     qa_pairs = [
#         (
#             "Жасанды интеллект дегеніміз не?",
#             "Жасанды интеллект - адам интеллектісінің функцияларын орындайтын және адамның ойлау процестерін симуляциялайтын компьютерлік жүйелер мен бағдарламалар."
#         ),
#         (
#             "Үлкен тілдік модельдер қандай тапсырмаларды орындай алады?",
#             "Үлкен тілдік модельдер мәтіндерді талдайды, жауап береді, мәтіндер құрастырады және күрделі тапсырмаларды орындай алады."
#         ),
#         (
#             "ҮТМ-нің негізгі артықшылықтары қандай?",
#             "Тілдерді түсіну, креативтілік, тиімділік және деректерді талдау ҮТМ-нің негізгі артықшылықтары болып табылады."
#         ),
#         (
#             "Қазақстан үшін ҮТМ-нің маңызы қандай?",
#             "Қазақстан үшін ҮТМ қазақ тілінің цифрлық дәуірде бәсекеге қабілетті болуына көмектеседі."
#         )
#     ]
    
#     benchmark = RAGBenchmark()
#     results = benchmark.run_benchmark(text, qa_pairs)
    
#     # Print results
#     print("\nBenchmark Results:")
#     print("=" * 80)
    
#     average_metrics = {
#         'rouge1': [], 'rouge2': [], 'rougeL': [], 'semantic_similarity': []
#     }
    
#     for i, result in enumerate(results, 1):
#         print(f"\nQuestion {i}: {result['question']}")
#         print(f"Reference Answer: {result['reference_answer']}")
#         print(f"Generated Answer: {result['generated_answer']}")
#         print("\nMetrics:")
#         for metric, value in result['metrics'].items():
#             print(f"{metric}: {value:.4f}")
#             average_metrics[metric].append(value)
#         print("-" * 80)
    
#     print("\nOverall Metrics:")
#     for metric, values in average_metrics.items():
#         avg_value = np.mean(values)
#         print(f"Average {metric}: {avg_value:.4f}")

# if __name__ == "__main__":
#     main()
