from typing import List, Dict
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class ComparisonMetrics:
    relevance_score: float
    correctness_score: float
    fluency_score: float
    overall_score: float
    feedback: str

class LLMComparator:
    def __init__(self):
        """Initialize both models"""
        self.client = OpenAI()
        self.comparison_prompt = """
        As an expert evaluator, compare these two answers to the given question in Kazakh language.
        
        Question: {question}
        Context provided: {context}
        
        KazLLM Answer: {kazllm_answer}
        GPT4-mini Answer: {gpt4_answer}
        
        Please evaluate KazLLM's answer compared to GPT4-mini's answer on a scale of 1-10 for:
        1. Relevance: How relevant is the answer to the question and context?
        2. Correctness: How factually correct is the answer?
        3. Fluency: How natural and fluent is the Kazakh language used?
        
        Provide scores in this format:
        Relevance Score: [score]
        Correctness Score: [score]
        Fluency Score: [score]
        Overall Score: [average]
        
        """

    def get_gpt4_response(self, question: str, context: str) -> str:
        """Get response from GPT4-mini"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use appropriate GPT4-mini model
                messages=[
                    {"role": "system", "content": "You are a Kazakh language expert. Answer questions based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting GPT4 response: {str(e)}")
            return None

    def compare_responses(self, question: str, context: str, 
                         kazllm_answer: str, gpt4_answer: str) -> ComparisonMetrics:
        """Compare KazLLM and GPT4-mini responses"""
        try:
            prompt = self.comparison_prompt.format(
                question=question,
                context=context,
                kazllm_answer=kazllm_answer,
                gpt4_answer=gpt4_answer
            )
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using gpt-4o-mini as judge
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of Kazakh language AI systems."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            evaluation = response.choices[0].message.content
            
            # Parse scores
            scores = {}
            for line in evaluation.split('\n'):
                if 'Score:' in line:
                    metric, score = line.split(':')
                    scores[metric.strip()] = float(score.strip())
            
            # Extract feedback
            feedback = evaluation.split('Brief Feedback:')[-1].strip()
            
            return ComparisonMetrics(
                relevance_score=scores.get('Relevance Score', 0),
                correctness_score=scores.get('Correctness Score', 0),
                fluency_score=scores.get('Fluency Score', 0),
                overall_score=scores.get('Overall Score', 0),
                feedback=feedback
            )
            
        except Exception as e:
            print(f"Error in comparison: {str(e)}")
            return None

class RAGEvaluator:
    def __init__(self, rag_model):
        self.rag_model = rag_model
        self.comparator = LLMComparator()
        
    def evaluate_system(self, questions: List[str]) -> Dict:
        """Evaluate RAG system against GPT4-mini"""
        results = []
        metrics_summary = {
            'relevance': [],
            'correctness': [],
            'fluency': [],
            'overall': []
        }
        
        for question in tqdm(questions, desc="Comparing responses"):
            # Get RAG response
            rag_response = self.rag_model.predict(question)
            context = rag_response['context']
            kazllm_answer = rag_response['answer']
            
            # Get GPT4-mini response
            gpt4_answer = self.comparator.get_gpt4_response(question, context)
            
            if gpt4_answer:
                # Compare responses
                metrics = self.comparator.compare_responses(
                    question=question,
                    context=context,
                    kazllm_answer=kazllm_answer,
                    gpt4_answer=gpt4_answer
                )
                
                if metrics:
                    results.append({
                        'question': question,
                        'context': context,
                        'kazllm_answer': kazllm_answer,
                        'gpt4_answer': gpt4_answer,
                        'metrics': metrics
                    })
                    
                    # Collect metrics for averaging
                    metrics_summary['relevance'].append(metrics.relevance_score)
                    metrics_summary['correctness'].append(metrics.correctness_score)
                    metrics_summary['fluency'].append(metrics.fluency_score)
                    metrics_summary['overall'].append(metrics.overall_score)
        
        # Calculate averages
        average_metrics = {
            metric: np.mean(scores) for metric, scores in metrics_summary.items()
        }
        
        return {
            'detailed_results': results,
            'average_metrics': average_metrics
        }
def evaluate_rag_system(rag_model, eval_questions: List[str]) -> Dict:
    """Main evaluation function to calculate and return metrics as key-value pairs."""
    evaluator = RAGEvaluator(rag_model)
    results = evaluator.evaluate_system(eval_questions)

    # Extract average metrics
    average_metrics = results['average_metrics']

    # Prepare response as a dictionary with metric names and scores
    metrics_response = {
        "Relevance": f"{np.mean(average_metrics['relevance']):.2f}/10",
        "Correctness": f"{np.mean(average_metrics['correctness']):.2f}/10",
        "Fluency": f"{np.mean(average_metrics['fluency']):.2f}/10",
        "Overall": f"{np.mean(average_metrics['overall']):.2f}/10",
    }

    return metrics_response
