from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from main import RAGModel
from vectorstore import VectorStore
from kazllm import KazLLM
from eval import LLMComparator
from fastapi.middleware.cors import CORSMiddleware
import uvicorn  


app = FastAPI()



origins = [
    "*",  

]

# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allowed origins
    allow_credentials=True,  # Allow cookies to be sent
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
try:
    vector_store = VectorStore()
    kaz_llm = KazLLM()
    rag_model = RAGModel(vector_store=vector_store, llm=kaz_llm)
    comparator = LLMComparator()
except Exception as e:
    raise RuntimeError(f"Error initializing core components: {e}")


class ProcessTextRequest(BaseModel):
    text: str  
@staticmethod
@staticmethod
def parse_scores(evaluation: str) -> Dict:
    """Extract scores from the LLM's evaluation response."""
    import re
    scores = {"relevance": 0.0, "correctness": 0.0, "fluency": 0.0}
    for line in evaluation.split("\n"):
        if "Relevance:" in line:
            try:
                scores["relevance"] = float(line.split(":")[1].strip())
            except ValueError:
                scores["relevance"] = 0.0  # Default to 0.0 if conversion fails
        elif "Correctness:" in line:
            try:
                scores["correctness"] = float(line.split(":")[1].strip())
            except ValueError:
                scores["correctness"] = 0.0
        elif "Fluency:" in line:
            try:
                scores["fluency"] = float(line.split(":")[1].strip())
            except ValueError:
                scores["fluency"] = 0.0
    return scores


class QuestionAnswerPair(BaseModel):
    question: str
    expected_answer: str


class EvaluationRequestWithAnswers(BaseModel):
    question_answer_pairs: List[QuestionAnswerPair]


class QuestionRequest(BaseModel):
    question: str

class EvaluationRequest(BaseModel):
    question: str 


class EvaluationResponseEntry(BaseModel):
    question: str
    expected_answer: str
    actual_answer: str
    relevance_score: float
    correctness_score: float
    fluency_score: float


class FullEvaluationResponse(BaseModel):
    results: List[EvaluationResponseEntry]
    report: str


def evaluate_rag_system_with_predefined_answers(rag_model: RAGModel, questions: List[str], answers: List[str]) -> Dict:
    """
    Evaluate the RAG model against predefined answers and return metrics.
    """
    total_relevance, total_correctness, total_fluency = 0.0, 0.0, 0.0
    num_entries = len(questions)

    for question, expected_answer in zip(questions, answers):
        # Get RAG's predicted answer
        prediction = rag_model.predict(question)

        # Evaluate relevance, correctness, and fluency based on expected and actual answers
        scores = rag_model.evaluate_with_llm(
            question=question,
            context=prediction["context"],
            expected_answer=expected_answer,
            actual_answer=prediction["answer"]
        )

        total_relevance += scores["relevance"]
        total_correctness += scores["correctness"]
        total_fluency += scores["fluency"]

    average_relevance = total_relevance / num_entries
    average_correctness = total_correctness / num_entries
    average_fluency = total_fluency / num_entries
    overall_score = (average_relevance + average_correctness + average_fluency) / 3

    return {
        "Relevance": f"{average_relevance:.2f}/10",
        "Correctness": f"{average_correctness:.2f}/10",
        "Fluency": f"{average_fluency:.2f}/10",
        "Overall": f"{overall_score:.2f}/10"
    }
@app.post("/evaluate_with_predefined")
async def evaluate_with_predefined(request: EvaluationRequestWithAnswers):
    """
    Endpoint to evaluate questions with predefined answers without storing the text in the vector store.
    """
    try:
        total_relevance, total_correctness, total_fluency = 0.0, 0.0, 0.0
        valid_entries = 0  # Count only successfully evaluated entries
        results = []  # Store individual results with answers

        for pair in request.question_answer_pairs:
            try:
                # Get KazLLM response
                prediction = rag_model.predict(pair.question)

                # Evaluate using gpt-4o-mini
                scores = comparator.compare_responses(
                    question=pair.question,
                    context=prediction["context"],
                    kazllm_answer=prediction["answer"],
                    gpt4_answer=pair.expected_answer
                )

                # Check if scores are valid
                if scores is not None:
                    total_relevance += scores.relevance_score
                    total_correctness += scores.correctness_score
                    total_fluency += scores.fluency_score
                    valid_entries += 1

                    # Append individual results
                    results.append({
                        "question": pair.question,
                        "expected_answer": pair.expected_answer,
                        "kazllm_answer": prediction["answer"],
                        "relevance_score": scores.relevance_score,
                        "correctness_score": scores.correctness_score,
                        "fluency_score": scores.fluency_score,
                    })

            except Exception as e:
                results.append({
                    "question": pair.question,
                    "expected_answer": pair.expected_answer,
                    "kazllm_answer": f"Error generating answer: {str(e)}",
                    "relevance_score": 0.0,
                    "correctness_score": 0.0,
                    "fluency_score": 0.0,
                })

        # If no valid entries were evaluated, return an error
        if valid_entries == 0:
            raise HTTPException(status_code=500, detail="No valid evaluations completed.")

        # Calculate average metrics
        average_relevance = total_relevance / valid_entries
        average_correctness = total_correctness / valid_entries
        average_fluency = total_fluency / valid_entries
        overall_score = (average_relevance + average_correctness + average_fluency) / 3

        # Return the metrics and individual results
        return {
            "results": results,
            "metrics": {
                "Relevance": f"{average_relevance:.2f}/10",
                "Correctness": f"{average_correctness:.2f}/10",
                "Fluency": f"{average_fluency:.2f}/10",
                "Overall": f"{overall_score:.2f}/10",
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating: {str(e)}")

    
@app.post("/store_text")
async def store_text(request: ProcessTextRequest):
    """
    Endpoint to process and save text into the VectorStore.
    """
    try:
        # Process and store the text in the vector store
        num_chunks = vector_store.process_and_store(request.text)

        # Return success message
        return {"message": f"Text successfully processed and stored.", "chunks_stored": num_chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error storing text: {str(e)}")     
    
@app.post("/evaluate")
async def evaluate_text(request: EvaluationRequest):
    """
    Endpoint to return an answer along with context for a given question without storing the text.
    """
    try:
        try:
            # Predict the answer and get the context
            prediction = rag_model.predict(request.question)

            # Clean and format the context
            formatted_context = " ".join(prediction["context"].split())

            # Return the result
            return {
                "question": request.question,
                "answer": prediction["answer"],
                "context": formatted_context
            }

        except Exception as e:
            # Handle prediction errors
            return {
                "question": request.question,
                "answer": f"Error generating answer: {str(e)}",
                "context": ""
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating: {str(e)}")
@app.get("/")
async def root():
    """
    Root endpoint to check API status.
    """
    return {"message": "RAG System API is running."}


# Add this block to start the server when app.py is run directly
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
