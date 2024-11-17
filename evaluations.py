from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EvaluationEntry:
    question: str
    expected_answer: str

class EvaluationDataset:
    def __init__(self):
        """Initialize the dataset with predefined questions and answers."""
        self.dataset: List[EvaluationEntry] = []

    def add_entry(self, question: str, expected_answer: str):
        """Add a question and expected answer to the dataset."""
        self.dataset.append(EvaluationEntry(question, expected_answer))

    def get_all_entries(self) -> List[EvaluationEntry]:
        """Retrieve all dataset entries."""
        return self.dataset
    
    def load_sample_data(self):
        """Load sample questions and answers for evaluation."""
        self.add_entry(
            question="What is artificial intelligence?",
            expected_answer="Artificial intelligence refers to the simulation of human intelligence in machines."
        )
        self.add_entry(
            question="What is the significance of AI in healthcare?",
            expected_answer="AI helps in diagnosing diseases, predicting patient outcomes, and improving treatments."
        )
        self.add_entry(
            question="What is the role of large language models?",
            expected_answer="Large language models process and generate human-like text for various applications."
        )
