from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Prepare data
data = {
    "question": ["Who wrote Harry Potter?"],
    "answer": ["J.K. Rowling wrote Harry Potter."],
    "contexts": [["J.K. Rowling is the author of Harry Potter series."]]
}
dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(result)