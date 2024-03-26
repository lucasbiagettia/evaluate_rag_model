from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
import os
import sys

main_proyect = '..'
sys.path.append(main_proyect)

from groq_model import Chatbot

api_key = os.getenv("GROQ_API_KEY")
model = Chatbot(api_key) 


# Replace this with the actual documents that you are passing as input to your LLM.
context=["A man with blond-hair, and a brown shirt drinking out of a public water fountain."]

# Replace this with the actual output from your LLM application
actual_output="A blond drinking water in public."

test_case = LLMTestCase(
    input="What was the blond doing?",
    actual_output=actual_output,
    context=context
)
metric = HallucinationMetric(threshold=0.5, model=model)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])