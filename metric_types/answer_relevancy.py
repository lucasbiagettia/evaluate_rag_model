import os
import sys
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
# Agregar la ruta deseada al sys.path
main_proyect = '..'
sys.path.append(main_proyect)

# Ahora puedes importar módulos desde esta ruta
from groq_model import Chatbot
# Replace this with the actual output from your LLM application
api_key = os.getenv("GROQ_API_KEY")
model = Chatbot(api_key) 

actual_output = "We offer a 30-day full refund at no extra cost."

metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=model,
    include_reason=True,
   #_mode= False
)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=actual_output
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])