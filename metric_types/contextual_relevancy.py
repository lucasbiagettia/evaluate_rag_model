import os
import sys
from deepeval import evaluate
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

main_proyect = '..'
sys.path.append(main_proyect)

from groq_model import Chatbot

api_key = os.getenv("GROQ_API_KEY")
model = Chatbot(api_key) 

# Replace this with the actual output from your LLM application
actual_output = "La función Ahora() devuelve tanto la fecha como la hora actual, mientras que la función Hoy() devuelve solo la fecha actual sin la hora."

# Replace this with the actual retrieved context from your RAG pipeline
retrieval_context = ['''Fecha hora 

Ahora()	
Devuelve la fecha y hora actuales.	
Ejemplos:
Ahora() retorna 16/08/2023 10:46

Hoy()	
Devuelve la fecha actual (sin hora).	
Ejemplos:
Hoy() retorna 16/08/2023"
''']

metric = ContextualRelevancyMetric(
    threshold=0.5,
    model=model,
    include_reason=True
)
test_case = LLMTestCase(
    input="Comparar las funciones Ahora y Hoy",
    actual_output=actual_output,
    retrieval_context=retrieval_context
)

metric.measure(test_case)
print(metric.score)
print(metric.reason)

# or evaluate test cases in bulk
evaluate([test_case], [metric])