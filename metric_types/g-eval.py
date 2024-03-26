import os
import sys
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

main_proyect = '..'
sys.path.append(main_proyect)

from groq_model import Chatbot

api_key = os.getenv("GROQ_API_KEY")
model = Chatbot(api_key) 



coherence_metric = GEval(
    name="Coherence",
    criteria="Coherencia- determine si la salida real es coherente con la entrada.",
    model= model,
    evaluation_steps=["Compruebe si las oraciones en 'input' se alinean con las de 'actual output'."],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)


from deepeval.test_case import LLMTestCase

def test_coherence(input, actual_output):
    test_case = LLMTestCase(
        input=  input,
        actual_output=actual_output)
    coherence_metric.measure(test_case)
    print(coherence_metric.score)
    print(coherence_metric.reason)


preguntas = [
    "Cuanto pesa una ballena",
    "Donde queda Paris",
    "Que produce cáncer",
    "Cuantos dientes tiene un humano"
]
respuestas = [
    "El peso de una ballena puede variar dependiendo de la especie y el tamaño. Por ejemplo, una ballena azul puede pesar hasta 200 toneladas.",
    "París es la capital de Francia y se encuentra en el norte de Francia, en la región de Île-de-France.",
    "Existen múltiples factores que pueden incrementar el riesgo de cáncer, incluyendo el tabaquismo, la exposición a radiaciones ultravioleta, la dieta poco saludable, la falta de actividad física, entre otros.",
    "Un adulto humano típicamente tiene 32 dientes, incluyendo cuatro molares, cuatro premolares, ocho incisivos y doce molares."
]
for i in range(4):
    test_coherence(preguntas[i], respuestas[i])
    print("pregunta: ", i)