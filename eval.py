import os
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


from groq_model import Chatbot

api_key = os.getenv("GROQ_API_KEY")
model = Chatbot(api_key) 



coherence_metric = GEval(
    name="Coherence",
    criteria="Coherencia- determine si la salida real es coherente con la entrada.",
    model= model,
    evaluation_steps=["Compruebe si las oraciones en 'input' se alinean con las de 'actual output'."],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    async_mode= False
)


from deepeval.test_case import LLMTestCase
...

test_case = LLMTestCase(
    input="cuantas casas hay en valencia?",
    actual_output='''

Lo siento, pero no dispongo de datos en tiempo real ni actualizados sobre el número exacto de casas en Valencia, España. La cantidad de casas en una ciudad puede variar debido a diversos factores, como el crecimiento demográfico, el desarrollo urbano, la construcción de nuevas viviendas y la demolición de las existentes. Te recomendaría buscar en fuentes locales, como el ayuntamiento de Valencia o instituciones de estadísticas, para obtener información más precisa y actualizada sobre este tema.
''')
coherence_metric.measure(test_case)
print(coherence_metric.score)
print(coherence_metric.reason)