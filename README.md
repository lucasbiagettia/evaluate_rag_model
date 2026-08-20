# Evaluar RAG Model

Este repositorio es un prototipo pequeño y práctico para evaluar respuestas de un modelo de lenguaje en un flujo tipo RAG (retrieval-augmented generation).

La idea principal es comparar la salida generada por un LLM con contexto relevante y medir qué tan buena es la respuesta en términos de:

- veracidad / ausencia de alucinación
- relevancia con respecto a la pregunta
- coherencia general
- solapamiento léxico con una referencia esperada

---

## ¿De qué se trata?

El proyecto combina:

- LangChain + Groq para crear un modelo personalizado
- DeepEval para métricas de evaluación de LLMs
- Hugging Face `evaluate` para métricas como ROUGE y BLEU
- scripts de ejemplo para probar distintos tipos de métricas

Es útil para hacer experimentos de evaluación de modelos antes de poner un sistema en producción.

---

## ¿Tiene interés?

Sí, tiene interés sobre todo como:

- laboratorio de prueba para evaluar RAGs
- base para comparar respuestas de distintos modelos
- herramienta de investigación para medir calidad en prompts y outputs
- punto de partida para construir un pipeline más robusto de evaluación automática

No es un proyecto terminado ni una librería generalista; es más bien un repositorio experimental / educativo con ejemplos concretos.

---

## Estructura del repositorio

```text
.
├── README.md
├── main.py                  # archivo de entrada; actualmente vacío
├── groq_model.py            # wrapper del modelo basado en Groq + LangChain
├── hallucination.py         # ejemplo de evaluación de alucinación + ROUGE
├── requirements.txt         # dependencias mínimas declaradas
├── metric_types/
│   ├── answer_relevancy.py
│   ├── contextual_relevancy.py
│   ├── g-eval.py
│   └── rouge_and_bleu.py
└── .gitignore
```

---

## Componentes principales

### 1) `groq_model.py`

Define una clase `Chatbot` que hereda de `DeepEvalBaseLLM` y encapsula un modelo de Groq con LangChain.

Ejemplo de comportamiento:

- recibe una API key de Groq
- crea un `ChatGroq`
- arma un prompt con sistema + usuario
- devuelve la respuesta del modelo

### 2) `hallucination.py`

Tiene un evaluador `ChatbotEvaluator` que:

- arma un `LLMTestCase`
- calcula una métrica de alucinación con `HallucinationMetric`
- calcula ROUGE contra un contexto de referencia
- devuelve un score y una tabla con métricas

### 3) `metric_types/`

Cada archivo muestra una métrica distinta:

- `answer_relevancy.py`: relevancia de la respuesta
- `contextual_relevancy.py`: relevancia con contexto recuperado
- `g-eval.py`: evaluación con `GEval`, basada en criterios de calidad
- `rouge_and_bleu.py`: métricas lexicográficas

---

## Dependencias

El archivo `requirements.txt` declara:

```txt
langchain_core
langchain_groq
langchain
deepeval
```

Sin embargo, para que los scripts funcionen correctamente también suelen necesitarse paquetes como:

```bash
pip install evaluate pandas
```

En especial, `evaluate` se usa para ROUGE y BLEU y `pandas` se usa para formatear resultados en DataFrames.

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install evaluate pandas
```

Luego configurar la clave de Groq:

```bash
export GROQ_API_KEY="tu_api_key_aqui"
```

---

## Ejecución

Ejemplo simple:

```bash
python hallucination.py
```

O cualquier script dentro de `metric_types/`:

```bash
python metric_types/answer_relevancy.py
```

---

## Ejemplo de uso conceptual

El flujo típico en este repo es:

1. tener una pregunta
2. tener una respuesta generada por el modelo
3. tener contexto o referencia
4. ejecutar una métrica
5. leer el score y, si aplica, la razón de la métrica

Ejemplo de idea:

```python
from hallucination import ChatbotEvaluator

score = evaluator.evaluate(
    input_text="¿Quién fue Astor Piazzolla?",
    actual_output="...respuesta del modelo...",
    context=["...contexto de referencia..."]
)
print(score)
```

---

## Limitaciones

Este proyecto tiene varios puntos de prototipo:

- `main.py` está vacío
- la configuración es muy manual
- no hay CLI ni interfaz de usuario
- no hay tests automatizados
- algunas dependencias podrían estar faltando en `requirements.txt`
- el enfoque está más orientado a pruebas de investigación que a productización

---

## Casos de uso recomendados

- comparar respuestas antes/después de un ajuste de prompt
- medir si un RAG está generando contenido no soportado por la recuperación
- evaluar calidad de respuesta en un conjunto de preguntas reales
- crear una base para un benchmarking más avanzado

---

## Conclusión

Es un repositorio útil como punto de partida para evaluar modelos LLM/RAG con métricas prácticas y realistas. Tiene valor principalmente en entornos de investigación, pruebas y prototipos, no como una solución final lista para producción.
