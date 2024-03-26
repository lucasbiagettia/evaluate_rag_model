from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from deepeval.models import DeepEvalBaseLLM



class Chatbot(DeepEvalBaseLLM):

    def __init__(self, api_key):
        self.initialize(api_key)

    def initialize(self, api_key):
        chat = ChatGroq(groq_api_key=api_key, model_name= "mixtral-8x7b-32768", temperature=0.2, verbose=True)
        system = "Responderas a una api que testea otro modelo. Siempre debes responder en español"
        human = "{human}"
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

        self.chain = LLMChain(llm=chat, prompt=prompt) 

    def load_model(self):
        return self
   
    def generate(self, prompt: str) -> str:
        ret  = self.chain.invoke({"human": prompt})['text']
        return ret
    
    def get_model_name(self):
        return "Custom model"
    async def a_generate(self, prompt: str) -> str:
        ret  = self.chain.invoke({"human": prompt})
        return ret['text']




