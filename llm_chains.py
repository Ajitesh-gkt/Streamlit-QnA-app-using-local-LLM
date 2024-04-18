from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.vectorstores import Chroma
from prompt_templates import memory_prompt_template
import chromadb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Below are functions for creating the LLM chain, embeddings, chat memory, prompt and LLM chain
def create_llm(model_path= config["model_path"]["large"],model_type = "mistral", model_config = config["model_config"]):
    llm = CTransformers(model = model_path, model_type= model_type, config = model_config)
    return llm

def create_embeddings(embeddings_path = config["embeddings_path"] ):
    return HuggingFaceBgeEmbeddings(model_name = embeddings_path)
    
def create_chat_memory(chat_history):
    return ConversationBufferMemory(memory_key ="history", chat_memory=chat_history, k=3)

def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)
    
def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm = llm, prompt =  chat_prompt, memory = memory)

def load_normal_chain(chat_history):
    return chatChain(chat_history)

# vectordb stores the embeddings of the documents (the tokens and text fetched from the documents)
def load_vectordb(embeddings):
    persistent_client = chromadb.PersistentClient("chromoa_db")

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name="collection_name",
        embedding_function=embeddings,
)
    return langchain_chroma

def load_pdf_chat_chain(chat_history):
    return pdfChatChain(chat_history)

def load_retrieval_chain(llm, memory, vector_db):
    return RetrievalQA.from_llm(llm = llm, memory = memory, retriever = vector_db.as_retriever())


# pdf chat chain
class pdfChatChain:

    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        self.vector_db = load_vectordb(create_embeddings())
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = load_retrieval_chain(llm, self.memory, self.vector_db)

    def run(self, user_input):
        return self.llm_chain.run(query = user_input, history=self.memory.chat_memory.messages, stop="Human:")


# normal LLM chat chain
class chatChain:

    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)
        llm = create_llm()
        chat_prompt = create_prompt_from_template(memory_prompt_template)
        self.llm_chain = create_llm_chain(llm,chat_prompt,self.memory)

    def run(self, user_input):
        return self.llm_chain.run(human_input = user_input, history=self.memory.chat_memory.messages, stop="Human:")