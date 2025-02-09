from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from dotenv import load_dotenv
import os
load_dotenv()
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings,GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA,LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import TFIDFRetriever
import pickle

class ChatBotUS():
    def __init__(self, k = 30, url = 'local',text = ''):  
        self.url = url
        self.text = text
        self.chat_history = []
        
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        self.system_prompt = (
            "Bạn là một trợ lí phục vụ sinh viên Khoa học tự nhiên"
            "Hãy dựa trên những thông tin được cung cấp sau đây để trả lời câu hỏi một cách thân thiện, xưng hô là mình và cậu"
            "Nếu không có bất kì thông tin thì trả mình 'Mình không có thông tin này'. Không tự bịa câu trả lời ra, hay lấy từ nguồn khác"
            "Không được tự bịa ra câu trả lời. Dưới đây là tài liệu cung cấp cho bạn."
            "\n"
            "### Tài liêu: {context}"
        )

        self.k = k
        self.load_if_idf()
        self.load_db()
        self.load_llm()
                    
        self.create_template()     
        self.create_qa_chain() 
    
    def load_if_idf(self):
        with open("vectorstores/tfidf_retriever.pkl", "rb") as f:
            self.retriever_tf_idf = pickle.load(f)

    
    def get_data_from_url(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)

        driver.get(self.url)

        html_content = driver.page_source

        driver.quit()

        soup = BeautifulSoup(html_content, 'html.parser')

        data = soup.getText(separator='\n',strip= True)  
        
        return data
    
    def load_db(self):
        embeddings_model = GPT4AllEmbeddings(model_file='model/all-MiniLM-L6-v2-f16.gguf')
        
        self.db = FAISS.load_local('vectorstores',embeddings_model,allow_dangerous_deserialization=True)
        return self.db
    

    def create_vectorstores(self,data):
    
        text_spliter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=400,
            length_function=len,
            is_separator_regex=False,
        )
        
        data = data.lower()
        chunks = text_spliter.split_text(data)
        embeddings_model = GPT4AllEmbeddings(model_file='model/all-MiniLM-L6-v2-f16.gguf')

        db = FAISS.from_texts(texts=chunks, embedding=embeddings_model)
        retriever_embeddings = db.as_retriever(search_kwargs={"k": 20})
        retriever_if_idf = TFIDFRetriever.from_texts(chunks)
        
        combine_retriever  = EnsembleRetriever(
                retrievers = [retriever_if_idf,retriever_embeddings],
                weights  = [0.3,0.7]
            )
        
        return combine_retriever

    def load_llm(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature= 0,
            max_tokens=None,
            timeout=None,
            api_key=os.getenv('API_KEY')
        )

    def create_template(self):
        
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
    def create_qa_chain(self):
        combine_retriever = None
        if self.url == 'local' :
            
            combine_retriever  = EnsembleRetriever(
                retrievers = [self.retriever_tf_idf,self.db.as_retriever(search_kwargs={"k": self.k})],
                weights  = [0.3,0.7]
            )
        if self.url != 'local':
            if self.get_data_from_url() != None:
                combine_retriever = self.create_vectorstores(self.get_data_from_url())
            
        if self.text != '':
            combine_retriever = self.create_vectorstores(self.text)
        
        history_aware_retriever = create_history_aware_retriever(
            llm=self.llm,
            retriever=combine_retriever,
            prompt=self.contextualize_q_prompt,
        )
        
        # Q&A
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, 
            self.qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, self.question_answer_chain)


    def make_response(self, question):
       
        ai_msg_1 = self.rag_chain.invoke({"input": question, "chat_history": self.chat_history})
        self.chat_history.extend(
            [
                HumanMessage(content=question),
                AIMessage(content=ai_msg_1["answer"]),
            ]
        )
        return ai_msg_1['answer']
