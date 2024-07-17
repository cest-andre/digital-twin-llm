from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory


class VectorDBAssistant():
    def __init__(self, llm, embeddings, vector_db_path, prompt_template, retrieve_count=12, dataload=None):
        retriever = None
        if dataload is not None:
            db = Chroma.from_texts(dataload, embeddings, persist_directory=vector_db_path)
            retriever = db.as_retriever(search_kwargs={"k": retrieve_count})
        else:
            retriever = Chroma(persist_directory=vector_db_path, embedding_function=embeddings).as_retriever(search_kwargs={"k": retrieve_count})

        custom_prompt = PromptTemplate(template=prompt_template, input_variables=['chat_history', 'context', 'question'])
        chat_memory = ConversationBufferMemory(k=8, memory_key='chat_history', return_messages=True)
        chain_type_kwargs = {"prompt": custom_prompt}

        self.chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=chat_memory, combine_docs_chain_kwargs=chain_type_kwargs)