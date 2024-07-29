from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from docx import Document as DocxDocument
import time
from typing import List, Any


app = FastAPI()

# Initialize model and components
local_llm = 'llama3'
cascade_directory = "../data/cascade"
policy_directory = "../data/policy"
directories = [cascade_directory, policy_directory]
pdf_file_paths = []
docx_file_paths = []

for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            pdf_file_paths.append(os.path.join(directory, filename))
        elif filename.endswith(".docx"):
            docx_file_paths.append(os.path.join(directory, filename))

docs_list = []

for pdf_path in pdf_file_paths:
    loader = PyMuPDFLoader(pdf_path)
    try:
        loaded_docs = loader.load()
        docs_list.extend(loaded_docs)
    except Exception as e:
        print(f"Error loading {pdf_path}: {e}")

for docx_path in docx_file_paths:
    try:
        docx = DocxDocument(docx_path)
        full_text = []
        for para in docx.paragraphs:
            full_text.append(para.text)
        docx_text = '\n'.join(full_text)
        docs_list.append(Document(page_content=docx_text, metadata={"source": docx_path}))
    except Exception as e:
        print(f"Error loading {docx_path}: {e}")

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splitter = text_splitter.split_documents(docs_list)

filtered_doc = []
for doc in doc_splitter:
    if isinstance(doc, Document) and hasattr(doc, 'metadata'):
        clean_metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
        filtered_doc.append(Document(page_content=doc.page_content, metadata=clean_metadata))

embedding = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={'allow_download': 'True'})
vectorstore = Chroma.from_documents(
    documents=filtered_doc,
    collection_name="rag-chroma",
    embedding=embedding,
)

retriever = vectorstore.as_retriever()
llm = ChatOllama(model=local_llm, temperature=0)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """system You are an assistant for question-answering tasks. Use the following context to answer the question. Avoid phrases like "Based on the provided context". Explain the answer in the end. and make a heading with paragraph.
Question: {input}
Context: {context}
Answer: assistant"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

chat_history = []

class QueryModel(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    responseTime: str
    # context: List[Any]
@app.post("/ask")
async def ask_question(query: QueryModel):
    start_time = time.time()
    try:
        question = query.question
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
        response_time = "{:.2f} sec".format(time.time() - start_time)
        chat_history.extend([HumanMessage(content=question), response["answer"]])

        # if "context" in response:
        #     context = response["context"]
        # else:
        #     context = []

        chat_history.extend([HumanMessage(content=question), response["answer"]])
        return AnswerResponse(answer=response["answer"], responseTime=response_time)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
