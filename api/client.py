import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import nest_asyncio
import uvicorn

# Load environment variables
load_dotenv()

# Load PDF documents
pdf_file_paths = ["../data/policy/Medical Policy FY 23-24.pdf", "../data/policy/Provident fund policy.pdf"]

docs_list = []
for pdf_path in pdf_file_paths:
    loader = PyMuPDFLoader(pdf_path)
    try:
        loaded_docs = loader.load()
        docs_list.extend(loaded_docs)
    except Exception as e:
        print(f"Error loading {pdf_path}: {e}")

# Split documents
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)
doc_splitter = text_splitter.split_documents(docs_list)

# Filter and clean metadata
filtered_doc = []
for doc in doc_splitter:
    if isinstance(doc, Document) and hasattr(doc, 'metadata'):
        clean_metadata = {k: v for k, v in doc.metadata.items() if isinstance(v, (str, int, float, bool))}
        filtered_doc.append(Document(page_content=doc.page_content, metadata=clean_metadata))

# Add to vectorDB
embedding = GPT4AllEmbeddings(model_name="all-MiniLM-L6-v2.gguf2.f16.gguf", gpt4all_kwargs={'allow_download': 'True'})
vectorstore = Chroma.from_documents(
    documents=filtered_doc,
    collection_name="rag-chroma",
    embedding=embedding,
)

retriever = vectorstore.as_retriever()

# Define prompt templates
context_prompt = PromptTemplate(
    template="""system You are an assistant for question-answering tasks.
    Use the following context to answer the question. Avoid phrases like "Based on the provided context". explain the answer in the end. and make a heading with paragraph
    Question: {question}
    Context: {context}
    Answer: assistant""",
    input_variables=["question", "context"],
)

no_context_prompt = PromptTemplate(
    template="""system You are an assistant for question-answering tasks.
    Answer the question concisely and directly.
    Question: {question}
    Answer: assistant""",
    input_variables=["question"],
)

# Load a pre-trained model for semantic similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load a text classification model for future usefulness prediction
usefulness_classifier = pipeline("text-classification", model="bert-base-uncased")

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Enhanced Function to check relevance using semantic similarity
def is_relevant(question, context_docs, threshold=0.3):
    context_text = format_docs(context_docs)

    # Split context into smaller chunks
    context_chunks = context_text.split('\n\n')

    # Encode the question
    question_embedding = model.encode(question, convert_to_tensor=True)

    max_similarity = 0
    for chunk in context_chunks:
        # Encode each chunk of context
        context_embedding = model.encode(chunk, convert_to_tensor=True)

        # Compute similarity
        similarity = util.pytorch_cos_sim(question_embedding, context_embedding)
        max_similarity = max(max_similarity, similarity.item())

    # Return True if similarity is above threshold
    return max_similarity > threshold

# Function to determine if a question/statement is useful for the future
def is_useful_for_future(question):
    # Use the classification pipeline to predict usefulness
    result = usefulness_classifier(question)
    is_useful = result[0]['label'] == 'USEFUL'
    if is_useful:
        with open("useful_question.txt", "a") as file:
            file.write(question + "\n")

    return result[0]['label'] == 'IMPORTANT'

# Function to get the appropriate chain based on relevance
def get_chain(question, context_docs, llm):
    if is_relevant(question, context_docs):
        print("Relevant")
        return context_prompt | llm | StrOutputParser()
    else:
        print("Irrelevant")
        return no_context_prompt | llm | StrOutputParser()

# Main function to run the question-answering process
def answer_question(question, llm_model):
    llm = ChatOllama(model=llm_model, temperature=0)
    docs = retriever.invoke(question)
    rag_chain = get_chain(question, docs, llm)
    if is_relevant(question, docs):
        generation = rag_chain.invoke({"context": docs, "question": question})
    else:
        generation = rag_chain.invoke({"question": question})

    # Check if the question is useful for the future
    important = is_useful_for_future(question)

    return generation, important

# FastAPI app
app = FastAPI()

# Request model
class QuestionRequest(BaseModel):
    question: str
    llm: str

# Response model
class AnswerResponse(BaseModel):
    answer: str
    responseTime: str
    important: bool

@app.post("/v1.0/ask/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    start_time = time.time()
    try:
        answer, important = answer_question(request.question, request.llm)
        response_time = "{:.2f} sec".format(time.time() - start_time)
        return AnswerResponse(answer=answer, responseTime=response_time, important=important)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Start the FastAPI server within the notebook
nest_asyncio.apply()
uvicorn.run(app, host="127.0.0.1", port=8000)
