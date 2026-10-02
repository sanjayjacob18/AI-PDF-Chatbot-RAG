import os
from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

app = FastAPI()

# 1. Connect to MongoDB Atlas (Paste your updated connection link here)
# Make sure to replace <db_password> with your actual database user password!
# Erase your actual link and replace it with a clean placeholder text
MONGO_URI = "mongodb+srv://sanjayjacobcross_db_user:<YOUR_PASSWORD>@cluster0.bflrnk7.mongodb.net/?appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["pdf_chat_db"]
collection = db["chunks"]

# 2. Load the free embedding model (runs locally on your PC)
print("Loading AI Embedding Model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_reader = PdfReader(file.file)
    raw_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text + "\n"
            
    if not raw_text.strip():
        return {"error": "Could not extract text from the PDF."}

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_text(raw_text)

    documents_to_insert = []
    for i, chunk in enumerate(text_chunks):
        vector_embedding = embedding_model.encode(chunk).tolist()
        doc = {
            "chunk_id": f"{file.filename}_chunk_{i}",
            "filename": file.filename,
            "text": chunk,
            "embedding": vector_embedding
        }
        documents_to_insert.append(doc)
        
    if documents_to_insert:
        collection.insert_many(documents_to_insert)
        
    return {"message": f"Successfully processed and stored {len(documents_to_insert)} text chunks!"}

@app.get("/query")
def ask_question(question: str):
    if not question.strip():
        return {"error": "Please provide a valid question."}

    question_vector = embedding_model.encode(question).tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",      
                "path": "embedding",          
                "queryVector": question_vector,
                "numCandidates": 100,         
                "limit": 3                    
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    results = list(collection.aggregate(pipeline))
    if not results:
        return {"answer": "No matching text chunks could be found."}

    retrieved_context = "\n---\n".join([doc["text"] for doc in results])
    ai_prompt_blueprint = (
        f"You are an expert AI Assistant. Use ONLY the following text fragments to answer the user's question.\n"
        f"TEXT FRAGMENTS:\n{retrieved_context}\n\n"
        f"USER QUESTION: {question}"
    )

    return {
        "matched_chunks": results,
        "llm_prompt_ready": ai_prompt_blueprint
    }

@app.get("/")
def home():
    return {"status": "AI Backend Engine Running"}
