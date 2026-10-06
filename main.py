import os
from fastapi import FastAPI, UploadFile, File
from pymongo import MongoClient
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

app = FastAPI()

# 1. Connect to MongoDB Atlas (Double-check your credentials look exactly like this)
# 1. Connect to MongoDB Atlas
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://sanjayjacobcross_db_user:<YOUR_PASSWORD>@cluster0.bflrnk7.mongodb.net/?appName=Cluster0")
client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
db = client["pdf_chat_db"]
collection = db["chunks"]

# 2. Load the free embedding model (runs locally on your PC)
print("Loading AI Embedding Model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully!")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Read the raw binary PDF data stream
        pdf_reader = PdfReader(file.file)
        raw_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text + "\n"
        
        if not raw_text.strip():
            return {"error": "The uploaded PDF contains no extractable text."}

        # Initialize LangChain text segmenter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)
        
        # Vectorize and push to cloud cluster
        documents_to_insert = []
        for i, chunk in enumerate(chunks):
            vector = embedding_model.encode(chunk).tolist()
            documents_to_insert.append({
                "filename": file.filename,
                "chunk_id": i,
                "text": chunk,
                "vector": vector
            })
            
        if documents_to_insert:
            collection.insert_many(documents_to_insert)
            
        return {"message": f"Successfully processed and stored {len(chunks)} text chunks!"}
        
    except Exception as e:
        return {"error": str(e)}

@app.get("/query")
async def query_pdf(question: str):
    try:
        # Convert user search question into mathematical array coordinates
        question_vector = embedding_model.encode(question).tolist()
        
        # Execute MongoDB Vector Proximity Search matching
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "vector",
                    "queryVector": question_vector,
                    "numCandidates": 10,
                    "limit": 3
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "filename": 1,
                    "text": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(collection.aggregate(pipeline))
        return {"results": results}
        
    except Exception as e:
        return {"error": str(e)}
