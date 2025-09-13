# # app.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from src import config
# from src.retriever import Retriever

# app = FastAPI(
#     title="Urdu QA Bot",
#     description="Retrieve answers + related answers from Urdu dataset",
#     version="1.0.0"
# )

# # Load retriever once at startup
# retriever = Retriever(
#     dataset_path=config.DATASET_PATH,
#     faiss_index_path=config.FAISS_INDEX_PATH,
#     meta_path=config.QUESTIONS_META_PATH,
#     model_name=config.EMBEDDING_MODEL_NAME,
#     device=config.DEVICE
# )

# class QueryRequest(BaseModel):
#     question: str
#     top_k: int = config.TOP_K

# @app.post("/query")
# def query_bot(request: QueryRequest):
#     results = retriever.search(request.question, top_k=request.top_k)

#     if not results:
#         return {
#             "query": request.question,
#             "main_answer": None,
#             "related": []
#         }

#     main_answer = results[0]
#     related = results[1:]

#     return {
#         "query": request.question,
#         "main_answer": main_answer,
#         "related": related
#     }

# @app.get("/")
# def home():
#     return {"message": "Urdu QA Bot API is running. Use /query endpoint."}


from fastapi import FastAPI
from pydantic import BaseModel
from src import config
from src.retriever import Retriever

app = FastAPI(
    title="Urdu QA Bot",
    description="Retrieve answers + related answers from Urdu dataset",
    version="1.0.0"
)

# ✅ Retriever Hugging Face dataset se load karega (paths ki zarurat nahi)
retriever = Retriever(
    model_name=config.EMBEDDING_MODEL_NAME,
    device=config.DEVICE
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K

@app.post("/query")
def query_bot(request: QueryRequest):
    results = retriever.search(request.question, top_k=request.top_k)

    if not results:
        return {
            "query": request.question,
            "main_answer": None,
            "related": []
        }

    main_answer = results[0]
    related = results[1:]

    return {
        "query": request.question,
        "main_answer": main_answer,
        "related": related
    }

@app.get("/")
def home():
    return {"message": "✅ Urdu QA Bot API is running. Use POST /query with {'question': '...'}"}
