import json
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src import config

def build_index():
    config.ensure_dirs()

    # Load cleaned dataset
    dataset_path = config.DATA_DIR / "dataset_clean.csv"
    df = pd.read_csv(dataset_path)

    # Load model
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE)

    # Generate embeddings (normalize for cosine similarity)
    embeddings = model.encode(
        df[config.COL_QUESTION].tolist(),
        batch_size=config.EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    # Save embeddings for debugging / reuse
    np.save(config.EMBEDDINGS_NPY_PATH, embeddings)

    # Create FAISS index
    dim = embeddings.shape[1]
    if config.FAISS_INDEX_FACTORY == "IndexFlatIP":
        index = faiss.IndexFlatIP(dim)  # cosine similarity
    else:
        index = faiss.IndexFlatL2(dim)  # euclidean distance

    index.add(embeddings)
    faiss.write_index(index, str(config.FAISS_INDEX_PATH))

    # Save metadata: map idx -> {id, question}
    meta = [
        {config.COL_ID: int(row[config.COL_ID]), config.COL_QUESTION: str(row[config.COL_QUESTION])}
        for _, row in df.iterrows()
    ]
    with open(config.QUESTIONS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"✅ Built FAISS index with {len(df)} entries.")

if __name__ == "__main__":
    build_index()
