import faiss
import numpy as np

DIM=384
index=faiss.IndexFlatL2(DIM)

vectors=[]
metadata=[]

def add_vector(vector,meta):
    global vectors,metadata,index
    vector=np.array(vector).astype("float32")
    index.add(np.array([vector]))
    metadata.append(meta)

def semantic_search(query_vec,top_k=3):
    query_vec=np.array(query_vec).astype("float32").reshape(1,-1)
    distances,ids=index.search(query_vec,top_k)

    results=[]

    for i, idx in enumerate(ids[0]):
        if idx < len(metadata):
            results.append({
                "distance": float(distances[0][i]),
                "metadata": metadata[idx]
            })
    return results