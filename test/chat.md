Great observation! But this is actually a **misconception** about how Pinecone works:

---

## One Index = Unlimited Videos 🎯

The **5 index limit** is about the number of *indexes* (like separate databases), **not** the number of vectors or videos inside one index.

A single Pinecone index can store **millions of vectors** from **thousands of videos**. You only ever need **1 index** for this entire project.

The way you separate videos inside that one index is through **namespaces** or **metadata filtering** — which you're already doing with `video_id`.

---

## Free Tier Limits (What Actually Matters)

| Limit | Free Tier |
|---|---|
| Indexes | 5 |
| Storage | 2 GB |
| Vectors | ~200K–300K vectors |
| Namespaces per index | Unlimited |

So with **1 index** you can comfortably store **hundreds of YouTube videos** before hitting the 2GB storage limit.

---

## Use Namespaces for Clean Separation (Optional but Recommended)

Instead of just metadata, you can isolate each video into its own **namespace** inside the single index:

```python
# Store chunks under a namespace = video_id
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    index_name=index_name,
    namespace=video_id        # 👈 each video gets its own namespace
)

# Query a specific video's namespace
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "namespace": video_id   # 👈 scoped to just this video
    }
)

# Query ALL videos at once (no namespace = searches everything)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)
```

---

## Architecture With 1 Index

```
Pinecone Index: "youtube-rag"
├── Namespace: iys_pmJSp9M      → 45 chunks (Video 1)
├── Namespace: dQw4w9WgXcQ      → 38 chunks (Video 2)
├── Namespace: another_id       → 52 chunks (Video 3)
└── ... hundreds more videos
```

So your original plan is totally fine — just use **1 index** for everything and you won't hit any real limits for a long time. Want to proceed with building this out?