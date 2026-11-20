# Streamlit Cloud Deployment & Data Storage Notes

Notes on how and where data is stored when running this ChromaDB RAG application on **Streamlit Community Cloud (`share.streamlit.io`)**.

---

## 1. Where the Data is Stored Right Now

In `simple_chroma_app.py`, ChromaDB is initialized with:

```python
@st.cache_resource
def get_chroma_collection():
    client = chromadb.Client()  # In-memory client
    ...
```

- **Location**: In the **RAM (Memory)** of the Streamlit Cloud Linux container allocated to your app.
- **Accessibility**: Because it is wrapped in `@st.cache_resource`, the in-memory ChromaDB instance is shared across all concurrent users/sessions currently connected to that running container.
- **No external cloud transfer**: No data is uploaded to GitHub or any third-party database.

---

## 2. What Happens When the App Goes to Sleep or Restarts?

Streamlit Community Cloud provides **ephemeral (temporary) container hosting**:

| Event | What Happens to the Entered Data? |
| :--- | :--- |
| **New user visits while app is awake** | ✅ The data is **available** (shared across users via `@st.cache_resource`). |
| **User refreshes browser tab** | ✅ The data **remains** in the running container. |
| **App goes to sleep** *(no visitors for a few days)* | ⚠️ The container shuts down; **RAM is cleared**. |
| **You push a new git commit** | ⚠️ The app rebuilds from GitHub; **data resets** to the default sample documents. |
| **You click "Reboot app" in Streamlit dashboard** | ⚠️ The container restarts fresh; **data resets**. |

---

## 3. What If You Used `PersistentClient(path="./data")` on Streamlit Cloud?

Even if you save to local disk inside Streamlit Cloud using:

```python
client = chromadb.PersistentClient(path="./my_data")
```

It writes to the **container's temporary virtual disk**. While it will survive Python script re-runs within the same active session, **it will still be wiped when Streamlit reboots or rebuilds the container**, because Streamlit Cloud containers are stateless.

---

## 4. How to Make Storage 100% Permanent on Streamlit Cloud

If you want data added through the web portal to **never disappear**, here are the common solutions:

### Option A: Pre-populate Documents in Your GitHub Repo (Easiest)
If you have documents or articles you want your RAG portal to always have:
1. Place text/markdown files directly in your GitHub repository (e.g. in a `data/` folder).
2. Have `simple_chroma_app.py` read and index them on startup.
3. Every time Streamlit rebuilds, your documents are automatically embedded and ready.

### Option B: Connect to a Remote Cloud ChromaDB / Vector Store
Instead of running ChromaDB inside the temporary container, connect Streamlit to a permanent hosted backend:

```python
# Connect to a remote ChromaDB server (e.g., hosted on Render, AWS, or Chroma Cloud)
client = chromadb.HttpClient(
    host="your-chroma-server.onrender.com",
    port=443,
    ssl=True
)
```

### Option C: External Database or Cloud Storage
- Save raw added documents to **AWS S3**, **Supabase**, or **Google Cloud Storage**.
- Or use a free cloud vector database like **Pinecone**, **Supabase (pgvector)**, or **Weaviate Cloud**.

---

## Summary

For **demos, tests, and interactive user queries**, the current in-memory setup on Streamlit Cloud is ideal because it is **fast, completely free, requires zero credentials, and automatically self-resets to a clean state**.

