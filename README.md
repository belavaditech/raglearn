# Simple ChromaDB Interactive RAG Application

A clean, easy-to-understand interactive Retrieval-Augmented Generation (RAG) application using **ChromaDB**, built with reference to GeeksforGeeks guides on vector databases and ChromaDB.

---

## 📚 References & Learning Resources

This demo is designed around core vector database and NLP concepts documented in:
- **[What is a Vector Database? - GeeksforGeeks](https://www.geeksforgeeks.org/data-science/what-is-a-vector-database/)**: An introduction to high-dimensional embeddings, similarity metrics (cosine similarity / Euclidean distance), indexing methods (HNSW), and how vector databases power LLM context retrieval.
- **[Introduction to ChromaDB - GeeksforGeeks](https://www.geeksforgeeks.org/nlp/introduction-to-chromadb/)**: A hands-on guide covering ChromaDB architecture, initializing clients (`chromadb.Client()`), creating collections, embedding generation, adding documents, and semantic querying.

---

## 🌟 Features Implemented

1. **Entering Data into Vector Database (ChromaDB)**:
   - Initializes an in-memory ChromaDB client: `chromadb.Client()`
   - Creates a collection with cosine space: `client.get_or_create_collection(name="...", metadata={"hnsw:space": "cosine"})`
   - Adds text documents, IDs, and metadata: `collection.add(documents=..., ids=..., metadatas=...)`

2. **Executing Natural Language Queries**:
   - Queries the collection: `collection.query(query_texts=[...], n_results=...)`
   - Automatically computes embeddings using ChromaDB's local ONNX model (`all-MiniLM-L6-v2`).

3. **Option to Display Vector Query Outputs**:
   - Displays the raw results returned by ChromaDB:
     - Matched document IDs (`results['ids']`)
     - Matched document texts (`results['documents']`)
     - Cosine distance scores (`results['distances']`)
     - Cosine similarity percentages ($1 - \text{distance}$)
     - Document metadata (`results['metadatas']`)
     - Full JSON output inspectable in one click.

4. **Option: Flexible vs. Strict Grounding**:
   - **Strict Mode**: Enforces a strict maximum distance cutoff (e.g., $\text{distance} \le 0.65$).
     - If query matches are above the cutoff (irrelevant query or out-of-domain), the system **strictly rejects** the query to eliminate hallucination.
     - Answers strictly cite verified vector facts.
   - **Flexible Mode**: Relaxes cutoff constraints, retrieving nearest semantic neighbors and synthesizing a broader summary.

5. **Option to Display Result Based on Vector Data Found**:
   - **Synthesized Answer**: Formulates an answer directly grounded on the retrieved vector chunks.
   - **Direct Matched Document Evidence**: Verbatim display of the exact matched document cards.
   - **Side-by-Side Comparison**: Live split view showing how Strict vs. Flexible modes process the same query.

---

## ❓ Frequently Asked Questions (FAQ)

### Q1: Do I need an API key to use ChromaDB?
**No.** ChromaDB is an open-source, embedded vector database library (similar to SQLite, but purpose-built for vector embeddings). It runs 100% locally on your computer with zero external accounts or paid API keys required.

### Q2: Where is the vector database currently stored in this demo?
In this application, we initialize ChromaDB using:
```python
client = chromadb.Client()
```
- **Storage Location**: Inside your computer's **System RAM (volatile memory)**.
- **Persistence**: Vectors, document texts, and HNSW indexes live in the active memory of the Python process. No database files are written to disk, and no cloud servers are contacted.

### Q3: How are vector embeddings calculated without an OpenAI or Gemini API key?
ChromaDB bundles its own **built-in local embedding model** (`all-MiniLM-L6-v2` running via ONNX runtime):
- The model files were downloaded on first run to your local cache:
  `C:\Users\User\.cache\chroma\onnx_models\all-MiniLM-L6-v2\`
- When you execute `collection.add(...)` or `collection.query(...)`, ChromaDB uses your local CPU to encode text into 384-dimensional vectors completely offline.

### Q4: How can I save the vector database permanently on disk instead of in-memory?
To keep your data across restarts, replace `chromadb.Client()` with `chromadb.PersistentClient()`:
```python
import chromadb

# Saves directly to a local folder on your disk
client = chromadb.PersistentClient(path="./my_chroma_data")
collection = client.get_or_create_collection(name="simple_rag_collection")
```
When run, ChromaDB creates a `./my_chroma_data/` folder on your hard drive containing:
- `chroma.sqlite3`: Local SQLite database storing document text and metadata.
- Vector index files: HNSW graph binary files storing the vector embeddings.

### Q5: How does ChromaDB compare to external cloud vector databases?

| Feature | `chromadb.Client()` (Current Demo) | `chromadb.PersistentClient()` | Cloud DBs (Pinecone, Weaviate Cloud) |
| :--- | :--- | :--- | :--- |
| **Storage Location** | System RAM | Local Hard Drive Folder | Remote Cloud Servers |
| **API Key Needed?** | ❌ No | ❌ No | ✅ Yes |
| **Network Dependent?** | ❌ 100% Offline | ❌ 100% Offline | ✅ Requires Internet |
| **Best For** | Demos, tests, fast experiments | Local tools, desktop apps, offline RAG | Large-scale multi-user production apps |

---

## 🚀 How to Run

### Method 1: Interactive Streamlit Web App (Recommended)

Run:
```bash
streamlit run simple_chroma_app.py
```
Open your browser at `http://localhost:8501`.

### Method 2: Standalone Interactive CLI

Run directly in your terminal:
```bash
python simple_chroma_demo.py
```

### Method 3: Run the Verification Test

Run:
```bash
python test_chroma.py
```

---

## 🌐 How to Host the Application as a Web Portal

You can make this RAG application accessible as a public or team web portal using any of the following 5 methods:

### Option 1: Streamlit Community Cloud (Recommended — 100% Free Public Web Portal)
Streamlit provides free cloud hosting with permanent HTTPS URLs directly from GitHub.

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial ChromaDB RAG demo"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```
2. **Deploy on Streamlit Cloud**:
   - Visit **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub.
   - Click **"Create app"**.
   - Select your repository and branch (`main`).
   - Set **Main file path** to:
     ```
     simple_chroma_app.py
     ```
   - Click **"Deploy!"**.
   - Streamlit Cloud reads `requirements.txt`, installs dependencies, and assigns you a live public URL (e.g., `https://your-rag-portal.streamlit.app`).

### Option 2: Hugging Face Spaces (Free Cloud AI Portal)
1. Go to **[Hugging Face Spaces](https://huggingface.co/spaces)** and click **"Create new Space"**.
2. Select **Streamlit** as the Space SDK and choose the **Free CPU** tier.
3. Upload `simple_chroma_app.py`, `requirements.txt`, and `.streamlit/config.toml` (or push via Git).
4. Your application will build and launch at `https://huggingface.co/spaces/<your-username>/<space-name>`.

### Option 3: Self-Host on Local Network / Private Server / Cloud VM (AWS, GCP, DigitalOcean)
To make it accessible to everyone on your local network, private server, or cloud VM:
```bash
streamlit run simple_chroma_app.py --server.port 8501 --server.address 0.0.0.0
```
- **From your own machine**: `http://localhost:8501`
- **From any device on the network / internet**: `http://<YOUR_SERVER_IP>:8501`

*(The included `.streamlit/config.toml` is pre-configured to bind to `0.0.0.0:8501` automatically).*

### Option 4: Instant Public URL from Local Machine (Cloudflare Tunnel / ngrok)
If you want to keep running the application locally on your computer but share a secure public HTTPS link with anyone in the world:
1. Start the Streamlit app:
   ```bash
   streamlit run simple_chroma_app.py
   ```
2. In a second terminal, launch a free tunnel:
   - **Using Cloudflare Tunnel (Free, no account required)**:
     ```bash
     cloudflared tunnel --url http://localhost:8501
     ```
   - **Using ngrok**:
     ```bash
     ngrok http 8501
     ```
3. Copy the generated temporary HTTPS URL (e.g. `https://random-subdomain.trycloudflare.com`) and open it from any browser worldwide.

### Option 5: Docker Container Deployment
Use the included `Dockerfile` to deploy as a containerized web service to AWS ECS, Google Cloud Run, Azure Container Apps, or Kubernetes:
```bash
# 1. Build the Docker image
docker build -t chromadb-rag-portal .

# 2. Run the container
docker run -p 8501:8501 chromadb-rag-portal
```

---

## 📂 Project Files

- **`simple_chroma_app.py`**: Clean, visual Streamlit web demo utilizing ChromaDB.
- **`simple_chroma_demo.py`**: Standalone terminal script directly reflecting the GeeksforGeeks tutorial style.
- **`test_chroma.py`**: Automated test verifying ChromaDB collection creation, document insertion, and cosine query search.
- **`requirements.txt`**: Python dependencies (`streamlit`, `chromadb`, `pandas`) for cloud deployments.
- **`.streamlit/config.toml`**: Server port, headless mode, and UI theme configuration.
- **`Dockerfile`**: Docker container build specification for portable web hosting.
- **`README.md`**: Architecture guide, GeeksforGeeks references, FAQ, and hosting documentation.
