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

## 5. OpenAI Query: With RAG vs. Without RAG Comparison

The application features a direct comparison mode between standard LLM queries and Vector RAG queries:

### ⚖️ Comparison Table

| Dimension | 🤖 OpenAI Query (WITHOUT RAG) | ⚡ OpenAI Query (WITH ChromaDB RAG) |
| :--- | :--- | :--- |
| **Prompt Structure** | Sent directly to OpenAI: `"Question: {query}"` | Augmented with retrieved vectors: `"Context from ChromaDB:\n{context}\n\nQuestion: {query}"` |
| **Knowledge Source** | Generic pre-training weights only. | Custom, verified vector facts stored in ChromaDB. |
| **Custom / Private Data** | **Fails / Hallucinates**: Cannot know private company policies, newly added notes, or domain data in ChromaDB. | **Grounded & Factual**: Directly answers from the nearest matched documents in ChromaDB. |
| **Citations** | None. | Explicit verifiable citations (e.g. `[doc_3]`, `[doc_1]`). |
| **Strict Mode Handling** | Answers regardless of knowledge accuracy. | **Strictly rejects** if the query is out-of-domain or below the distance cutoff. |

### 🖥️ Implementation in the Streamlit Web Application

1. **Sidebar Configuration**:
   - Provide an `OPENAI_API_KEY` in the sidebar (or reads automatically from `st.secrets` or environment variable).
   - Select the target OpenAI model: `gpt-4o-mini`, `gpt-4o`, or `gpt-3.5-turbo`.
2. **Display Mode**:
   - Select **`⚖️ OpenAI Compare: With RAG vs. Without RAG`** from the Display Mode dropdown.
3. **Split-Column Interface**:
   - **Left Column**: Displays the raw, unaugmented OpenAI response highlighting limitations on custom knowledge.
   - **Right Column**: Displays the ChromaDB vector-grounded response with document citations `[doc_X]`, distance metrics, and strict threshold enforcement.
   - **Demo Preview**: If no API key is provided, an interactive simulation is displayed explaining the exact contrast without failing.

### 💻 Implementation in the Terminal CLI

In `simple_chroma_demo.py`, selecting option `1` (Query Vector Database) runs the query and offers an automated OpenAI comparison view when `OPENAI_API_KEY` is present.

---

## 6. OpenAI API Key Security & Best Practices in Streamlit

### Why Pre-Filling an API Key into a Text Input is Dangerous
If you have an API key stored on your server (e.g., in `st.secrets` or environment variables) and write:
```python
# ❌ INSECURE: Do NOT pre-fill server secrets into client inputs!
st.text_input("OpenAI Key:", value=st.secrets["OPENAI_API_KEY"], type="password")
```
Even though the password field masks the characters on the screen (`••••••••`), the actual plain text secret is sent to the visitor's browser DOM. Any visitor can right-click the input box, select **Inspect Element**, and view your secret API key!

### The Secure Solution Implemented in `simple_chroma_app.py`
We decoupled server-side secrets from user inputs:
1. **Server Secret (Streamlit Cloud Secrets / Env Var)**:
   - When `OPENAI_API_KEY` is found in `st.secrets` or environment variables, the key is used **strictly in server memory**.
   - The client-side text box is **hidden**, and a secure badge is shown:
     `🔒 API Key: Secured on Server (Hidden from browser visitors)`
2. **User Input Fallback (Bring Your Own Key - BYOK)**:
   - If no server key exists, users get an **empty** password input box (`value=""`).
   - Each visitor's input is isolated to their own individual browser session in Streamlit memory; other visitors cannot see or access what another user typed into their own session.

---

## 7. The Effect of Temperature in LLMs & RAG Systems

### What Temperature Does
In Large Language Models, temperature ($T$) controls the randomness and probability distribution when sampling the next word/token from the model's vocabulary logits ($z_i$):

$$P(w_i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

### Comparison: Low vs. High Temperature

| Dimension | 🧊 Low Temperature ($T = 0.0$ to $0.2$) | 🔥 High Temperature ($T = 0.8$ to $1.2+$) |
| :--- | :--- | :--- |
| **Sampling Mechanism** | **Greedy Decoding**: Picks the top-scoring token with near 100% probability. | **Stochastic Sampling**: Flattens probabilities; lower-ranked tokens get picked. |
| **Output Style** | Concise, direct, literal, deterministic. | Creative, narrative, diverse vocabulary, expressive metaphors. |
| **Repeatability** | **100% Deterministic**: Asking the same question produces the identical answer every time. | **Probabilistic**: Asking the same question produces varied phrasing every time. |
| **Hallucination Risk** | **Near 0%** when grounded in RAG context. | **Elevated**: Model may embellish, speculate, or introduce unsupported claims. |
| **Best Used For** | Enterprise RAG, legal, medical, coding, financial compliance. | Storytelling, brainstorming, generating analogies, marketing copy. |

### Practical Example: Querying Venus Atmosphere with ChromaDB RAG

- **Query**: *"Describe the hostile atmosphere on Venus."*
- **Vector Evidence in ChromaDB `[doc_2]`**: *"Venus has a thick runaway greenhouse atmosphere of carbon dioxide and sulfuric acid, reaching 465°C."*

1. **At Low Temperature ($T = 0.0$)**:
   > *"Based on document `[doc_2]`, Venus has a thick runaway greenhouse atmosphere composed of carbon dioxide with clouds of sulfuric acid. Its surface temperature is approximately 465°C."*
   >
   > *(Observation: Strict, factual, zero filler, repeatable).*

2. **At High Temperature ($T = 1.2$)**:
   > *"Picture a hellish, suffocating landscape cloaked in a dense, poisonous shroud of carbon dioxide. According to `[doc_2]`, torrents of caustic sulfuric acid clouds lock in blistering heat, cooking the barren terrain at an unimaginable 465°C—a furnace intense enough to liquefy solid lead!"*
   >
   > *(Observation: Vivid, engaging, but introduces figurative embellishments like 'liquefy solid lead' that were not in the source text).*

### How It Is Implemented in the Application
1. **Interactive Temperature Slider**: Configurable from $0.0$ to $1.5$ in the sidebar.
2. **Dedicated Mode**: **`🌡️ Temperature Impact Lab: T=0.0 (Factual) vs. T=1.2 (Creative)`** executes the query at both extremes simultaneously and displays the outputs side-by-side.




