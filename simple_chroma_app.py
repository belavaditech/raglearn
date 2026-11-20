"""
Simple ChromaDB Interactive RAG Application with OpenAI Comparison
Built with Streamlit, ChromaDB & OpenAI
Reference: https://www.geeksforgeeks.org/nlp/introduction-to-chromadb/
"""

import os
import streamlit as st
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="Simple ChromaDB RAG Demo | With vs Without RAG",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Simple ChromaDB RAG Demo")
st.caption("A clean implementation of Vector Database RAG using **ChromaDB** based on the [GeeksforGeeks Guide](https://www.geeksforgeeks.org/nlp/introduction-to-chromadb/), with **OpenAI With-RAG vs. Without-RAG** comparison.")

# Initialize ChromaDB Client & Collection in Session State
@st.cache_resource
def get_chroma_collection():
    import chromadb
    client = chromadb.Client()
    # Using cosine distance: distance = 1 - cosine_similarity (0 is identical)
    collection = client.get_or_create_collection(
        name="simple_rag_collection",
        metadata={"hnsw:space": "cosine"}
    )
    # Populate with initial sample data if empty
    if collection.count() == 0:
        sample_documents = [
            "Mercury is the closest planet to the Sun with extreme temperature swings from -180°C to 430°C.",
            "Venus has a thick runaway greenhouse atmosphere of carbon dioxide and sulfuric acid, reaching 465°C.",
            "Mars is the Red Planet featuring Olympus Mons, the largest volcano in the Solar System.",
            "Jupiter is the largest planet in our solar system, with 95 moons including Europa which has a subsurface ocean.",
            "Saturn has prominent rings made of ice and rock particles, and holds 146 confirmed moons."
        ]
        collection.add(
            documents=sample_documents,
            ids=[f"doc_{i+1}" for i in range(len(sample_documents))],
            metadatas=[{"category": "astronomy", "topic": f"Planet {i+1}"} for i in range(len(sample_documents))]
        )
    return client, collection

try:
    client, collection = get_chroma_collection()
except Exception as e:
    st.error(f"Error initializing ChromaDB: {e}")
    st.stop()


# Helper function to call OpenAI
def call_openai_api(prompt: str, system_message: str = "", api_key: str = "", model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].message.content


# Sidebar Controls
with st.sidebar:
    st.header("⚙️ RAG Options & Settings")

    st.subheader("1. Grounding Strictness")
    rag_mode = st.radio(
        "Mode:",
        options=["Strict", "Flexible"],
        index=0,
        help="Strict: Rejects queries where distance is above cutoff and enforces factual grounding.\nFlexible: Broader matching and synthesis."
    )

    max_distance_cutoff = st.slider(
        "Strict Distance Cutoff (Max Cosine Distance):",
        min_value=0.10,
        max_value=1.00,
        value=0.65,
        step=0.05,
        help="Lower distance = closer semantic match. Documents with distance > cutoff are rejected in Strict Mode."
    )

    n_results = st.slider("Number of Vector Matches (n_results):", min_value=1, max_value=5, value=2)

    st.divider()
    st.subheader("2. Result Presentation")
    result_display_mode = st.selectbox(
        "Display Mode:",
        options=[
            "⚖️ OpenAI Compare: With RAG vs. Without RAG",
            "Synthesized Answer Grounded on Vector Data (Local)",
            "Direct Matched Document Evidence",
            "Compare Side-by-Side (Strict vs Flexible)"
        ],
        index=0,
        help="Choose 'OpenAI Compare' to see how the same query is answered by OpenAI with and without vector data retrieved from ChromaDB."
    )

    show_vector_outputs = st.checkbox(
        "🔍 Display Vector Query Outputs (Distances & Chunks)",
        value=True,
        help="Display the raw outputs returned by collection.query(): distances, IDs, and document texts."
    )

    st.divider()
    st.subheader("3. OpenAI Configuration")
    # Read key from secrets or env var if available
    default_key = ""
    try:
        default_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    except Exception:
        default_key = os.getenv("OPENAI_API_KEY", "")

    openai_key = st.text_input("OpenAI API Key:", value=default_key, type="password", placeholder="sk-proj-...")
    openai_model = st.selectbox("OpenAI Model:", options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

    st.divider()
    if st.button("🗑️ Reset Collection to Sample Data", use_container_width=True):
        client.delete_collection("simple_rag_collection")
        st.cache_resource.clear()
        st.success("Reset ChromaDB collection!")
        st.rerun()


# Top Summary
doc_count = collection.count()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("ChromaDB Collection", collection.name)
with col2:
    st.metric("Total Indexed Documents", doc_count)
with col3:
    st.metric("Current Mode", f"{rag_mode} (Max Dist: {max_distance_cutoff})")

st.markdown("---")


# Section 1: Data Entry into ChromaDB
with st.expander("📥 Step 1: Enter New Data into ChromaDB", expanded=False):
    st.write("Add your own custom documents into the ChromaDB vector database:")
    col_in1, col_in2 = st.columns([3, 1])
    with col_in1:
        new_doc_text = st.text_area("Document Content:", placeholder="Type or paste custom knowledge (e.g. Project ApolloX launch date is Nov 24, 2026)...")
    with col_in2:
        new_doc_id = st.text_input("Document ID:", value=f"doc_{doc_count + 1}")
        category_tag = st.text_input("Category Tag:", value="general")
        add_btn = st.button("➕ Add to ChromaDB", type="primary", use_container_width=True)

    if add_btn:
        if new_doc_text.strip():
            collection.add(
                documents=[new_doc_text.strip()],
                ids=[new_doc_id.strip() or f"doc_{doc_count + 1}"],
                metadatas=[{"category": category_tag.strip()}]
            )
            st.success(f"Added document `{new_doc_id}` to ChromaDB collection!")
            st.rerun()
        else:
            st.warning("Please enter some text before adding.")

    # View existing items in ChromaDB
    if st.checkbox("👁️ View All Current Documents in ChromaDB"):
        all_data = collection.get()
        if all_data and all_data["documents"]:
            table_records = []
            for did, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
                table_records.append({"ID": did, "Document": doc, "Metadata": str(meta)})
            st.dataframe(pd.DataFrame(table_records), use_container_width=True)


# Section 2: Query ChromaDB
st.subheader("🔍 Step 2: Query the Vector Database")

example_queries = [
    "What is the largest volcano on Mars?",
    "Which planet has liquid water oceans under its crust?",
    "What are the rings of Saturn composed of?",
    "What is the recipe for chocolate chip cookies?"  # Non-matching test query
]

st.caption("Click an example query or type your own:")
cols_ex = st.columns(len(example_queries))
selected_query = ""
for i, eq in enumerate(example_queries):
    if cols_ex[i].button(eq, key=f"ex_{i}"):
        selected_query = eq

query_input = st.text_input("Enter your query:", value=selected_query, placeholder="e.g. Which planet is the hottest?")

if query_input.strip():
    # Execute query using ChromaDB API
    # GeeksforGeeks syntax: collection.query(query_texts=[...], n_results=...)
    query_results = collection.query(
        query_texts=[query_input.strip()],
        n_results=min(n_results, doc_count),
        include=["documents", "metadatas", "distances"]
    )

    matched_docs = query_results["documents"][0] if query_results["documents"] else []
    matched_distances = query_results["distances"][0] if query_results["distances"] else []
    matched_ids = query_results["ids"][0] if query_results["ids"] else []
    matched_metas = query_results["metadatas"][0] if query_results["metadatas"] else []

    # OPTION: Display Vector Query Outputs
    if show_vector_outputs:
        st.markdown("### 📊 Vector Query Outputs (ChromaDB Results)")
        st.caption(f"Nearest {len(matched_docs)} neighbors retrieved from ChromaDB for: *\"{query_input}\"*")

        vector_rows = []
        for i in range(len(matched_docs)):
            dist = matched_distances[i]
            sim = max(0.0, 1.0 - dist)
            is_pass = dist <= max_distance_cutoff
            vector_rows.append({
                "Rank": f"#{i+1}",
                "ID": matched_ids[i],
                "Cosine Distance": f"{dist:.4f}",
                "Similarity %": f"{sim * 100:.1f}%",
                f"Strict Check (<= {max_distance_cutoff})": "✅ Pass" if is_pass else "❌ Rejected",
                "Metadata": str(matched_metas[i]),
                "Document Preview": matched_docs[i]
            })

        st.dataframe(pd.DataFrame(vector_rows), use_container_width=True)

        # Raw JSON output expander
        with st.expander("🛠️ Inspect Raw ChromaDB Output Dictionary (`results`)"):
            st.json(query_results)

    st.markdown("---")

    # Filter passing documents for Strict Mode
    passing_indices = [i for i, d in enumerate(matched_distances) if d <= max_distance_cutoff]
    has_strict_pass = len(passing_indices) > 0

    # OPTION: Display Result Based on Vector Data Found
    st.markdown("### 🎯 Result Based on Vector Data Found")

    # -------------------------------------------------------------
    # FEATURE: OpenAI Comparison (With RAG vs Without RAG)
    # -------------------------------------------------------------
    if result_display_mode == "⚖️ OpenAI Compare: With RAG vs. Without RAG":
        st.write("Compare the exact same query evaluated by **OpenAI Without RAG** vs. **OpenAI With ChromaDB RAG**:")

        # Build context string from vector search
        valid_indices = passing_indices if rag_mode == "Strict" else list(range(len(matched_docs)))
        rag_context = "\n".join([f"[{matched_ids[i]}]: {matched_docs[i]}" for i in valid_indices])

        col_no_rag, col_rag = st.columns(2)

        # --- LEFT COLUMN: OpenAI WITHOUT RAG ---
        with col_no_rag:
            st.markdown("#### 🤖 OpenAI Query (WITHOUT RAG)")
            st.caption("Prompt sent directly to OpenAI with zero vector context.")

            no_rag_box = st.empty()

            if openai_key.strip():
                try:
                    with st.spinner("Calling OpenAI (Without RAG)..."):
                        no_rag_answer = call_openai_api(
                            prompt=query_input,
                            system_message="You are a helpful assistant. Answer the user question directly based solely on your pre-trained knowledge.",
                            api_key=openai_key.strip(),
                            model=openai_model
                        )
                    no_rag_box.markdown(f"**Answer:**\n\n{no_rag_answer}")
                except Exception as err:
                    no_rag_box.error(f"OpenAI API Error: {err}")
            else:
                no_rag_box.warning(
                    "🔑 **No OpenAI API Key Entered in Sidebar**\n\n"
                    "*Demo Preview (Without RAG):*\n"
                    f"> The base model relies **only** on its public training data. If your query asks about custom private documents "
                    f"added to ChromaDB, the model will say: *'I do not have access to your private database or current real-time documents.'*"
                )

            st.markdown(
                """
                <div style="background-color: #FEF2F2; border-left: 4px solid #EF4444; padding: 10px; border-radius: 4px; font-size: 0.88rem; margin-top: 10px;">
                    <b>⚠️ Without RAG Limitations:</b>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Has no access to documents in ChromaDB</li>
                        <li>Susceptible to hallucinations on custom data</li>
                        <li>Cannot provide verifiable document citations</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        # --- RIGHT COLUMN: OpenAI WITH RAG ---
        with col_rag:
            st.markdown("#### ⚡ OpenAI Query (WITH ChromaDB RAG)")
            st.caption(f"Prompt augmented with {len(valid_indices)} retrieved ChromaDB document(s).")

            rag_box = st.empty()

            if rag_mode == "Strict" and not has_strict_pass:
                rag_box.error(
                    f"⛔ **Strict Grounding Rejection:**\n\n"
                    f"ChromaDB nearest match distance (`{matched_distances[0]:.4f}`) exceeded strict threshold (`{max_distance_cutoff}`).\n"
                    f"In Strict Mode, the system refuses to send ungrounded context to OpenAI."
                )
            elif openai_key.strip():
                try:
                    if rag_mode == "Strict":
                        system_msg = (
                            "You are a strict, factual assistant. Answer the question ONLY using the facts explicitly stated in the provided context.\n"
                            "Cite your sources using [doc_X]. If the context does not contain sufficient facts to answer, "
                            "reply: 'I cannot answer this question based on the provided ChromaDB documents.'"
                        )
                    else:
                        system_msg = (
                            "You are a helpful assistant. Use the provided context from ChromaDB as your primary source of truth.\n"
                            "Synthesize a clear, helpful response and cite sources [doc_X]."
                        )

                    rag_prompt = f"Context from ChromaDB:\n{rag_context}\n\nQuestion: {query_input}"

                    with st.spinner("Calling OpenAI (With ChromaDB RAG)..."):
                        rag_answer = call_openai_api(
                            prompt=rag_prompt,
                            system_message=system_msg,
                            api_key=openai_key.strip(),
                            model=openai_model
                        )
                    rag_box.markdown(f"**Answer:**\n\n{rag_answer}")
                except Exception as err:
                    rag_box.error(f"OpenAI API Error: {err}")
            else:
                # Simulated demonstration output when no API key is provided
                top_doc_text = matched_docs[0] if matched_docs else "No context"
                top_id_text = matched_ids[0] if matched_ids else "doc_1"
                rag_box.success(
                    f"🛡️ **Grounded Answer (With ChromaDB Context `[{top_id_text}]`):**\n\n"
                    f"Based on the vector match retrieved from ChromaDB (Distance: {matched_distances[0]:.4f}):\n\n"
                    f"> *\"{top_doc_text}\"* `[{top_id_text}]`\n\n"
                    f"*(Enter your `OPENAI_API_KEY` in the sidebar to run live generation via `{openai_model}`).*"
                )

            st.markdown(
                """
                <div style="background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 10px; border-radius: 4px; font-size: 0.88rem; margin-top: 10px;">
                    <b>✅ With RAG Advantages:</b>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li>Directly grounded in ChromaDB vector facts</li>
                        <li>Verifiable citations linked to specific document IDs</li>
                        <li>Strict mode eliminates hallucination on unknown topics</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -------------------------------------------------------------
    # OTHER MODES (Local Synthesis, Evidence, Strict vs Flexible)
    # -------------------------------------------------------------
    elif result_display_mode == "Synthesized Answer Grounded on Vector Data (Local)":
        if rag_mode == "Strict":
            if not has_strict_pass:
                st.error(
                    f"⛔ **Strict Grounding Rejection:**\n\n"
                    f"No document in ChromaDB met the strict distance threshold (required distance <= {max_distance_cutoff}).\n\n"
                    f"The nearest match had distance `{matched_distances[0]:.4f}`.\n"
                    f"In Strict Mode, the system refuses to speculate or generate an ungrounded answer."
                )
            else:
                top_idx = passing_indices[0]
                top_doc = matched_docs[top_idx]
                top_id = matched_ids[top_idx]
                top_sim = (1.0 - matched_distances[top_idx]) * 100

                st.success(f"🛡️ **Strict Grounded Answer (Confidence: {top_sim:.1f}%):**")
                st.markdown(f"> **Fact from ChromaDB `[{top_id}]`:**\n> {top_doc}")
                st.info("💡 **Strict Mode Guarantee:** The response is derived strictly from the indexed vector evidence above.")

        else:  # Flexible Mode
            top_doc = matched_docs[0]
            top_id = matched_ids[0]
            top_dist = matched_distances[0]
            top_sim = (1.0 - top_dist) * 100

            st.info(f"🌐 **Flexible Synthesized Answer (Confidence: {top_sim:.1f}%):**")
            all_context = " ".join(matched_docs)
            st.markdown(f"Based on the most relevant vector matches in ChromaDB:\n\n**{all_context}**")
            st.caption(f"Primary source: `{top_id}` (Distance: {top_dist:.4f})")

    elif result_display_mode == "Direct Matched Document Evidence":
        st.write("Verbatim matched text documents retrieved from ChromaDB:")
        display_indices = passing_indices if rag_mode == "Strict" else list(range(len(matched_docs)))

        if not display_indices:
            st.warning(f"No documents passed the strict distance cutoff of {max_distance_cutoff}.")
        else:
            for idx in display_indices:
                dist = matched_distances[idx]
                sim = (1.0 - dist) * 100
                st.markdown(f"#### 📄 Document `{matched_ids[idx]}` (Distance: `{dist:.4f}` | Sim: `{sim:.1f}%`)")
                st.code(matched_docs[idx], language="text")
                st.caption(f"Metadata: {matched_metas[idx]}")

    elif result_display_mode == "Compare Side-by-Side (Strict vs Flexible)":
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### 🛡️ Strict Mode")
            if not has_strict_pass:
                st.error(f"⛔ Rejected: Distance {matched_distances[0]:.4f} > {max_distance_cutoff}")
            else:
                st.success(f"✅ Grounded in `{matched_ids[passing_indices[0]]}`:\n\n{matched_docs[passing_indices[0]]}")
        with col_s2:
            st.markdown("#### 🌐 Flexible Mode")
            st.info(f"Context from top matches:\n\n{' '.join(matched_docs)}")

else:
    st.info("Enter a query above or click one of the example questions to see ChromaDB in action.")
