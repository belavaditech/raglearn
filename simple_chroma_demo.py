"""
Simple ChromaDB Interactive RAG Demo
Based on: https://www.geeksforgeeks.org/nlp/introduction-to-chromadb/

Demonstrates:
1. Creating a ChromaDB client & collection.
2. Entering text documents into the vector database.
3. Querying the collection using natural language.
4. Option to display raw vector query outputs (documents, distances, IDs, metadata).
5. Option to be Flexible vs. Strict (distance cutoff and grounding constraints).
6. Option to display result based on vector data found (direct match vs synthesized answer).
7. OpenAI query comparison: With RAG vs. Without RAG.
"""

import os
import chromadb

def create_chroma_rag():
    # 1. Initialize ChromaDB client (in-memory)
    # Following GeeksforGeeks: client = chromadb.Client()
    client = chromadb.Client()

    # 2. Create or get collection
    # Setting cosine distance: distance = 1 - cosine_similarity (0 is identical, 2 is opposite)
    collection = client.get_or_create_collection(
        name="rag_demo",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

def populate_sample_data(collection):
    """Add initial sample knowledge into ChromaDB collection."""
    sample_docs = [
        "Mercury is the closest planet to the Sun and experiences extreme temperatures from -180°C to 430°C.",
        "Venus has a thick carbon dioxide atmosphere with sulfuric acid clouds, making it the hottest planet at 465°C.",
        "Mars is known as the Red Planet and features Olympus Mons, the largest volcano in the Solar System.",
        "Jupiter is the largest planet, with at least 95 moons including Europa which has a subsurface liquid ocean.",
        "Saturn has a massive ring system composed of ice and rock particles, with 146 confirmed moons including Titan."
    ]
    ids = [f"doc_{i+1}" for i in range(len(sample_docs))]
    metadatas = [{"category": "astronomy", "topic": f"Planet {i+1}"} for i in range(len(sample_docs))]

    collection.add(
        documents=sample_docs,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ Loaded {len(sample_docs)} sample documents into ChromaDB collection '{collection.name}'.")

def run_openai_comparison(query_text: str, context: str, api_key: str = None):
    """Executes query with and without RAG using OpenAI."""
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("\n🔑 No OPENAI_API_KEY set. (Set OPENAI_API_KEY environment variable to test live).")
        print("\n--- [SIMULATED OPENAI COMPARISON] ---")
        print("🤖 [WITHOUT RAG]: Answers using solely pre-training data. If asking about private docs, it cannot know them.")
        print(f"⚡ [WITH RAG]: Answers using retrieved ChromaDB context:\n   \"{context}\"")
        return

    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)

        print("\n🤖 1. Querying OpenAI WITHOUT RAG (Direct base model)...")
        res_no_rag = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer based only on pre-trained knowledge."},
                {"role": "user", "content": query_text}
            ],
            temperature=0.2
        )
        print(f"Answer (Without RAG):\n{res_no_rag.choices[0].message.content}\n")

        print("⚡ 2. Querying OpenAI WITH ChromaDB RAG (Vector context injected)...")
        res_rag = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a factual RAG assistant. Answer strictly based on the provided ChromaDB context and cite document IDs [doc_X]."},
                {"role": "user", "content": f"Context from ChromaDB:\n{context}\n\nQuestion: {query_text}"}
            ],
            temperature=0.2
        )
        print(f"Answer (With RAG):\n{res_rag.choices[0].message.content}\n")

    except Exception as e:
        print(f"⚠️ OpenAI Error: {e}")

def query_rag(collection, query_text: str, n_results: int = 2,
              strict_mode: bool = True, strict_max_distance: float = 0.65,
              show_vector_outputs: bool = True, result_view: str = "synthesized"):
    """
    Executes a vector query against ChromaDB with user-selected options:
    - show_vector_outputs: Display raw vector query matches & distances
    - strict_mode: Strict distance threshold & grounding vs Flexible loose matching
    - result_view: 'synthesized', 'direct', or 'openai_compare'
    """
    print("\n" + "=" * 65)
    print(f"🔍 QUERY: \"{query_text}\"")
    print(f"⚙️  Settings: Mode = [{'STRICT' if strict_mode else 'FLEXIBLE'}] | "
          f"Cutoff Distance <= {strict_max_distance} | Results = {n_results}")
    print("=" * 65)

    # Query ChromaDB collection
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []
    ids = results["ids"][0] if results["ids"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    if not docs:
        print("❌ No matching documents found in vector database.")
        return

    # OPTION 1: Display Vector Query Outputs
    if show_vector_outputs:
        print("\n📊 --- [OPTION 1: VECTOR QUERY OUTPUTS & DISTANCES] ---")
        for i in range(len(docs)):
            dist = distances[i]
            similarity = max(0.0, 1.0 - dist)
            is_pass = dist <= strict_max_distance
            pass_status = "✅ PASSES STRICT" if is_pass else "❌ REJECTED BY STRICT"

            print(f"Rank #{i+1} | ID: {ids[i]} | Distance: {dist:.4f} (Cosine Sim: {similarity:.4f} / {similarity*100:.1f}%)")
            print(f"  • Status  : {pass_status} (Threshold: <= {strict_max_distance})")
            print(f"  • Metadata: {metadatas[i]}")
            print(f"  • Document: \"{docs[i]}\"\n")

    # Filter by strict mode
    valid_pairs = []
    for i in range(len(docs)):
        if not strict_mode or distances[i] <= strict_max_distance:
            valid_pairs.append((docs[i], distances[i], ids[i]))

    # OPTION 2 & 3: Display Result Based on Vector Data Found
    print("🎯 --- [RESULT BASED ON VECTOR DATA FOUND] ---")

    if not valid_pairs:
        print(f"⛔ STRICT MODE REJECTION:")
        print(f"   No document in ChromaDB met the strict distance requirement (distance <= {strict_max_distance}).")
        print(f"   Closest match had distance {distances[0]:.4f}.")
        print("   In Strict Mode, the system refuses to hallucinate without verified vector evidence.")
        return

    context_str = "\n".join([f"[{doc_id}]: {text}" for text, dist, doc_id in valid_pairs])

    if result_view == "openai_compare":
        run_openai_comparison(query_text, context_str)
    elif result_view == "direct":
        print("📋 Showing verbatim matched documents retrieved from ChromaDB:")
        for rank, (doc_text, dist, doc_id) in enumerate(valid_pairs, 1):
            print(f"   [{rank}] ({doc_id} | Distance: {dist:.4f}): {doc_text}")
    else:
        top_doc, top_dist, top_id = valid_pairs[0]
        confidence = round(max(0.0, 1.0 - top_dist) * 100, 1)

        if strict_mode:
            print(f"🛡️  Strict Grounded Answer (Confidence: {confidence}%):")
            print(f"   Based strictly on ChromaDB document [{top_id}]:")
            print(f"   -> {top_doc}")
            print("\n   [Strict Mode Guarantee: Factual answer directly grounded in retrieved vector data.]")
        else:
            print(f"🌐 Flexible Synthesized Answer (Confidence: {confidence}%):")
            combined_context = " ".join([d[0] for d in valid_pairs])
            print(f"   Based on available vector context: {combined_context}")
            print("\n   [Flexible Mode: Broad semantic context integrated from all retrieved neighbors.]")

def interactive_cli():
    print("=" * 65)
    print("      🚀 SIMPLE CHROMADB INTERACTIVE RAG DEMO (GeeksforGeeks)     ")
    print("=" * 65)

    client, collection = create_chroma_rag()
    populate_sample_data(collection)

    strict_mode = True
    show_vector_outputs = True
    result_view = "openai_compare"
    strict_max_dist = 0.65

    while True:
        print("\n" + "-" * 65)
        print("1. 🔍 Query Vector Database")
        print("2. ✍️  Add New Document to ChromaDB")
        print(f"3. 🔄 Toggle Strict / Flexible Mode (Current: {'STRICT' if strict_mode else 'FLEXIBLE'})")
        print(f"4. 🎚️  Adjust Strict Distance Cutoff (Current: {strict_max_dist})")
        print(f"5. 👁️  Toggle Display Vector Outputs (Current: {'ON' if show_vector_outputs else 'OFF'})")
        print(f"6. 📋 Result View Mode (Current: {result_view})")
        print("7. 🚪 Exit")
        print("-" * 65)

        choice = input("Select option (1-7): ").strip()

        if choice == "1":
            q = input("\nEnter query: ").strip()
            if q:
                query_rag(
                    collection=collection,
                    query_text=q,
                    n_results=2,
                    strict_mode=strict_mode,
                    strict_max_distance=strict_max_dist,
                    show_vector_outputs=show_vector_outputs,
                    result_view=result_view
                )
        elif choice == "2":
            new_text = input("\nEnter text document to add into ChromaDB: ").strip()
            if new_text:
                doc_id = f"doc_{collection.count() + 1}"
                collection.add(
                    documents=[new_text],
                    ids=[doc_id],
                    metadatas=[{"category": "user_added"}]
                )
                print(f"✅ Document successfully added with ID: {doc_id} (Total docs: {collection.count()})")
        elif choice == "3":
            strict_mode = not strict_mode
            print(f"✅ Mode changed to: {'STRICT' if strict_mode else 'FLEXIBLE'}")
        elif choice == "4":
            try:
                new_dist = float(input(f"Enter max distance cutoff (0.1 to 1.0, current {strict_max_dist}): "))
                strict_max_dist = new_dist
                print(f"✅ Updated cutoff distance to {strict_max_dist}")
            except ValueError:
                print("Invalid number.")
        elif choice == "5":
            show_vector_outputs = not show_vector_outputs
            print(f"✅ Display Vector Outputs is now: {'ON' if show_vector_outputs else 'OFF'}")
        elif choice == "6":
            print("\nSelect Result View:")
            print("a. OpenAI Compare (With vs Without RAG)")
            print("b. Synthesized Answer (Local)")
            print("c. Direct Vector Match")
            v_choice = input("Select (a/b/c): ").strip().lower()
            if v_choice == "a":
                result_view = "openai_compare"
            elif v_choice == "b":
                result_view = "synthesized"
            elif v_choice == "c":
                result_view = "direct"
            print(f"✅ Result View set to: {result_view}")
        elif choice == "7":
            print("Goodbye!")
            break

if __name__ == "__main__":
    interactive_cli()
