import chromadb

def test_chroma():
    client = chromadb.Client()
    collection = client.create_collection("test_col", metadata={"hnsw:space": "cosine"})

    # 1. Add data
    collection.add(
        documents=[
            "Olympus Mons is the largest volcano on Mars.",
            "Jupiter has 95 known moons including Europa."
        ],
        ids=["mars_doc", "jupiter_doc"],
        metadatas=[{"category": "mars"}, {"category": "jupiter"}]
    )
    print("Documents added. Count:", collection.count())

    # 2. Query data
    query = "What is the biggest volcano on Mars?"
    res = collection.query(
        query_texts=[query],
        n_results=2,
        include=["documents", "metadatas", "distances"]
    )
    print("\nQuery:", query)
    print("Result IDs:", res["ids"][0])
    print("Result Distances:", res["distances"][0])
    print("Top matched doc:", res["documents"][0][0])

    top_dist = res["distances"][0][0]
    # Check strict threshold logic
    strict_cutoff = 0.65
    is_pass = top_dist <= strict_cutoff
    print(f"Strict Check (dist {top_dist:.4f} <= {strict_cutoff}): {is_pass}")
    assert is_pass, "Expected top result to pass strict cutoff"
    assert "Olympus Mons" in res["documents"][0][0], "Expected Olympus Mons to be top match"
    print("\n✅ All ChromaDB tests passed successfully!")

if __name__ == "__main__":
    test_chroma()

