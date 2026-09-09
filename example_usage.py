from client import CosineLSH

def main():
    print("=== Locality-Sensitive Hashing (LSH) Cosine ===")
    lsh = CosineLSH(dim=4, num_planes=6)

    v1 = [1.0, 2.0, 3.0, 4.0]
    v2 = [1.1, 1.9, 3.2, 4.1] # High cosine similarity with v1
    v3 = [-1.0, -2.0, -3.0, -4.0] # Opposite direction

    lsh.index("doc_similar_1", v1)
    lsh.index("doc_similar_2", v2)
    lsh.index("doc_opposite", v3)

    hits = lsh.query(v1)
    print("Query Hits for v1:", hits)
    hit_ids = [h["id"] for h in hits]
    assert "doc_similar_1" in hit_ids
    assert "doc_similar_2" in hit_ids
    assert "doc_opposite" not in hit_ids

    print("Cosine LSH verified successfully!")

if __name__ == "__main__":
    main()
