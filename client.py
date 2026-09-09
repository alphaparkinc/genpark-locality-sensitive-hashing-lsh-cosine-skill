import math
import random

class CosineLSH:
    """Random Hyperplane Projection LSH for high-dimensional vector search."""
    def __init__(self, dim: int, num_planes: int = 8, seed: int = 42):
        self.dim = dim
        self.num_planes = num_planes
        random.seed(seed)
        # Generate random projection hyperplanes
        self.hyperplanes = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(num_planes)]
        self.buckets = {}

    def _hash(self, vector: list[float]) -> int:
        hash_val = 0
        for i, plane in enumerate(self.hyperplanes):
            dot = sum(v * p for v, p in zip(vector, plane))
            if dot >= 0:
                hash_val |= (1 << i)
        return hash_val

    def index(self, doc_id: str, vector: list[float]):
        h = self._hash(vector)
        if h not in self.buckets:
            self.buckets[h] = []
        self.buckets[h].append((doc_id, vector))

    def query(self, vector: list[float]) -> list:
        h = self._hash(vector)
        candidates = self.buckets.get(h, [])
        return [{"id": doc_id, "bucket": h} for doc_id, _ in candidates]
