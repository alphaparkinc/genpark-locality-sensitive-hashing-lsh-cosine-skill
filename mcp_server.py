import sys
import json
from client import CosineLSH

def handle_request(req):
    method = req.get("method")
    params = req.get("params", {})
    if method == "lsh_query":
        dim = params.get("dim", 3)
        lsh = CosineLSH(dim)
        for d in params.get("documents", []):
            lsh.index(d["id"], d["vector"])
        return {"hits": lsh.query(params.get("query_vector", [1.0] * dim))}
    return {"error": "Unknown method"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_request(req)
        print(json.dumps(res))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
