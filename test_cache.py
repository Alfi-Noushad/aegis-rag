import time
from cache import semanticCache

cache = semanticCache()

#first sample query setting in the cache for testing
q1 = "How many paid leave days do employees get?"
a1 = "Employees receive 20 days of paid annual vacation per year."

print("Ingesting the first query")
cache.set(q1,a1)
print(f"Stored: '{q1}' \n")

#Second query with diff phrasing (cache hit expected over here lets test it here)
q2 = "How many days of paid vacation do employees receive?"

start_time = time.time()
hit, response, score = cache.lookup(q2)
elapsed_ms = (time.time() - start_time) * 1000

print("Testing Semantic similar query...")
print(f"Query: '{q2}'")
print(f"Similarity score: {score:.4f}")
print(f"returned response: '{response}'")
print(f"Latency: {elapsed_ms:.2f} ms")