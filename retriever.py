from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi


class hybridRetriever:
    def __init__(self,collection_name="enterprise_docs"):
        #load the required model for embedding..
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        #Setup Chromadb vector client
        #create instance of chromadb and create a collection[table or index]
        self.client = chromadb.Client()
        self.collection  = client.get_or_create_collection(name=collection_name)


        #attributes for BM25 sparse search
        self.documents = []
        self.bm25 = None

    def ingest_documents(self):
        pass

    def dense_search(self):
        pass
    
    def sparse_search(self):
        pass

    
