# 测试 LlamaIndex + Qdrant + 局域网 Ollama/TGI 的典型例子
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
from sentence_transformers import SentenceTransformer
import qdrant_client

# 1. Embedding model (CPU)
embed_model = SentenceTransformer("BAAI/bge-m3")

# 2. Connect to Qdrant
client = qdrant_client.QdrantClient(url="http://127.0.0.1:6333")
vector_store = QdrantVectorStore(client=client, collection_name="docs")

storage = StorageContext.from_defaults(vector_store=vector_store)

# 3. Load documents
docs = SimpleDirectoryReader("data/").load_data()

# 4. Create Index
index = VectorStoreIndex.from_documents(
    docs,
    embed_model=embed_model,
    storage_context=storage
)

# 5. Connect to remote Ollama GPU node
llm = Ollama(
    model="qwen2.5:14b",
    base_url="http://192.168.1.88:11434"  # GPU node IP
)

query_engine = index.as_query_engine(llm=llm)

res = query_engine.query("请用中文解释这些文档内容的核心要点")
print(res)
