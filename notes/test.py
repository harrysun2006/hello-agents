import os
import shutil

from pathlib import Path
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv(override=True)

# clean up temp folders/files, sqlite, neo4j entities & qdrant collections
# to make sure each sample running from a clean environment
def clean_up():
    # delete temp folders & sqlite db file
    root = Path("../code")
    folders = [
        "chapter6/CAMEL/context_files"
        "chapter8/advanced_search_kb",
        "chapter8/agent_integration_kb",
        "chapter8/demo_rag_kb",
        "chapter8/memory_data",
        "chapter8/qa_demo_kb"
    ]
    for folder in folders: 
        sub = (root / Path(folder)).resolve()
        shutil.rmtree(sub, ignore_errors=True)
        print(f"✅ 已经删除目录 {sub}!")
    
    # remove all Entity objects in neo4j
    uri = os.getenv("NEO4J_BOLT", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    db = os.getenv("NEO4J_DATABASE", "neo4j")
    driver = GraphDatabase.driver(uri, auth=(username, password))
    cypher = "MATCH (n:Entity) DETACH DELETE n"
    with driver.session(database=db) as session:
        session.run(cypher)
    driver.close()
    
    # delete collections created previously
    colls = [
        "ch8ex04_01", "ch8ex05_01", "ch8ex08_01",
        "hello_agents_vectors",
        "hello_agents_vectors_perceptual_audio",
        "hello_agents_vectors_perceptual_image",
        "hello_agents_vectors_perceptual_text",
        "rag_knowledge_base",
    ]
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url)
    for coll in colls: 
        client.delete_collection(coll)
        print(f"✅ 已经删除向量集合 {coll}!")

if __name__ == "__main__":
    clean_up()
