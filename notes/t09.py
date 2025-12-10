# 学习 qdrant
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, CollectionStatus
from qdrant_client.models import Filter, FieldCondition, MatchValue

QDRANT_URL = "http://192.168.18.88:6333"
QDRANT_COLLECTION = "harry_test"
QDRANT_VECTOR_SIZE = 384
QDRANT_DISTANCE = "cosine"
QDRANT_TIMEOUT = 30

client = QdrantClient(url=QDRANT_URL)

# 示例 2：创建 Collection（例如 768 维向量）
def t02():
    coll = t07()
    if coll is not None and coll.status == CollectionStatus.GREEN:
        print(f"✅ 使用现有Qdrant集合: {QDRANT_COLLECTION}")
    else:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"✅ 创建Qdrant集合: {QDRANT_COLLECTION}")

# 示例 3：插入向量（带 payload）
def t03():
    vectors = [
        np.random.rand(768).tolist(),
        np.random.rand(768).tolist(),
    ]

    payloads = [
        {"text": "这是第一段文本", "category": "sports"},
        {"text": "这是第二段文本", "category": "technologies"},
    ]

    r = client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            {
                "id": 1,
                "vector": vectors[0],
                "payload": payloads[0],
            },
            {
                "id": 2,
                "vector": vectors[1],
                "payload": payloads[1],
            },
        ],
    )
    print(f"✅ {payloads} 已经插入/更新到 {QDRANT_COLLECTION}! -- {r}")

# 示例 4：向量相似度搜索
def t04():
    query_vector = np.random.rand(768).tolist()

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=3,
    )

    for r in results:
        print(r)
    print(f"✅ 向量相似度搜索已完成!")

# 示例 5：带过滤条件的向量搜索
def t05():
    query_vector = np.random.rand(768).tolist()
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=5,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="category",
                    match=MatchValue(value="sports")
                )
            ]
        ),
    )
    print(f"✅ 带过滤条件的向量搜索: {results}")

# 示例 6：删除向量和集合
def t06():
    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=[1, 2, 3]
    )
    client.delete_collection(QDRANT_COLLECTION)
    print(f"✅ 已经删除向量集合 {QDRANT_COLLECTION}!")

# 示例 7：获取 collection 信息
def t07():
    try:
        info = client.get_collection(QDRANT_COLLECTION)
        print(f"✅ Qdrant 找到向量集合{QDRANT_COLLECTION}, 信息: {info}")
        return info
    except Exception as e:
        print(f"❌ Qdrant 未找到向量集合{QDRANT_COLLECTION}!")

def t08():
    coll = client.get_collections()
    print(coll)

if __name__ == "__main__":
    # t02()
    # t03()
    # t04()
    # t05()
    # t06()
    # t08()
    pass