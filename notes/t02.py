# 测试embedding (gemini, sentence-transformers, TF-IDF etc.), 向量相似度等
import google.generativeai as genai
import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba
import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
# warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")
# warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources.*")

load_dotenv(override=True)
TEST_TEXT = "你好，我正在测试 Gemini embedding。"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "dummy"))

# embedding 一条文本 (Gemini)
# models/embedding-001: vec[768], [0.05167833, -0.0066265333, -0.06856554, -0.0025570767, 0.051431578]
# models/text-embedding-004: vec[768], [0.027582636, 0.048994534, -0.05323547, 0.02172168, 0.04741841]
# models/gemini-embedding-001: vec[3072], [-0.021769231, 0.0135185225, 0.009128436, -0.09916112, 0.0037866938]
def t01():
    model = "models/embedding-001"          # 768d
    # model = "models/text-embedding-004"     # 768d
    # model = "models/gemini-embedding-001"   # 3072d
    response = genai.embed_content(
        model=model,
        content=TEST_TEXT,
        task_type="retrieval_document"  # 常用场景
    )
    vec = response["embedding"]
    print("Vector dim:", len(vec))
    print("前 5 维向量:", vec[:5])

# embedding 多条文本 (Gemini)
# models/embedding-001
# vec[768]: [0.045319725, 0.0004357293, -0.05145393, -0.013893462, 0.04570605]
# vec[768]: [0.04470099, -0.016133074, -0.070899494, -0.0061208634, 0.060846467]
# vec[768]: [0.052355852, -0.012403951, -0.04703385, -0.008424252, 0.049596984]
def t02():
    model = "models/embedding-001"
    texts = [
        "今天是星期三，我在测试 embedding。",
        "Neo4j 是一个图数据库。",
        "Google Gemini 的 embedding 效果很好。"
    ]
    response = genai.embed_content(
        model=model,
        content=texts,
        task_type="retrieval_document"
    )

    vecs = response["embedding"]
    print(len(vecs), "条向量")
    for vec in vecs:
        print(f"vec[{len(vec)}]: {vec[:5]}")

# embedding 一条文本 (sentence-transformers)
# sentence-transformers 3.4.1 + transformers 4.39.3
# BAAI/bge-large-zh-v1.5: vec[1024], [ 0.02104814 -0.03504314  0.0170575   0.01183293 -0.05737676]
# all-MiniLM-L6-v2: vec[384], [-0.01849279  0.08543978  0.08318555  0.03782603 -0.01265049]
def t03():
    # model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode(TEST_TEXT)
    print(f"vec[{vec.shape}]: {vec[:5]}")

def jieba_tokenizer(text):
    return list(jieba.cut(text))

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product

def similarity(v1, v2):
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    return np.dot(v1_norm, v2_norm)

# TF-IDF
def t05():
    docs = [
        "我喜欢机器学习",
        "我喜欢深度学习"
    ]

    vectorizer = TfidfVectorizer(tokenizer=jieba_tokenizer)
    # vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
    X = vectorizer.fit_transform(docs)

    sim = cosine_similarity(X[0], X[1])
    print("相似度 =", sim[0][0])    
    print("特征：", vectorizer.get_feature_names_out())
    print("文档向量：", X.toarray())

    # 0, 0 TF-IDF 只做分词，没有语义理解
    query = ["什么是人工智能的分支？"]
    # 0.63, 0.63
    # query = ["我不喜欢打游戏"]
    query_vec = vectorizer.transform(query)
    scores = cosine_similarity(query_vec, X)[0]
    for idx in scores.argsort()[::-1]:
        print(scores[idx], docs[idx])

# 比较相似度 (bge vs. TF-IDF)
def t06():
    docs = [
        "我喜欢机器学习",
        "我喜欢深度学习"
    ]

    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    doc_emb = model.encode(docs, convert_to_tensor=True)
    # [0.3509, 0.3360]
    query = "什么是人工智能的分支？"
    # [0.4284, 0.4512]
    # query = "我不喜欢打游戏"
    query_emb = model.encode(query, convert_to_tensor=True)

    scores = util.cos_sim(query_emb, doc_emb)
    print(scores)

def gemini(model, text):
    response = genai.embed_content(
        model=model,
        content=text,
        task_type="classification"
    )
    return np.array(response["embedding"], dtype=np.float32)

# king - man + woman ~= queen:
# all-MiniLM-L6-v2: [ 0.00448471  0.01810496 -0.06230409] .. [ 0.03548696 -0.06560463 -0.00993497] ~= 0.579
# BAAI/bge-large-zh-v1.5: [-0.02002675  0.01483777 -0.04235058] .. [ 0.00810071  0.02194249 -0.0383593 ] ~= 0.694
# gemini/embedding-001: [ 0.04843168 -0.0254853   0.01375712] .. [ 0.02494862 -0.03891581  0.00158959] ~= 0.834
# gemini/text-embedding-004: [-0.05954755 -0.02513115  0.01067936] .. [-0.05373279  0.00757144 -0.02361631] ~= 0.738
# gemini/gemini-embedding-001: [-0.03771621  0.01532772 -0.00618936] .. [-0.0320245   0.02014215  0.00023862] ~= 0.891
# 
# gemini/gemini-embedding-001 (semantic_similarity):
# king - man + woman ~= queen: [-0.03771621  0.01532772 -0.00618936] .. [-0.0320245   0.02014215  0.00023862] ~= 0.891
# king - male + female ~= queen: [-0.03185991  0.02244489  0.00687677] .. [-0.0320245   0.02014215  0.00023862] ~= 0.906
# 国王 - 男人 + 女人 ~= 王后: [-0.02611095  0.02173588 -0.00038134] .. [-0.02632829  0.01777293 -0.00090081] ~= 0.889
# uncle - man + woman ~= aunt: [-0.02426458  0.01641373 -0.02043177] .. [-0.01961219  0.01472826 -0.0015362 ] ~= 0.900
# uncle - male + female ~= aunte: [-0.01840827  0.02353091 -0.00736564] .. [-0.01993692  0.01404219  0.00289279] ~= 0.883
# 叔叔 - 男 + 女 ~= 婶婶: [-0.01745144  0.03033821 -0.00191252] .. [-0.01809324  0.02228117 -0.00737954] ~= 0.898
# 叔叔 - 男 + 女 ~= 姑姑: [-0.01745144  0.03033821 -0.00191252] .. [-0.03012826  0.01696179 -0.00584408] ~= 0.905
# 叔叔 - 男 + 女 ~= 舅妈: [-0.01745144  0.03033821 -0.00191252] .. [-0.01283066  0.01911757 -0.00039904] ~= 0.875
# 
# gemini/gemini-embedding-001 (classification):
# king - man + woman ~= queen: 0.832
# king - male + female ~= queen: 0.885
# 国王 - 男人 + 女人 ~= 王后: 0.859
# uncle - man + woman ~= aunt: 0.831
# uncle - male + female ~= aunte: 0.818
# 叔叔 - 男 + 女 ~= 婶婶: 0.853
# 叔叔 - 男 + 女 ~= 姑姑: 0.844
# 叔叔 - 男 + 女 ~= 舅妈: 0.812
def t07():
    # king - man + woman
    cbs0 = {
        "all-MiniLM-L6-v2": lambda t: SentenceTransformer("all-MiniLM-L6-v2").encode(t),
        "BAAI/bge-large-zh-v1.5": lambda t: SentenceTransformer("BAAI/bge-large-zh-v1.5").encode(t),
        "gemini/embedding-001": lambda t: gemini("models/embedding-001", t),
        "gemini/text-embedding-004": lambda t: gemini("models/text-embedding-004", t),
        "gemini/gemini-embedding-001": lambda t: gemini("models/gemini-embedding-001", t),
    }
    cbs = {"gemini/gemini-embedding-001": lambda t: gemini("models/gemini-embedding-001", t),}
    words_list = [
        ["king", "man", "woman", "queen"],
        ["king", "male", "female", "queen"],
        ["国王", "男人", "女人", "王后"],
        ["uncle", "man", "woman", "aunt"],
        ["uncle", "male", "female", "aunte"],
        ["叔叔", "男", "女", "婶婶"],
        ["叔叔", "男", "女", "姑姑"],
        ["叔叔", "男", "女", "舅妈"],
    ]
    
    for words in words_list:
        [w1, w2, w3, w4] = words
        print(f"{w1} - {w2} + {w3} ~= {w4}:")
        for key in cbs:
            cb = cbs[key]
            v1, v2, v3, v4 = cb(w1), cb(w2), cb(w3), cb(w4)
            vr = v1 - v2 + v3
            sim = cosine_similarity(vr, v4)
            # sim = similarity(vr, v4)
            print(f"{key}: {vr[:3]} .. {v4[:3]} ~= {sim:.3f}")

# semantic_similarity: cosine similarity1 = 1.0000001192092896, cosine similarity2 = 1.0
# retrieval_document: cosine similarity1 = 1.0000001192092896, cosine similarity2 = 1.0
# classification: cosine similarity1 = 1.0, cosine similarity2 = 0.9999999403953552
# clustering: cosine similarity1 = 1.0, cosine similarity2 = 1.0
# question_answering: cosine similarity1 = 1.0, cosine similarity2 = 1.0
def t08():
    v1 = gemini("models/text-embedding-004", "什么是人工智能？")
    v2 = gemini("models/text-embedding-004", "人工智能的分支有哪些？")

    sim1 = cosine_similarity(v1, v2)
    sim2 = similarity(v1, v2)
    print(f"cosine similarity1 = {sim1}, cosine similarity2 = {sim2}")
    # print(f"cosine similarity1 = {sim1:.3f}, cosine similarity2 = {sim2:.3f}")

if __name__ == "__main__":
    # t01()
    # t02()
    # t03()
    # t05()
    # t06()
    t07()
    # t08()
    pass
