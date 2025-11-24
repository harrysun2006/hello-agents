import collections

# 示例语料库
corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens = len(tokens)

# 预先计算 unigram 和 bigram 频数
unigram_counts = collections.Counter(tokens)
bigrams = list(zip(tokens, tokens[1:]))       # 重要：保存为列表，不让迭代器耗尽
bigram_counts = collections.Counter(bigrams)


def unigram_prob(word):
    """计算 P(word)"""
    return unigram_counts[word] / total_tokens if unigram_counts[word] > 0 else 0.0


def bigram_prob(w2, w1):
    """
    计算 P(w2 | w1)
    即 count(w1, w2) / count(w1)
    """
    if unigram_counts[w1] == 0:   # 如果前一个词不存在
        return 0.0
    return bigram_counts[(w1, w2)] / unigram_counts[w1]


def ngram_prob(sentence):
    """
    sentence: 由三个词组成的字符串，例如 "datawhale agent learns"
    返回：P(w1) * P(w2|w1) * P(w3|w2)
    """
    words = sentence.split()
    if len(words) != 3:
        raise ValueError("本示例仅支持 3 个词的句子。")

    w1, w2, w3 = words

    p1 = unigram_prob(w1)
    p2 = bigram_prob(w2, w1)
    p3 = bigram_prob(w3, w2)

    print(f"P({sentence!r}) = P({w1}) * P({w2}|{w1}) * P({w3}|{w2})")
    print(f"             = {p1:.3f} * {p2:.3f} * {p3:.3f}")

    return p1 * p2 * p3


# -----------------------------------------
# 调用方法：计算 3 个句子的概率
# -----------------------------------------

sentences = [
    "datawhale agent learns",
    "datawhale agent works",
    "datawhale agent evolves"
]

for s in sentences:
    p = ngram_prob(s)
    print(f"结果: P('{s}') = {p:.6f}\n")