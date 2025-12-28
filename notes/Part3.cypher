// 09_Memory_Types_Deep_Dive.py
// 保存 semantic memory 时, _extract_entities 和 _extract_relations ==(spaCy)==> entities/relations ==> neo4j
use neo4j match(n) where n.source_text='梯度下降是一种优化算法，通过迭代更新参数来最小化损失函数' return n;
use neo4j match(n) where n.name in ['梯度','是','算法','更新','参数'] return n;
// 搜索一句话解析出来的 entities (NER + RE)
use neo4j match(n) where n.source_text='梯度下降是一种优化算法，通过迭代更新参数来最小化损失函数' or n.name in ['梯度','是','算法','更新','参数'] return n;

// perceptual memory 不保存 neo4j!
use neo4j match(n) where n.source_text='这是一段优美的诗歌：春江潮水连海平，海上明月共潮生' return n;