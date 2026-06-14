"""
第二步优化：Query 扩写 + 余弦校验
增强版：针对短查询进行更激进的扩写
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 配置
MODEL_PATH = "models/damo/nlp_corom_sentence-embedding_chinese-base"

class QueryExpander:
    """Query 扩写器（带余弦校验）- 增强版"""
    
    def __init__(self, model_path=MODEL_PATH):
        """初始化模型"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 增强的扩写模板（针对短查询）
        self.short_query_templates = [
            # 定义类
            "{} 是什么",
            "什么是 {}",
            "{} 的意思是什么",
            "{} 指的是什么",
            "解释一下 {}",
            "{} 的定义",
            
            # 功能类
            "{} 有什么用",
            "{} 的功能是什么",
            "{} 怎么样",
            "{} 好不好",
            
            # 比较类
            "{} 和什么有关",
            "{} 的特点是什么",
            "{} 有哪些",
            
            # 场景类
            "请介绍一下 {}",
            "{} 的相关信息",
        ]
        
        # 通用模板
        self.general_templates = [
            "{} 是什么",
            "{} 怎么样",
            "什么是 {}",
            "请解释 {}",
        ]
        
        self.history = []
    
    def get_embedding(self, text):
        """获取文本向量"""
        encoded = self.tokenizer([text], padding=True, truncation=True, 
                                  return_tensors='pt', max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            output = self.model(**encoded)
            embedding = output.last_hidden_state[:, 0, :]
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()
    
    def cosine_similarity(self, emb1, emb2):
        """计算余弦相似度"""
        return cosine_similarity(emb1, emb2)[0][0]
    
    def is_short_query(self, query, threshold=10):
        """判断是否为短查询（字数少于阈值）"""
        return len(query) < threshold
    
    def expand_query(self, query, threshold=0.75):
        """
        扩写查询并校验相似度
        针对短查询使用更激进的扩写策略
        """
        # 1. 获取原查询向量
        original_emb = self.get_embedding(query)
        
        # 2. 根据查询长度选择模板
        if self.is_short_query(query):
            templates = self.short_query_templates
            expansion_strategy = "aggressive"
        else:
            templates = self.general_templates
            expansion_strategy = "normal"
        
        # 3. 生成扩写查询
        expanded_queries = []
        for template in templates:
            expanded = template.format(query)
            expanded_queries.append(expanded)
        
        # 4. 去重
        expanded_queries = list(set(expanded_queries))
        
        # 5. 计算每个扩写的相似度
        candidates = []
        for eq in expanded_queries:
            eq_emb = self.get_embedding(eq)
            similarity = self.cosine_similarity(original_emb, eq_emb)
            
            # 动态阈值：短查询放宽要求
            if self.is_short_query(query):
                actual_threshold = threshold - 0.1  # 短查询降低阈值
            else:
                actual_threshold = threshold
            
            candidates.append({
                'query': eq,
                'similarity': similarity,
                'is_valid': similarity >= actual_threshold
            })
        
        # 6. 按相似度排序
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 7. 选择最佳扩写
        best_candidate = candidates[0] if candidates else None
        
        # 8. 决策
        if best_candidate and best_candidate['is_valid']:
            final_query = best_candidate['query']
            similarity_score = best_candidate['similarity']
            used_expansion = True
        else:
            # 短查询即使相似度低也尝试使用最佳候选
            if self.is_short_query(query) and best_candidate:
                final_query = best_candidate['query']
                similarity_score = best_candidate['similarity']
                used_expansion = True
            else:
                final_query = query
                similarity_score = 1.0
                used_expansion = False
        
        # 9. 记录历史
        self.history.append({
            'original': query,
            'expanded': final_query,
            'similarity': similarity_score,
            'used_expansion': used_expansion,
            'expansion_strategy': expansion_strategy,
            'candidates': candidates[:5]
        })
        
        return final_query, similarity_score, used_expansion
    
    def batch_expand(self, queries, threshold=0.75, verbose=True):
        """批量扩写"""
        results = []
        for q in queries:
            expanded, score, used = self.expand_query(q, threshold)
            results.append({
                'original': q,
                'expanded': expanded,
                'similarity': score,
                'used_expansion': used
            })
            if verbose:
                strategy = "短查询" if self.is_short_query(q) else "正常查询"
                status = "✓ 扩写" if used else "✗ 保留"
                print(f"  {status} [{strategy}] {q[:15]}... -> {expanded[:35]}... (相似度: {score:.3f})")
        
        return results
    
    def get_stats(self):
        """获取扩写统计"""
        if not self.history:
            return {}
        
        total = len(self.history)
        used = sum(1 for h in self.history if h['used_expansion'])
        short_queries = sum(1 for h in self.history if h['expansion_strategy'] == 'aggressive')
        short_used = sum(1 for h in self.history if h['expansion_strategy'] == 'aggressive' and h['used_expansion'])
        
        return {
            'total_queries': total,
            'expanded_count': used,
            'expansion_rate': used / total,
            'short_queries': short_queries,
            'short_expansion_rate': short_used / short_queries if short_queries > 0 else 0,
            'avg_similarity': np.mean([h['similarity'] for h in self.history])
        }
    
    def reset_history(self):
        """重置历史记录"""
        self.history = []


# ============ 测试 ============
if __name__ == "__main__":
    print("=" * 50)
    print("Query 扩写测试（增强版）")
    print("=" * 50)
    
    expander = QueryExpander()
    
    test_queries = [
        "发膜",           # 极短查询
        "蜂蜜",           # 极短查询
        "狮虎兽",         # 极短查询
        "刘邦厉害吗",     # 短查询
        "笔记本贴膜",     # 短查询
    ]
    
    print("\n测试查询扩写:")
    results = expander.batch_expand(test_queries, threshold=0.75)
    
    print("\n" + "=" * 50)
    print("扩写统计:")
    stats = expander.get_stats()
    print(f"总查询数: {stats['total_queries']}")
    print(f"扩写数量: {stats['expanded_count']}")
    print(f"扩写率: {stats['expansion_rate']:.1%}")
    print(f"短查询数: {stats['short_queries']}")
    print(f"短查询扩写率: {stats['short_expansion_rate']:.1%}")
    print(f"平均相似度: {stats['avg_similarity']:.3f}")