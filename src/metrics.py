"""
RAG 评估指标：召回率 + 忠实度 + 准确率
优化版：忠实度使用多种策略计算
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
import re
import jieba

# ============ 基础检索指标 ============

def compute_recall_at_k(retrieved_docs: List[int], 
                         relevant_docs: List[int], 
                         k: int = 5) -> float:
    """
    计算召回率：检索到的相关文档比例
    """
    if not relevant_docs:
        return 1.0
    
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    hit_count = len(retrieved_set & relevant_set)
    return hit_count / len(relevant_docs)


def compute_precision_at_k(retrieved_docs: List[int], 
                            relevant_docs: List[int], 
                            k: int = 5) -> float:
    """
    计算精确率：检索结果中相关文档的比例
    """
    if k == 0:
        return 0.0
    
    retrieved_set = set(retrieved_docs[:k])
    relevant_set = set(relevant_docs)
    
    hit_count = len(retrieved_set & relevant_set)
    return hit_count / k


def compute_ndcg_at_k(retrieved_docs: List[int], 
                       relevant_docs: List[int], 
                       k: int = 5) -> float:
    """
    计算 NDCG@k：考虑排序位置的评估指标
    """
    relevant_set = set(relevant_docs)
    
    # 计算 DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    
    # 计算 IDCG（理想情况）
    ideal_docs = relevant_docs[:k]
    idcg = 0.0
    for i in range(min(len(ideal_docs), k)):
        idcg += 1.0 / np.log2(i + 2)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============ 忠实度计算（优化版） ============

def compute_faithfulness_v2(generated_answer: str, 
                            retrieved_contexts: List[str],
                            method: str = "hybrid") -> float:
    """
    计算忠实度（优化版）：生成答案是否被检索内容支持
    
    Args:
        generated_answer: 模型生成的答案
        retrieved_contexts: 检索返回的文档内容列表
        method: 计算方法
            - "contains": 直接包含检查
            - "keyword": 关键词匹配
            - "sentence": 句子级匹配
            - "hybrid": 混合策略（推荐）
    
    Returns:
        忠实度 (0-1)
    """
    if not generated_answer or not retrieved_contexts:
        return 0.0
    
    # 合并所有检索内容
    all_context = " ".join(retrieved_contexts)
    
    if method == "contains":
        # 方法1：直接包含检查
        if generated_answer in all_context:
            return 1.0
        short_answer = generated_answer[:50]
        return 1.0 if short_answer in all_context else 0.0
    
    elif method == "keyword":
        # 方法2：关键词匹配
        stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '会', '能', '对', '等', '与', '及', '或', '并', '而', '且', '还', '这', '那'}
        
        answer_words = [w for w in jieba.cut(generated_answer) if w not in stopwords and len(w) > 1]
        context_words = set([w for w in jieba.cut(all_context) if w not in stopwords and len(w) > 1])
        
        if not answer_words:
            return 0.0
        
        matched = sum(1 for w in answer_words if w in context_words)
        return matched / len(answer_words)
    
    elif method == "sentence":
        # 方法3：分句检查
        sentences = re.split(r'[。！？!?]', generated_answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 3]
        
        if not sentences:
            return 0.0
        
        supported = 0
        for sent in sentences:
            if sent in all_context:
                supported += 1
            else:
                key_phrase = sent[:20]
                if key_phrase in all_context:
                    supported += 0.8
                else:
                    sent_words = set(jieba.cut(sent))
                    context_words = set(jieba.cut(all_context))
                    overlap = len(sent_words & context_words) / len(sent_words) if sent_words else 0
                    if overlap > 0.5:
                        supported += 0.6
        
        return supported / len(sentences)
    
    elif method == "hybrid":
        # 方法4：混合策略（推荐）
        scores = []
        
        # 子策略1：直接包含（权重0.4）
        if generated_answer in all_context:
            scores.append(1.0)
        else:
            short_answer = generated_answer[:50]
            scores.append(1.0 if short_answer in all_context else 0.0)
        
        # 子策略2：关键词匹配（权重0.3）
        stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '会', '能', '对', '等', '与', '及', '或', '并', '而', '且', '还', '这', '那'}
        
        answer_words = [w for w in jieba.cut(generated_answer) if w not in stopwords and len(w) > 1]
        context_words = set([w for w in jieba.cut(all_context) if w not in stopwords and len(w) > 1])
        
        if answer_words:
            matched = sum(1 for w in answer_words if w in context_words)
            keyword_score = matched / len(answer_words)
        else:
            keyword_score = 0.0
        scores.append(keyword_score)
        
        # 子策略3：字符重叠率（权重0.3）
        answer_chars = set(generated_answer)
        context_chars = set(all_context)
        if answer_chars:
            char_overlap = len(answer_chars & context_chars) / len(answer_chars)
        else:
            char_overlap = 0.0
        scores.append(char_overlap)
        
        # 加权平均
        weights = [0.4, 0.3, 0.3]
        final_score = sum(s * w for s, w in zip(scores, weights))
        
        return min(1.0, max(0.0, final_score))
    
    else:
        return 0.0


# ============ RAGEvaluator 类 ============

class RAGEvaluator:
    """RAG 系统评估器（优化版）"""
    
    def __init__(self, faithfulness_method: str = "hybrid"):
        self.results = []
        self.faithfulness_method = faithfulness_method
    
    def add_result(self, 
                   query: str,
                   predicted_answer: str,
                   true_answer: str,
                   retrieved_docs: List[int],
                   relevant_docs: List[int],
                   retrieved_contexts: List[str]):
        """
        添加一个查询的结果
        """
        # 计算各项指标
        recall = compute_recall_at_k(retrieved_docs, relevant_docs, k=5)
        precision = compute_precision_at_k(retrieved_docs, relevant_docs, k=5)
        ndcg = compute_ndcg_at_k(retrieved_docs, relevant_docs, k=5)
        
        # 使用优化后的忠实度计算
        faithfulness = compute_faithfulness_v2(predicted_answer, retrieved_contexts, method=self.faithfulness_method)
        
        # 准确率
        exact_match = (predicted_answer.strip() == true_answer.strip())
        partial_match = self._compute_partial_match(predicted_answer, true_answer)
        
        self.results.append({
            'query': query,
            'predicted_answer': predicted_answer,
            'true_answer': true_answer,
            'recall@5': recall,
            'precision@5': precision,
            'ndcg@5': ndcg,
            'faithfulness': faithfulness,
            'exact_match': exact_match,
            'partial_match': partial_match
        })
    
    def _compute_partial_match(self, predicted: str, true: str, threshold: float = 0.5) -> bool:
        """计算部分匹配"""
        pred_keywords = set(jieba.cut(predicted))
        true_keywords = set(jieba.cut(true))
        
        if not true_keywords:
            return True
        
        overlap = len(pred_keywords & true_keywords)
        score = overlap / len(true_keywords)
        return score >= threshold
    
    def get_summary(self) -> Dict[str, float]:
        """获取汇总指标"""
        if not self.results:
            return {}
        
        metrics = ['recall@5', 'precision@5', 'ndcg@5', 'faithfulness']
        
        summary = {}
        for metric in metrics:
            values = [r[metric] for r in self.results]
            summary[f'avg_{metric}'] = np.mean(values)
            summary[f'std_{metric}'] = np.std(values)
        
        summary['exact_match_rate'] = np.mean([r['exact_match'] for r in self.results])
        summary['partial_match_rate'] = np.mean([r['partial_match'] for r in self.results])
        
        return summary
    
    def get_bottleneck(self) -> str:
        """诊断瓶颈"""
        if not self.results:
            return "无数据"
        
        summary = self.get_summary()
        recall = summary.get('avg_recall@5', 0)
        faithfulness = summary.get('avg_faithfulness', 0)
        
        if recall < 0.6:
            return "检索环节问题：召回率低，需要优化检索（增加文档、改进排序）"
        elif faithfulness < 0.6:
            return "生成环节问题：忠实度低，需要优化生成或压缩上下文"
        else:
            return "系统平衡，可继续优化准确率"
    
    def print_report(self):
        """打印评估报告"""
        summary = self.get_summary()
        bottleneck = self.get_bottleneck()
        
        print("\n" + "=" * 60)
        print("RAG 系统评估报告")
        print("=" * 60)
        print(f"总查询数: {len(self.results)}")
        print(f"忠实度计算方法: {self.faithfulness_method}")
        print()
        print("检索指标:")
        print(f"  平均召回率@5:   {summary.get('avg_recall@5', 0):.2%}")
        print(f"  平均精确率@5:   {summary.get('avg_precision@5', 0):.2%}")
        print(f"  平均 NDCG@5:    {summary.get('avg_ndcg@5', 0):.2%}")
        print()
        print("生成指标:")
        print(f"  平均忠实度:     {summary.get('avg_faithfulness', 0):.2%}")
        print(f"  完全匹配率:     {summary.get('exact_match_rate', 0):.2%}")
        print(f"  部分匹配率:     {summary.get('partial_match_rate', 0):.2%}")
        print()
        print("瓶颈诊断:")
        print(f"  {bottleneck}")
        print("=" * 60)
    
    def export_to_json(self, filepath: str):
        """导出结果"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'results': self.results,
                'summary': self.get_summary()
            }, f, indent=2, ensure_ascii=False)
        print(f"结果已导出到: {filepath}")


# ============ 测试 ============
if __name__ == "__main__":
    print("=" * 50)
    print("忠实度计算测试（优化版）")
    print("=" * 50)
    
    # 测试数据
    answer = "北京故宫面积约15万平方米，建于明朝永乐年间。"
    contexts = [
        "北京故宫是中国明清两代的皇家宫殿，位于北京中轴线中心。",
        "故宫占地面积约72万平方米，建筑面积约15万平方米。",
        "故宫于明成祖永乐四年开始建设，到永乐十八年建成。"
    ]
    
    print("\n测试答案:")
    print(f"答案: {answer}")
    print(f"上下文: {contexts[0][:50]}...")
    
    # 测试不同方法
    methods = ["contains", "keyword", "sentence", "hybrid"]
    for method in methods:
        score = compute_faithfulness_v2(answer, contexts, method=method)
        print(f"\n{method} 方法: {score:.2%}")
    
    # 测试评估器
    print("\n" + "=" * 50)
    print("测试评估器")
    print("=" * 50)
    
    evaluator = RAGEvaluator(faithfulness_method="hybrid")
    
    evaluator.add_result(
        query="北京故宫面积多少？",
        predicted_answer="15万平方米",
        true_answer="约15万平方米",
        retrieved_docs=[0, 1, 2],
        relevant_docs=[0, 5, 6],
        retrieved_contexts=contexts
    )
    
    evaluator.print_report()