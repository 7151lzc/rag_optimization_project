"""
LambdaMART 排序模型
用于混合检索的统一打分
"""

import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict
import jieba

class LambdaMARTRanker:
    """
    LambdaMART 排序器
    使用 GBDT 实现（LightGBM 的简化版）
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            'bm25_score',           # BM25 原始分数
            'vector_score',         # 向量相似度分数
            'query_len',            # 查询长度
            'doc_len',              # 文档长度
            'word_overlap',         # 词重叠率
            'char_overlap',         # 字符重叠率
        ]
    
    def extract_features(self, query: str, doc: str, bm25_score: float, vector_score: float) -> np.ndarray:
        """
        提取特征向量
        
        Args:
            query: 查询文本
            doc: 文档文本
            bm25_score: BM25 分数
            vector_score: 向量相似度分数
        
        Returns:
            特征向量
        """
        # 1. BM25 分数
        f1 = bm25_score
        
        # 2. 向量相似度
        f2 = vector_score
        
        # 3. 查询长度
        f3 = len(query)
        
        # 4. 文档长度
        f4 = len(doc)
        
        # 5. 词重叠率
        query_words = set(jieba.cut(query))
        doc_words = set(jieba.cut(doc))
        if query_words:
            f5 = len(query_words & doc_words) / len(query_words)
        else:
            f5 = 0.0
        
        # 6. 字符重叠率
        query_chars = set(query)
        doc_chars = set(doc)
        if query_chars:
            f6 = len(query_chars & doc_chars) / len(query_chars)
        else:
            f6 = 0.0
        
        return np.array([f1, f2, f3, f4, f5, f6])
    
    def prepare_training_data(self, queries: List[str], docs: List[str], 
                               bm25_scores: List[List[float]], 
                               vector_scores: List[List[float]],
                               relevance_labels: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练数据
        
        Args:
            queries: 查询列表
            docs: 文档列表
            bm25_scores: 每个查询的 BM25 分数列表
            vector_scores: 每个查询的向量分数列表
            relevance_labels: 相关性标签（0/1/2/3）
        
        Returns:
            X: 特征矩阵
            y: 标签矩阵
        """
        X = []
        y = []
        
        for q_idx, query in enumerate(queries):
            for d_idx, doc in enumerate(docs):
                if d_idx < len(bm25_scores[q_idx]) and d_idx < len(vector_scores[q_idx]):
                    features = self.extract_features(
                        query, doc,
                        bm25_scores[q_idx][d_idx],
                        vector_scores[q_idx][d_idx]
                    )
                    X.append(features)
                    y.append(relevance_labels[q_idx][d_idx])
        
        return np.array(X), np.array(y)
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """
        训练排序模型
        """
        print(f"训练数据: {X.shape[0]} 个样本, {X.shape[1]} 个特征")
        
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 使用 GBDT 回归
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        # 验证
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val)
        
        print(f"训练集 R²: {train_score:.4f}")
        print(f"验证集 R²: {val_score:.4f}")
        
        # 特征重要性
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        print("特征重要性:")
        for name, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {name}: {imp:.4f}")
        
        return self.model
    
    def predict(self, features: np.ndarray) -> float:
        """
        预测排序分数
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用 train()")
        return self.model.predict(features.reshape(1, -1))[0]
    
    def rank(self, query: str, docs: List[str], 
             bm25_scores: List[float], 
             vector_scores: List[float]) -> List[int]:
        """
        对文档进行排序
        
        Returns:
            排序后的文档索引
        """
        scores = []
        for idx, doc in enumerate(docs):
            features = self.extract_features(
                query, doc,
                bm25_scores[idx],
                vector_scores[idx]
            )
            score = self.predict(features)
            scores.append((idx, score))
        
        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores]
    
    def save(self, path: str):
        """保存模型"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"模型已保存到: {path}")
    
    def load(self, path: str):
        """加载模型"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"模型已加载: {path}")


def generate_synthetic_data(num_queries: int = 100, num_docs_per_query: int = 50):
    """
    生成合成训练数据（演示用）
    实际应用中应该使用人工标注数据
    """
    np.random.seed(42)
    
    queries = [f"查询_{i}" for i in range(num_queries)]
    docs = [f"文档_{j}" for j in range(num_docs_per_query)]
    
    bm25_scores = []
    vector_scores = []
    relevance_labels = []
    
    for _ in range(num_queries):
        # 随机生成分数
        bm25 = np.random.rand(num_docs_per_query)
        vector = np.random.rand(num_docs_per_query)
        
        # 相关性标签：基于分数组合
        combined = 0.6 * bm25 + 0.4 * vector
        labels = np.zeros(num_docs_per_query)
        
        # top-5 为正样本
        top_indices = np.argsort(combined)[-5:]
        labels[top_indices] = 1
        
        bm25_scores.append(bm25.tolist())
        vector_scores.append(vector.tolist())
        relevance_labels.append(labels.tolist())
    
    return queries, docs, bm25_scores, vector_scores, relevance_labels


# ============ 测试 ============
if __name__ == "__main__":
    print("=" * 50)
    print("LambdaMART 排序模型测试")
    print("=" * 50)
    
    # 生成合成数据
    print("\n生成合成训练数据...")
    queries, docs, bm25_scores, vector_scores, labels = generate_synthetic_data(100, 50)
    
    # 初始化排序器
    ranker = LambdaMARTRanker()
    
    # 准备训练数据
    print("准备训练数据...")
    X, y = ranker.prepare_training_data(queries, docs, bm25_scores, vector_scores, labels)
    
    # 训练
    print("\n训练模型...")
    ranker.train(X, y)
    
    # 测试排序
    print("\n测试排序...")
    test_query = "测试查询"
    test_docs = [f"文档_{i}" for i in range(10)]
    test_bm25 = np.random.rand(10).tolist()
    test_vector = np.random.rand(10).tolist()
    
    ranked_indices = ranker.rank(test_query, test_docs, test_bm25, test_vector)
    print(f"排序结果（文档索引）: {ranked_indices[:5]}")
    
    # 保存模型
    ranker.save("lambdamart_model.pkl")