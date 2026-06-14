"""
完整 RAG 系统（使用问题自带文档）
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from query_expansion import QueryExpander
from llm_api import LLMGenerator

MODEL_PATH = "models/damo/nlp_corom_sentence-embedding_chinese-base"
DATA_FILE = "data/raw/dev.json"


def mean_pooling(model_output, attention_mask):
    """平均池化：将token向量转换为句子向量"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def load_questions_with_docs():
    """加载问题及其自带的文档"""
    questions = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            
            # 提取文档内容
            docs = []
            for doc in item.get('documents', []):
                if isinstance(doc, dict):
                    # 取 paragraphs 或 text
                    if 'paragraphs' in doc:
                        text = ' '.join(doc['paragraphs'])
                    elif 'text' in doc:
                        text = doc['text']
                    else:
                        text = doc.get('title', '')
                else:
                    text = str(doc)
                docs.append(text)
            
            questions.append({
                'id': item['id'],
                'question': item['question'],
                'answer': item['answer'],
                'documents': docs
            })
    
    print(f"加载 {len(questions)} 个问题")
    return questions


class RAGSystem:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self.expander = None
        self.llm = None
        self.questions = None
    
    def init_retriever(self):
        """初始化检索模型"""
        print("初始化检索器...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.expander = QueryExpander(MODEL_PATH)
        print(f"设备: {self.device}")
    
    def init_llm(self):
        """初始化 LLM"""
        print("初始化 LLM...")
        self.llm = LLMGenerator()
    
    def load_data(self):
        """加载数据"""
        print("加载数据...")
        self.questions = load_questions_with_docs()
    
    def build_index(self, docs):
        """兼容旧接口（不再使用）"""
        print("信息: 系统已切换为按问题检索模式，不再构建全局索引。")
        pass
    
    def find_question(self, query):
        """根据问题文本查找对应的数据"""
        # 精确匹配
        for q in self.questions:
            if q['question'] == query:
                return q
        # 模糊匹配（包含关系）
        for q in self.questions:
            if query in q['question'] or q['question'] in query:
                return q
        return None
    
    def answer(self, query):
        """回答问题：先找到问题，再用它的文档检索"""
        # 1. 找到对应的问题数据
        q_data = self.find_question(query)
        
        if not q_data:
            return {
                'query': query,
                'answer': "未找到相关问题",
                'contexts': [],
                'scores': [],
                'doc_ids': []
            }
        
        # 2. 使用该问题的文档作为检索库
        docs = q_data['documents']
        
        if not docs:
            return {
                'query': query,
                'answer': "该问题没有相关文档",
                'contexts': [],
                'scores': [],
                'doc_ids': []
            }
        
        # 3. 向量化文档
        doc_embeddings = []
        for doc in docs:
            encoded = self.tokenizer([doc], padding=True, truncation=True, return_tensors='pt', max_length=512)
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self.model(**encoded)
                emb = mean_pooling(output, encoded['attention_mask'])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                doc_embeddings.append(emb.cpu().numpy())
        doc_embeddings = np.vstack(doc_embeddings)
        
        # 4. Query 扩写
        expanded, _, _ = self.expander.expand_query(query)
        
        # 5. 向量化查询
        encoded = self.tokenizer([expanded], padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = self.model(**encoded)
            query_emb = mean_pooling(output, encoded['attention_mask'])
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            query_emb = query_emb.cpu().numpy()
        
        # 6. 检索
        scores = cosine_similarity(query_emb, doc_embeddings).flatten()
        top_indices = np.argsort(scores)[-5:][::-1]
        
        contexts = [docs[idx] for idx in top_indices]
        doc_scores = scores[top_indices].tolist()
        
        # 7. LLM 生成答案
        answer = self.llm.generate(query, contexts)
        
        return {
            'query': query,
            'answer': answer,
            'contexts': contexts,
            'scores': doc_scores,
            'doc_ids': top_indices.tolist()
        }


def load_data():
    """兼容旧接口：返回问题列表、文档列表、映射"""
    questions = []
    docs = []
    doc_to_qid = []
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            qid = item['id']
            question = item['question']
            answer = item['answer']
            
            questions.append({
                'id': qid,
                'question': question,
                'answer': answer
            })
            
            for doc in item.get('documents', []):
                if isinstance(doc, dict):
                    if 'paragraphs' in doc:
                        doc_text = ' '.join(doc['paragraphs'])
                    elif 'text' in doc:
                        doc_text = doc['text']
                    else:
                        doc_text = doc.get('title', '')
                else:
                    doc_text = str(doc)
                docs.append(doc_text)
                doc_to_qid.append(qid)
    
    return questions, docs, doc_to_qid


def main():
    print("=" * 60)
    print("RAG 系统（使用问题自带文档）")
    print("=" * 60)
    
    rag = RAGSystem()
    rag.init_retriever()
    rag.init_llm()
    rag.load_data()
    
    test_queries = [
        "笔记本电脑有没有必要贴膜",
        "断食减肥法有用吗",
        "刘邦厉害吗"
    ]
    
    for query in test_queries:
        print(f"\n问题: {query}")
        result = rag.answer(query)
        print(f"答案: {result['answer'][:150]}...")
        print(f"检索分数: {result['scores'][0] if result['scores'] else 0:.3f}")


if __name__ == "__main__":
    main()