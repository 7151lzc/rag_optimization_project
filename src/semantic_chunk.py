"""
第一步优化：语义切分 + 重叠
功能：将长文档切分成语义完整的段落，保留上下文重叠
"""

import json
import re
from typing import List, Dict, Tuple

def split_by_sentence(text: str) -> List[str]:
    """按句子切分（中文）"""
    # 中文句子分隔符
    sentences = re.split(r'[。！？!?]', text)
    # 过滤空句子
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    return sentences

def semantic_chunk(text: str, 
                   chunk_size: int = 3,      # 每个chunk包含几个句子
                   overlap: int = 1,         # 重叠几个句子
                   min_chunk_size: int = 2) -> List[str]:
    """
    语义切分：按句子边界切分，保留重叠
    
    Args:
        text: 输入文档
        chunk_size: 每个chunk包含的句子数
        overlap: 相邻chunk重叠的句子数
        min_chunk_size: 最小chunk句子数
    
    Returns:
        切分后的chunk列表
    """
    # 1. 按句子切分
    sentences = split_by_sentence(text)
    
    if len(sentences) <= chunk_size:
        return [text] if sentences else []
    
    # 2. 滑动窗口切分
    chunks = []
    step = chunk_size - overlap
    
    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i:i + chunk_size]
        
        # 跳过太短的chunk
        if len(chunk_sentences) < min_chunk_size:
            continue
        
        chunk_text = "。".join(chunk_sentences) + "。"
        chunks.append(chunk_text)
        
        # 如果已经到末尾，退出
        if i + chunk_size >= len(sentences):
            break
    
    return chunks if chunks else [text]

def process_documents(docs: List[str], 
                      chunk_size: int = 3, 
                      overlap: int = 1) -> Tuple[List[str], List[int]]:
    """
    处理所有文档：切分 + 记录原始索引
    
    Args:
        docs: 原始文档列表
        chunk_size: 每个chunk的句子数
        overlap: 重叠句子数
    
    Returns:
        chunks: 切分后的chunk列表
        chunk_to_doc_idx: 每个chunk对应的原始文档索引
    """
    all_chunks = []
    chunk_to_doc_idx = []
    
    for idx, doc in enumerate(docs):
        chunks = semantic_chunk(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        chunk_to_doc_idx.extend([idx] * len(chunks))
    
    print(f"原始文档数: {len(docs)}, 切分后chunk数: {len(all_chunks)}")
    print(f"平均每个文档切分成 {len(all_chunks)/len(docs):.1f} 个chunk")
    
    return all_chunks, chunk_to_doc_idx

def reconstruct_document(chunk_indices: List[int], 
                         chunk_to_doc_idx: List[int]) -> List[int]:
    """
    将检索到的chunk索引还原为原始文档索引
    
    Args:
        chunk_indices: 检索返回的chunk索引列表
        chunk_to_doc_idx: chunk索引 -> 原始文档索引
    
    Returns:
        原始文档索引列表
    """
    return [chunk_to_doc_idx[idx] for idx in chunk_indices]

def compare_chunking(doc_text: str):
    """对比固定切分 vs 语义切分"""
    print("\n" + "=" * 50)
    print("对比切分效果")
    print("=" * 50)
    
    # 固定切分（按字符）
    fixed_chunks = [doc_text[i:i+200] for i in range(0, len(doc_text), 200)]
    
    # 语义切分
    semantic_chunks = semantic_chunk(doc_text, chunk_size=2, overlap=1)
    
    print(f"原文档长度: {len(doc_text)} 字符")
    print(f"固定切分: {len(fixed_chunks)} 个chunk")
    print(f"语义切分: {len(semantic_chunks)} 个chunk")
    
    print("\n语义切分样例:")
    for i, chunk in enumerate(semantic_chunks[:3]):
        print(f"\nChunk {i+1}: {chunk[:100]}...")
    
    return semantic_chunks

# ============ 测试 ============
if __name__ == "__main__":
    # 测试文档
    test_doc = """
    北京故宫是中国明清两代的皇家宫殿，旧称紫禁城。位于北京中轴线的中心。故宫以三大殿为中心，占地面积约72万平方米，建筑面积约15万平方米。有大小宫殿七十多座，房屋九千余间。故宫于明成祖永乐四年开始建设，到永乐十八年建成。故宫是世界上现存规模最大、保存最为完整的木质结构古建筑群之一。1961年被列为第一批全国重点文物保护单位。1987年被列为世界文化遗产。
    """
    
    # 对比切分效果
    chunks = compare_chunking(test_doc)
    
    # 测试批量处理
    test_docs = [test_doc, test_doc]
    all_chunks, mapping = process_documents(test_docs, chunk_size=2, overlap=1)
    
    print(f"\n批量处理: {len(all_chunks)} chunks, 映射长度: {len(mapping)}")