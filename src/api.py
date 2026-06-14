"""
RAG 系统后端 API
使用 FastAPI 封装
"""

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_complete import RAGSystem

# ============ 全局变量 ============
rag = None


# ============ 生命周期管理 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动和关闭时的处理"""
    global rag
    print("正在初始化 RAG 系统...")
    rag = RAGSystem()
    rag.init_retriever()
    rag.init_llm()
    rag.load_data()
    print("RAG 系统初始化完成！")
    yield
    print("正在关闭 RAG 系统...")


# ============ 创建 FastAPI 应用 ============
app = FastAPI(
    title="RAG 问答系统 API",
    description="基于检索增强生成的问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ 请求/响应模型 ============
class QueryRequest(BaseModel):
    """查询请求"""
    query: str
    top_k: Optional[int] = 5


class AnswerResponse(BaseModel):
    """答案响应"""
    success: bool
    query: str
    answer: str
    contexts: List[str]
    scores: List[float]
    doc_ids: List[int]


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    chunks_count: int


# ============ API 端点 ============
@app.get("/", response_model=HealthResponse)
async def root():
    """根路径，返回服务状态"""
    return {
        "status": "running",
        "version": "1.0.0",
        "chunks_count": len(rag.questions) if rag and rag.questions else 0
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    if rag is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    return {
        "status": "healthy",
        "version": "1.0.0",
        "chunks_count": len(rag.questions) if rag.questions else 0
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QueryRequest):
    """问答接口"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    if rag is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        result = rag.answer(request.query)
        
        return {
            "success": True,
            "query": result['query'],
            "answer": result['answer'],
            "contexts": result.get('contexts', []),
            "scores": result.get('scores', []),
            "doc_ids": result.get('doc_ids', [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/search")
async def search_only(request: QueryRequest):
    """仅检索接口（不生成答案）"""
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="问题不能为空")
    
    if rag is None:
        raise HTTPException(status_code=503, detail="系统未就绪")
    
    try:
        # 先找到问题
        q_data = rag.find_question(request.query)
        if not q_data:
            return {
                "success": False,
                "query": request.query,
                "message": "未找到相关问题",
                "contexts": [],
                "doc_ids": [],
                "scores": []
            }
        
        docs = q_data['documents']
        
        # 向量化文档
        doc_embeddings = []
        for doc in docs:
            encoded = rag.tokenizer([doc], padding=True, truncation=True, return_tensors='pt', max_length=512)
            encoded = {k: v.to(rag.device) for k, v in encoded.items()}
            with torch.no_grad():
                output = rag.model(**encoded)
                emb = mean_pooling(output, encoded['attention_mask'])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                doc_embeddings.append(emb.cpu().numpy())
        doc_embeddings = np.vstack(doc_embeddings)
        
        # 向量化查询
        encoded = rag.tokenizer([request.query], padding=True, truncation=True, return_tensors='pt', max_length=512)
        encoded = {k: v.to(rag.device) for k, v in encoded.items()}
        with torch.no_grad():
            output = rag.model(**encoded)
            query_emb = mean_pooling(output, encoded['attention_mask'])
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            query_emb = query_emb.cpu().numpy()
        
        # 检索
        scores = cosine_similarity(query_emb, doc_embeddings).flatten()
        top_indices = np.argsort(scores)[-request.top_k:][::-1]
        
        contexts = [docs[idx] for idx in top_indices]
        doc_ids = top_indices.tolist()
        
        return {
            "success": True,
            "query": request.query,
            "contexts": contexts,
            "doc_ids": doc_ids,
            "scores": scores[top_indices].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")


# ============ 启动服务 ============
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )