"""
使用阿里云百炼 API 生成答案
"""

import os
from dashscope import Generation

class LLMGenerator:
    def __init__(self):
        """直接从环境变量读取 API Key"""
        self.api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            print("警告: 未设置 DASHSCOPE_API_KEY 环境变量")
    
    def generate(self, query, contexts):
        """基于检索结果生成答案"""
        if not contexts:
            return "未找到相关信息"
        
        # 构建 prompt
        context_str = "\n".join([f"- {c[:300]}" for c in contexts[:3]])
        
        prompt = f"""请基于以下信息回答问题。只使用提供的信息，不要编造。

相关信息：
{context_str}

问题：{query}

答案："""
        
        try:
            response = Generation.call(
                api_key=self.api_key,
                model='qwen-turbo',
                prompt=prompt,
                temperature=0.3,
                max_tokens=200
            )
            
            if response.status_code == 200:
                return response.output.text.strip()
            else:
                return f"基于检索结果：{contexts[0][:100]}..."
                
        except Exception as e:
            print(f"API调用异常: {e}")
            return f"基于检索结果：{contexts[0][:100]}..."


if __name__ == "__main__":
    llm = LLMGenerator()
    answer = llm.generate("北京故宫的面积是多少？", ["故宫建筑面积约15万平方米"])
    print(f"生成答案: {answer}")