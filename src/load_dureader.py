"""
任务1：加载 DuReader 数据集（支持 JSONL 格式）
"""

import json
import os

RAW_FILE = "DuReader\\DuReader\\dev.json"      # 输入文件
OUTPUT_FILE = "data/processed/questions.json"  # 输出文件

def main():
    print("=" * 50)
    print("任务1：加载 DuReader 数据集")
    print("=" * 50)
    
    # 检查文件是否存在
    if not os.path.exists(RAW_FILE):
        print(f"❌ 找不到文件: {RAW_FILE}")
        return
    
    # 读取 JSONL 格式（每行一个 JSON 对象）
    print("正在读取文件...")
    data = []
    with open(RAW_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行解析失败: {e}")
    
    print(f"读取完成，共 {len(data)} 条数据")
    
    # 提取问题、答案
    questions = []
    for idx, item in enumerate(data):
        # 获取问题
        question_text = item.get("question", "")
        
        # 获取答案（DuReader 可能有多个字段）
        answer_text = item.get("answer", "")
        if not answer_text and item.get("answers"):
            answers = item["answers"]
            answer_text = answers[0] if answers else ""
        
        questions.append({
            "id": idx,                      # 问题编号
            "question": question_text,      # 问题文本
            "answer": answer_text,          # 标准答案
        })
    
    # 保存处理后的数据
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 保存 {len(questions)} 个问题到: {OUTPUT_FILE}")
    
    # 打印样例
    print("\n数据样例（前3条）:")
    print("-" * 40)
    for q in questions[:3]:
        print(f"Q: {q['question'][:80]}")
        print(f"A: {q['answer'][:80] if q['answer'] else '(无答案)'}")
        print()

if __name__ == "__main__":
    main()