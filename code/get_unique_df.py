import os
import json
import time
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(".env")
# ——— 설정 ———
API_KEY       = os.getenv("API_KEY")
MODEL_NAME    = "gpt-4o-mini"
FINAL_QA_PATH = "./data/final_qa.jsonl"
METADB_PATH   = "./data/MetaDB_with_date_id.jsonl"
OUTPUT_CSV    = "./data/gpt4o_unique_df.csv"
API_DELAY     = 0.1  # seconds
# —————————

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=API_KEY)

def load_metadb(path: str) -> List[Dict]:
    """MetaDB 청크 JSONL 로드"""
    chunks = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def get_paper_content_from_paper_id(paper_id: str, metadb_chunks: List[Dict]) -> Dict[str,str]:
    """paper_id에 해당하는 청크들을 모아 title, text(최대 5000자)를 반환"""
    title = ""
    combined = []
    for c in metadb_chunks:
        if c.get("arxiv_id","") == paper_id:
            if not title and c.get("title"):
                title = c["title"]
            if c.get("text"):
                combined.append(c["text"])
    text = " ".join(combined)
    if len(text) > 5000:
        text = text[:5000] + "..."
    return {"title": title, "text": text}

def create_question_generation_prompt(paper_content: Dict[str,str], answer: str) -> str:
    return f"""
Below is information about a research paper and an answer. Please generate a natural question that fits this answer.

Paper Title: {paper_content['title']}

Paper Content (partial): {paper_content['text']}

Answer: {answer}

Requirements:
1. Create a question that matches well with the given answer
2. Write the question as if a real human is reading the paper and asking about it
3. Do not use demonstrative expressions like "this paper" or "the paper"
4. Write concisely in one sentence
5. Write in English

Question:"""

def generate_question_with_openai(paper_content: Dict[str,str], answer: str) -> str:
    """OpenAI API 호출로 question 생성"""
    prompt = create_question_generation_prompt(paper_content, answer)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system",  "content": "You are an expert at generating academic paper questions."},
            {"role": "user",    "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )
    q = resp.choices[0].message.content.strip()
    # 레이블 앞부분 제거
    if q.startswith("Question:"):
        q = q[len("Question:"):].strip()
    if q.startswith("질문:"):
        q = q[len("질문:"):].strip()
    return q

def main():
    # 1) final_qa.jsonl 로드
    records = []
    with open(FINAL_QA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            records.append({
                "paper_id": obj.get("paper_id",""),
                "question": obj.get("question",""),
                "answer":   obj.get("answer","")
            })
    df = pd.DataFrame(records)

    # 2) MetaDB 청크 로드
    metadb_chunks = load_metadb(METADB_PATH)

    # 3) 고유 paper_id 리스트 추출
    unique_ids = df['paper_id'].unique().tolist()

    # 4) tqdm으로 진행 표시하며 역생성
    unique_rows = []
    for paper_id in tqdm(unique_ids, desc="Generating questions", unit="paper"):
        group = df[df['paper_id'] == paper_id]
        row   = group.iloc[0]
        answer = row["answer"]

        # 논문 내용 조합
        paper_content = get_paper_content_from_paper_id(paper_id, metadb_chunks)

        # 질문 생성
        try:
            new_q = generate_question_with_openai(paper_content, answer)
        except Exception as e:
            tqdm.write(f"❌ [{paper_id}] API 실패, 기존 질문 사용: {e}")
            new_q = row["question"]

        unique_rows.append({
            "paper_id": paper_id,
            "model":    MODEL_NAME,
            "question": new_q,
            "answer":   answer
        })

        time.sleep(API_DELAY)  # Rate limit 완화

    # 5) DataFrame → CSV 저장
    df_unique = pd.DataFrame(unique_rows)
    df_unique.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"\n완료: {len(df_unique)}건, '{OUTPUT_CSV}' 생성됨")

if __name__ == "__main__":
    main()
