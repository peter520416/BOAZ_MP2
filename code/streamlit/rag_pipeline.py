"""
MAPPO RAG 추론 파이프라인
run_mappo.py의 추론 기능을 스트림릿 앱에서 사용할 수 있도록 추출한 모듈
사전 훈련된 peter520416/llama1b-MMOA_RAG_Final_cp180 모델 사용
"""

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging
)
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from streamlit_config import Config  # 스트림릿 전용 config 사용

# 로깅 설정
logging.set_verbosity_error()

# 환경변수 로드 (프로젝트 루트에서)
def find_project_root():
    """프로젝트 루트를 찾습니다."""
    current = Path.cwd()
    
    # 현재 디렉토리가 BOAZ_MP2인지 확인
    if current.name == "BOAZ_MP2":
        return current
    
    # 상위 디렉토리들을 확인
    for parent in current.parents:
        if parent.name == "BOAZ_MP2":
            return parent
    
    # 찾지 못한 경우 현재 디렉토리 반환
    return current

project_root = find_project_root()
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

class RAGPipeline:
    """MAPPO RAG 추론 파이프라인 - 사전 훈련된 모델 사용"""
    
    def __init__(self):
        """모든 필요한 모델과 데이터를 초기화합니다."""
        self.device = Config.DEVICE
        self._load_models()
        self._load_corpus()
        self._setup_bm25()
        
    def _load_models(self):
        """모든 필요한 모델들을 로드합니다."""
        print("📦 사전 훈련된 모델 로드 중...")
        
        # 정책 모델 (QR & Selector용) - 사전 훈련된 모델 사용
        print(f"🔄 Policy 모델 로드: {Config.POLICY_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'meta-llama/Llama-3.2-1B-Instruct',
            use_fast=False
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 8bit 양자화 설정
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=Config.LOAD_IN_8BIT
        )
        
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            Config.POLICY_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=Config.LOW_CPU_MEM_USAGE,
            use_auth_token=HF_TOKEN
        )
        self.policy_model.eval()
        self.policy_model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # 생성 모델
        print(f"🔄 Generator 모델 로드: {Config.GENERATOR_MODEL_NAME}")
        self.generator_tokenizer = AutoTokenizer.from_pretrained(
            Config.GENERATOR_MODEL_NAME, 
            use_fast=False
        )
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            Config.GENERATOR_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=Config.LOW_CPU_MEM_USAGE,
            use_auth_token=HF_TOKEN
        )
        self.generator_model.eval()
        
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
            
        # SBERT 모델
        print(f"🔄 SBERT 모델 로드: {Config.SBERT_MODEL_NAME}")
        self.sbert = SentenceTransformer(Config.SBERT_MODEL_NAME)
        
        print("✅ 모든 모델 로드 완료!")
    
    def _load_corpus(self):
        """코퍼스 데이터를 로드합니다."""
        print("📚 코퍼스 로드 중...")
        
        # 절대 경로로 변환
        if Config.METADB_PATH.startswith("./"):
            corpus_path = project_root / Config.METADB_PATH[2:]
        else:
            corpus_path = Path(Config.METADB_PATH)
        
        self.corpus = []
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    paper_id = data.get("arxiv_id", data.get("id"))
                    title = data.get("title", "")
                    abstract = data.get("text", "")
                    
                    if paper_id and title and abstract:
                        self.corpus.append({
                            "paper_id": paper_id,
                            "title": title,
                            "abstract": abstract
                        })
        except FileNotFoundError:
            print(f"❌ 코퍼스 파일을 찾을 수 없습니다: {corpus_path}")
            raise
        
        # 전체 문서 생성 (paper_id별로 모든 청크 합치기)
        tmp = defaultdict(list)
        for entry in self.corpus:
            pid = entry["paper_id"]
            tmp[pid].append(entry["abstract"])
        self.full_docs = {pid: " ".join(chunks) for pid, chunks in tmp.items()}
        
        print(f"✅ {len(self.corpus)}개의 논문 로드 완료")
    
    def _setup_bm25(self):
        """BM25 인덱스를 구축합니다."""
        print("🔍 BM25 인덱스 구축 중...")
        
        self.chunk_to_pid = [entry["paper_id"] for entry in self.corpus]
        corpus_texts = [entry["abstract"].lower().split() for entry in self.corpus]
        self.bm25 = BM25Okapi(corpus_texts)
        
        print("✅ BM25 인덱스 구축 완료")
    
    def make_qr_prompts(self, questions):
        """QR용 프롬프트를 생성합니다."""
        prompts = []
        for q in questions:
            p = (
                f"{Config.QR_SYSTEM_PROMPT}\n"
                f"{Config.QR_INSTRUCTION}"
                f"input: given question -> {q}\n"
                "response:"
            )
            prompts.append(p)
        return prompts
    
    def make_sel_prompts_strict(self, questions, candidates_batch):
        """Selector용 프롬프트를 생성합니다."""
        prompts = []
        for q, candidates in zip(questions, candidates_batch):
            cand_lines = [f"{pid}: {title}" for pid, title in candidates]
            cand_text = "\n".join(cand_lines)
            
            p = (
                f"{Config.SEL_SYSTEM_PROMPT}\n"
                f"{Config.SEL_INSTRUCTION}\n\n"
                f"Question: {q}\n"
                f"Candidates:\n{cand_text}\n"
                "Answer:"
            )
            prompts.append(p)
        return prompts
    
    def query_rewrite(self, question):
        """단일 질문에 대해 질문 재작성을 수행합니다."""
        prompts = self.make_qr_prompts([question])
        
        toks = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_INPUT_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            out = self.policy_model.generate(
                **toks,
                do_sample=False,
                num_beams=1,
                max_new_tokens=Config.MAX_NEW_TOKENS_QR,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        prompt_len = toks["input_ids"].shape[-1]
        gen_ids = out[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]
        qr_text = decoded.strip().split("\n")[0].strip()
        
        return qr_text
    
    def retrieve_documents(self, question, qr_text):
        """BM25를 사용하여 문서를 검색합니다."""
        # 원본 질문으로 검색
        q_tokens = question.lower().split()
        scores_q = self.bm25.get_scores(q_tokens)
        top_q_idx = np.argsort(scores_q)[-Config.K_RETRIEVE:][::-1]
        
        # 재작성된 질문으로 검색
        sq_tokens = qr_text.lower().split()
        scores_sq = self.bm25.get_scores(sq_tokens)
        top_sq_idx = np.argsort(scores_sq)[-Config.K_RETRIEVE:][::-1]
        
        # 결합하여 후보 논문 추출
        all_chunk_ids = np.concatenate((top_q_idx, top_sq_idx), axis=0)
        seen = set()
        candidates = []
        
        for cid in all_chunk_ids:
            if cid < len(self.chunk_to_pid):  # 인덱스 범위 확인
                pid = self.chunk_to_pid[cid]
                title = self.corpus[cid]["title"]
                if pid not in seen:
                    seen.add(pid)
                    candidates.append((pid, title))
        
        return candidates
    
    def select_documents(self, question, candidates):
        """Selector 모델을 사용하여 최종 문서를 선택합니다."""
        prompts = self.make_sel_prompts_strict([question], [candidates])
        
        toks = self.tokenizer(
            prompts, 
            return_tensors="pt",
            padding=True, 
            truncation=True, 
            max_length=Config.MAX_SEL_INPUT_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            out = self.policy_model.generate(
                **toks,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        prompt_len = toks["input_ids"].size(1)
        gen_ids = out[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        
        # ID 추출
        sel_ids = re.findall(r"\d{4}\.\d{5}", decoded)
        
        # 부족한 경우 후보에서 보충
        if len(sel_ids) < Config.K_SELECT:
            for pid, _ in candidates:
                if pid not in sel_ids:
                    sel_ids.append(pid)
                if len(sel_ids) == Config.K_SELECT:
                    break
        
        return sel_ids[:Config.K_SELECT]
    
    def generate_answer(self, question, selected_ids):
        """선택된 문서들을 바탕으로 최종 답변을 생성합니다."""
        # 선택된 문서들 준비
        docs = []
        for sid in selected_ids:
            full_text = self.full_docs.get(sid, "")
            # candidates에서 제목 찾기
            title = next((t for (p, t) in self.candidates if p == sid), f"Paper {sid}")
            docs.append(f"[ArXiv:{sid}] {title}. {full_text}")
        
        # 프롬프트 생성
        system_msg = Config.DEFAULT_SYSTEM
        doc_section = ""
        for idx_d, d in enumerate(docs):
            doc_section += f"Document{idx_d}: {d}\n"
        
        user_prompt = f"Question: {question}\n{doc_section}Answer:"
        concat_prompt = system_msg + "\n" + user_prompt
        
        # 토크나이징
        inputs = self.generator_tokenizer(
            [concat_prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_GEN_INPUT_LENGTH
        ).to(self.device)
        
        # 답변 생성
        with torch.no_grad():
            out = self.generator_model.generate(
                **inputs,
                max_new_tokens=Config.MAX_NEW_TOKENS_GEN,
                temperature=Config.TEMPERATURE,
                do_sample=True if Config.TEMPERATURE > 0 else False,
                pad_token_id=self.generator_tokenizer.pad_token_id,
                eos_token_id=self.generator_tokenizer.eos_token_id,
            )
        
        # 결과 디코딩
        input_len = inputs["input_ids"].shape[1]
        generated_ids = out[0, input_len:]
        answer = self.generator_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        return answer, docs

# 전역 파이프라인 인스턴스
_pipeline = None

def get_pipeline():
    """전역 파이프라인 인스턴스를 반환합니다."""
    global _pipeline
    if _pipeline is None:
        print("🚀 RAG 파이프라인 초기화 중...")
        _pipeline = RAGPipeline()
        print("✅ RAG 파이프라인 준비 완료!")
    return _pipeline

def infer_attention_question(question):
    """
    단일 질문에 대해 전체 RAG 파이프라인을 실행합니다.
    
    Args:
        question (str): 입력 질문
        
    Returns:
        dict: 다음 키를 포함하는 결과 딕셔너리
            - qr_text: 재작성된 질문
            - sel_ids: 선택된 문서 ID들
            - final_answer: 최종 답변
            - documents: 사용된 문서들
    """
    pipeline = get_pipeline()
    
    print(f"🔍 질문 처리 시작: {question}")
    
    # 1. 질문 재작성
    print("🔄 1단계: 질문 재작성 중...")
    qr_text = pipeline.query_rewrite(question)
    print(f"✅ 재작성된 질문: {qr_text}")
    
    # 2. 문서 검색
    print("📚 2단계: 문서 검색 중...")
    candidates = pipeline.retrieve_documents(question, qr_text)
    pipeline.candidates = candidates  # generate_answer에서 사용하기 위해 저장
    print(f"✅ {len(candidates)}개 후보 문서 발견")
    
    # 3. 문서 선택
    print("🎯 3단계: 최적 문서 선택 중...")
    selected_ids = pipeline.select_documents(question, candidates)
    print(f"✅ 선택된 문서: {selected_ids}")
    
    # 4. 답변 생성
    print("🤖 4단계: 답변 생성 중...")
    final_answer, documents = pipeline.generate_answer(question, selected_ids)
    print("✅ 답변 생성 완료!")
    
    return {
        "qr_text": qr_text,
        "sel_ids": selected_ids,
        "final_answer": final_answer,
        "documents": documents
    } 