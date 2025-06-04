# 표준 라이브러리
import os
import re
import json
import time
import random
import shutil
import yaml
from pathlib import Path
from collections import defaultdict

# 써드파티 라이브러리 - 기본
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch 관련
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch import amp
from torch.cuda.amp import autocast
from torch.distributions import Categorical

# Transformers 관련
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging
)

# 기타 ML/NLP 라이브러리
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# Hugging Face Hub
from huggingface_hub import login, create_repo, Repository

# 환경설정
from dotenv import load_dotenv

# 로컬 설정
from config import Config

# 로깅 설정
logging.set_verbosity_error()

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_dotenv(".env")
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=os.environ["HF_TOKEN"])
assert HF_TOKEN, "HF_TOKEN 환경변수 필요."

print(HF_TOKEN)

# Base 모델의 slow 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    Config.POLICY_MODEL_NAME,
    use_fast=False
)
tokenizer.padding_side = "left"

# pad_token을 eos_token과 동일하게 추가
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# 8-bit 양자화된 fine-tuned 모델 로드
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    Config.POLICY_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    use_auth_token=HF_TOKEN
)
model.eval()

# pad_token_id를 모델 설정에도 반영
model.config.pad_token_id = tokenizer.eos_token_id

# 하나의 shared policy 모델 정의 (Parameter Sharing)
# QR, Selector가 이 policy_model을 사용
policy_model = model

# JSONL 파일 경로
input_path = Config.METADB_PATH

corpus = []
with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        # 레코드에 'paper_id' 키가 없다면 'id' 키를 사용하도록 get() 활용
        paper_id = data.get("arxiv_id", data.get("id"))
        title    = data.get("title", "")
        abstract = data.get("text", "")

        # 필수 필드가 모두 존재할 때만 추가
        if paper_id and title and abstract:
            corpus.append({
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract
            })

# # 예시 출력
# print(f"총 {len(corpus)}개의 논문을 불러왔습니다.")
# print(corpus[:3])  # 앞 3개만 미리보기

gpt4o_unique_df=pd.read_csv(Config.GPT4O_DATA_PATH,index_col=False)
gpt4o_unique_df['paper_id']
gpt4o_unique_df['paper_id'] = gpt4o_unique_df['paper_id'].apply(lambda x: f"{x:.5f}")
gpt4o_unique_df=gpt4o_unique_df[gpt4o_unique_df['paper_id']!='1210.12070']
gpt4o_unique_records = gpt4o_unique_df.to_dict(orient='records')
len(gpt4o_unique_records)

# 6) CriticNetwork 정의
class CriticNetwork(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # QR 벡터와 Selector 벡터 연결 (각각 차원 H)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, qr_vec, sel_vec):
        # qr_vec, sel_vec: tensor [1, H]
        x = torch.cat([qr_vec, sel_vec], dim=-1)  # [1, 2H]
        return self.fc(x).squeeze(-1)             # scalar
    
    # # 프롬프트
# import re

def make_qr_prompts(questions: list) -> list:
    """
    QR용 prompt를 SFT 때 학습한 그대로 재현합니다.
    system + instruction + input + response:
    """
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

def make_sel_prompts_strict(questions: list, candidates_batch: list) -> list:
    """
    Llama-1B에 최적화된 간결한 Selector 프롬프트:
    • 역할과 태스크를 짧고 명확하게 전달합니다.
    • 후보 ID + 제목만 제시하고, 세 개의 ID만 출력하도록 지시합니다.
    • 출력 형식을 엄격히 고정해서 다른 텍스트가 섞이지 않게 합니다.
    """
    prompts = []
    for q, candidates in zip(questions, candidates_batch):
        # "ID: 제목" 형식으로 각 후보를 나열
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

# Generator 모델 로드.
generator_model_name = Config.GENERATOR_MODEL_NAME
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name, use_fast=False)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
generator_model = AutoModelForCausalLM.from_pretrained(
    generator_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    low_cpu_mem_usage=True,
    use_auth_token=HF_TOKEN
)
generator_model.eval()

sbert = SentenceTransformer(Config.SBERT_MODEL_NAME)  # 빠르고 괜찮은 모델

failure_phrases = Config.FAILURE_PHRASES

# 8) RAGEnv with BM25 Retriever + Generator + QR/S 단계 페널티

DEFAULT_SYSTEM = Config.DEFAULT_SYSTEM

class RAGEnv:
    def __init__(
        self,
        critic,
        corpus,
        policy_model,
        generator_model,
        generator_tokenizer
    ):
        """
        - critic: Critic 모델 (PyTorch 모듈)
        - corpus: 각 청크별 정보 리스트. entry마다 {'chunk_id':int, 'paper_id':str, 'title':str, 'abstract':str}
        - policy_model: Llama 계열의 질문 재작성(QR) 및 Selector용 LLM
        - retriever_model: (사용하지 않음 또는 BM25용 tokenizer)
        - generator_model: lmsys/vicuna-7b-v1.5 (FP16, A100) 로드된 AutoModelForCausalLM
        - generator_tokenizer: 해당 모델의 AutoTokenizer
        """
        self.device               = Config.DEVICE
        self.policy_model         = policy_model.eval()
        self.corpus               = corpus
        self.critic               = critic.to(self.device)
        self.k_retrieve           = Config.K_RETRIEVE  # BM25에서 Top-10 청크
        self.k_select             = Config.K_SELECT    # Selector 단계에서 최종 3개 논문 선택
        # 나중에 dataframe 형태로 결과 확인하기
        self.logs = []

        # (1) 하나의 논문(paper_id)에 속한 모든 청크의 abstract을 합쳐서 full-doc 생성
        tmp = defaultdict(list)
        for entry in corpus:
            pid = entry["paper_id"]
            tmp[pid].append(entry["abstract"])
        self.full_docs = {pid: " ".join(chunks) for pid, chunks in tmp.items()}

        # (2) BM25 인덱스 구축 (chunk 단위)
        self.chunk_to_pid = [entry["paper_id"] for entry in corpus]
        corpus_texts     = [entry["abstract"].lower().split() for entry in corpus]
        self.bm25        = BM25Okapi(corpus_texts)

        # ✅ 로컬 Vicuna-7B 모델 및 토크나이저 연결
        self.generator_model     = generator_model.eval()
        self.generator_tokenizer = generator_tokenizer
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token

    def batch_query_rewrite(self, questions):
        """
        questions: List[str] of 원질문 B개
        반환값:
          - seqs: Tensor([B, prompt_len + gen_len])  (토큰 ID)
          - logp: Tensor([B])                      (QR 단계에서 생성된 토큰들의 로그 확률 합)
          - qr_texts: List[str] (서브쿼리 B개)
        """
        qr_prompts = make_qr_prompts(questions)
        toks = tokenizer(
            qr_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_INPUT_LENGTH
        ).to(self.device)

        with amp.autocast(device_type="cuda"), torch.no_grad():
            out = self.policy_model.generate(
                **toks,
                do_sample=False,
                num_beams=1,
                max_new_tokens=Config.MAX_NEW_TOKENS_QR,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True
            )

        seqs   = out.sequences        # [B, prompt_len + gen_len]
        scores = out.scores           # 튜플(gen_len) of (B, vocab_size)

        # (1) 서브쿼리 텍스트 추출
        prompt_len = toks["input_ids"].shape[-1]
        gen_ids    = seqs[:, prompt_len:]  # [B, gen_len]
        decoded_all = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        qr_texts = []
        for dec in decoded_all:
            first_line = dec.strip().split("\n")[0]
            qr_texts.append(first_line.strip())

        # (2) 로그확률 계산 (QR 단계)
        scores_stack = torch.stack(scores, dim=0)               # [T, B, V]
        log_probs    = torch.log_softmax(scores_stack, dim=-1)  # [T, B, V]
        B = seqs.size(0)
        logp = torch.zeros(B, device=log_probs.device)
        idxs = torch.arange(B, device=log_probs.device)
        for t in range(gen_ids.size(1)):
          logp += log_probs[t, idxs, gen_ids[:, t]]

        return seqs, logp, qr_texts

    def get_qr_vecs(self, qr_seqs):
        """
        qr_seqs: Tensor([B, seq_len]) of QR 단계에서 생성된 토큰 ID
        반환값: Tensor([B, hidden_size])  (CLS 토큰 벡터)
        """
        with torch.no_grad():
            out = self.policy_model.model(
                input_ids=qr_seqs,
                attention_mask=qr_seqs.ne(tokenizer.pad_token_id),
                return_dict=True
            )
        # [B, seq_len, hidden_size] 중 첫 번째 토큰(CLS)만 꺼내기
        return out.last_hidden_state[:, 0, :]

    def batch_generate(self, questions, docs_batch, max_new_tokens=None, temperature=None, gen_batch_size=None):
        """
        - questions:      List[str] of 원질문 B개
        - docs_batch:     List[List[str]] of 길이 B, 각 원질문마다 3개의 "[ArXiv:PID] Title. FullAbstract"
        - max_new_tokens: 생성할 최대 토큰 수
        - temperature:    디코딩 온도
        - gen_batch_size: "한 번에" 배치로 처리할 크기

        → 반환값: List[str] of 생성된 답변 B개
        """
        # Config에서 기본값 가져오기
        if max_new_tokens is None:
            max_new_tokens = Config.MAX_NEW_TOKENS_GEN
        if temperature is None:
            temperature = Config.TEMPERATURE
        if gen_batch_size is None:
            gen_batch_size = Config.GEN_BATCH_SIZE
            
        B = len(questions)
        answers = [""] * B

        # 1) B개의 프롬프트 리스트 생성 (원본과 동일한 방식)
        prompts = []
        for q, docs in zip(questions, docs_batch):
            system_msg = DEFAULT_SYSTEM
            doc_section = ""
            for idx_d, d in enumerate(docs):
                doc_section += f"Document{idx_d}: {d}\n"
            user_prompt = f"Question: {q}\n{doc_section}Answer:"
            concat_prompt = system_msg + "\n" + user_prompt
            prompts.append(concat_prompt)

        # 2) gen_batch_size 단위로 나눠서 한 번에 batch 처리
        for start in range(0, B, gen_batch_size):
            end = min(start + gen_batch_size, B)
            batch_prompts = prompts[start:end]

            # (a) 토크나이징
            inputs = self.generator_tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=Config.MAX_GEN_INPUT_LENGTH  # 필요에 따라 조절
            ).to(self.device)

            # (b) 배치 단위로 한 번에 generate 호출
            with torch.no_grad():
                out = self.generator_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=False,
                    pad_token_id=self.generator_tokenizer.pad_token_id,
                    eos_token_id=self.generator_tokenizer.eos_token_id,
                )

            # (c) 입력 길이 이후의 토큰만 추출해서 디코딩
            input_len = inputs["input_ids"].shape[1]  # (배치 내 모든 시퀀스 동일)
            for i in range(end - start):
                generated_ids = out[i, input_len:]
                decoded = self.generator_tokenizer.decode(generated_ids, skip_special_tokens=True)
                answers[start + i] = decoded.strip()

        return answers

    def step_batch(self, batch_idxs, batch_qs, batch_gt):
        B = len(batch_qs)

        # 1) QR 단계
        qr_seqs, logp_qr, qr_texts = self.batch_query_rewrite(batch_qs)
        qr_vecs = self.get_qr_vecs(qr_seqs)
        qr_invalid = [ (qr_texts[i].strip()=="" or qr_texts[i].strip()==batch_qs[i].strip()) for i in range(B) ]

        rewards, values = [], []
        docs_for_gen, s_invalid = [], [False]*B

        # 2) BM25 → all_candidates_simple 생성 (for i in range(B))
        all_candidates_simple = []
        for i in range(B):
            q_tokens  = batch_qs[i].lower().split()
            scores_q  = self.bm25.get_scores(q_tokens)
            top_q_idx = np.argsort(scores_q)[-self.k_retrieve:][::-1]

            sq_tokens  = qr_texts[i].lower().split()
            scores_sq  = self.bm25.get_scores(sq_tokens)
            top_sq_idx = np.argsort(scores_sq)[-self.k_retrieve:][::-1]

            all_chunk_ids = np.concatenate((top_q_idx, top_sq_idx), axis=0)
            seen = set()
            candidates_simple = []
            for cid in all_chunk_ids:
                pid   = self.chunk_to_pid[cid]
                title = self.corpus[cid]["title"]
                if pid not in seen:
                    seen.add(pid)
                    candidates_simple.append((pid, title))
            all_candidates_simple.append(candidates_simple)

        # 3) Selector 병렬화: B개 프롬프트 한 번에 생성
        sel_prompts = make_sel_prompts_strict(batch_qs, all_candidates_simple)

        toks = tokenizer(
            sel_prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=Config.MAX_SEL_INPUT_LENGTH
        ).to(self.device)

        with torch.no_grad():
            out_sel = self.policy_model.generate(
                **toks,
                max_new_tokens=32,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        seqs_sel     = out_sel.sequences   # [B, total_len]
        scores_sel   = out_sel.scores      # (gen_len,) of (B, V)
        scores_stack = torch.stack(scores_sel, dim=0)  # [gen_len, B, V]
        log_probs    = torch.log_softmax(scores_stack, dim=-1)

        total_len  = seqs_sel.size(1)
        prompt_len = toks["input_ids"].size(1)
        gen_len    = total_len - prompt_len

        gen_ids = seqs_sel[:, prompt_len:]          # [B, gen_len]
        decoded_list = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)  # List[str], 길이 B


        # 4) decoded_list를 순회하며 selected_ids_batch 생성, Critic 계산
        for i in range(B):
            decoded_sel = decoded_list[i].strip()
            pattern = r"^\s*\d{4}\.\d{5}\s*,\s*\d{4}\.\d{5}\s*,\s*\d{4}\.\d{5}\s*$"
            if not re.match(pattern, decoded_sel):
                s_invalid[i] = True

            sel_ids = re.findall(r"\d{4}\.\d{5}", decoded_sel)
            if len(sel_ids) < self.k_select:
                for pid2, _ in all_candidates_simple[i]:
                    if pid2 not in sel_ids:
                        sel_ids.append(pid2)
                    if len(sel_ids) == self.k_select:
                        break

            # chosen_docs 준비
            chosen_docs = []
            for sid in sel_ids[: self.k_select]:
                full_text = self.full_docs.get(sid, "")
                title     = next((t for (p, t) in all_candidates_simple[i] if p == sid), "")
                chosen_docs.append(f"[ArXiv:{sid}] {title}. {full_text}")
            docs_for_gen.append(chosen_docs)

            # Critic 계산
            with torch.no_grad():
                d_toks   = tokenizer(chosen_docs, return_tensors="pt", padding=True, truncation=True).to(self.device)
                d_out    = self.policy_model.model(**d_toks, return_dict=True)
                sel_embs = d_out.last_hidden_state.mean(dim=1)  # [3, hidden]
            sel_emb = sel_embs.mean(dim=0, keepdim=True)      # [1, hidden]

            with torch.cuda.amp.autocast(enabled=False):
                v = self.critic(qr_vecs[i].unsqueeze(0).float(), sel_emb.float()).squeeze(0)
            values.append(v)

        # 5) Selector 로그확률 합산 (배치 전체)
        idxs   = torch.arange(B, device=log_probs.device)
        logp_s = torch.zeros(B, device=log_probs.device)
        for t in range(gen_len):
            logp_s += log_probs[t, idxs, gen_ids[:, t]]

        # 6) Generator 단계 (Config 기본값 사용)
        gen_ans = self.batch_generate(batch_qs, docs_for_gen)

        # 7) Reward 계산 (기존과 동일)
        rewards = []
        for i, ans in enumerate(gen_ans):
          try:
            if any(phrase in ans.lower() for phrase in failure_phrases):
              reward = 0.4
            else:
              emb_pred = sbert.encode(ans, convert_to_tensor=True, device="cpu")
              emb_gt   = sbert.encode(batch_gt[i], convert_to_tensor=True, device="cpu")
              reward = util.cos_sim(emb_pred, emb_gt).item()
            if qr_invalid[i]:
              reward = max(0.0, reward - 0.1)
            if s_invalid[i]:
              reward = max(0.0, reward - 0.1)
          except Exception as e:
            reward = 0.0
          self.logs.append({
              "question": batch_qs[i],
              "qr_text": qr_texts[i],
              "selector_output": decoded_list[i].strip(),
              "gen_answer": ans,
              "gt": batch_gt[i],
              "qr_invalid": qr_invalid[i],
              "selector_invalid": s_invalid[i],
              "reward": reward })
          rewards.append(reward)

        rewards    = torch.tensor(rewards, device=self.device)
        values     = torch.stack(values)          # [B]
        logp_joint = logp_s + logp_qr.to(self.device)
        return rewards, values, logp_joint

# 하이퍼파라미터
critic      = CriticNetwork(policy_model.config.hidden_size).to(Config.DEVICE)
env         = RAGEnv(critic, corpus, policy_model, generator_model, generator_tokenizer)

# 하나의 policy optimizer로 QR/S(shared LLM) 업데이트
opt_policy  = Adam(list(policy_model.parameters()), lr=Config.POLICY_LEARNING_RATE)
# Critic만 별도 업데이트
opt_critic  = Adam(critic.parameters(), lr=Config.CRITIC_LEARNING_RATE)
# retriever_model = retriever_model.cpu()

# ------------------------------------------------------------
# 설정
# ------------------------------------------------------------
os.environ["CUDA_LAUNCH_BLOCKING"] = Config.CUDA_LAUNCH_BLOCKING

# (1) Hugging Face에선 먼저 아래 명령어로 토큰 설정이 필요합니다:
#     export HUGGINGFACE_HUB_TOKEN="당신의_HF_토큰"
#
# (2) repo_id를 아직 만들지 않았다면, create_repo로 자동 생성합니다.
repo_id = Config.REPO_ID
create_repo(repo_id, exist_ok=True)

local_repo_dir = Config.LOCAL_REPO_DIR

# ------------------------------------------------------------
# 로컬 레포 준비
# ------------------------------------------------------------
def setup_repo(local_dir: str, repo_id: str):
    """
    로컬 폴더가 이미 존재하지만 Git 레포가 아니면 삭제 후 새로 클론,
    없는 경우에는 바로 클론합니다.
    """
    if os.path.isdir(local_dir):
        # .git 디렉토리가 없으면 기존 폴더를 지우고 다시 클론
        if not os.path.isdir(os.path.join(local_dir, ".git")):
            shutil.rmtree(local_dir)
            print(f"Deleted non-repo folder: {local_dir}")
            Repository(local_dir=local_dir, clone_from=repo_id)
            print(f"Cloned fresh repo into {local_dir}")
        else:
            # 이미 올바른 Git 레포가 있으면 그대로 사용
            print(f"Using existing repo at {local_dir}")
    else:
        # 로컬 폴더가 없으면 클론
        Repository(local_dir=local_dir, clone_from=repo_id)
        print(f"Cloned repo into {local_dir}")

# 레포 설정 실행
setup_repo(local_repo_dir, repo_id)
repo = Repository(local_dir=local_repo_dir)

# ------------------------------------------------------------
# 체크포인트 로드 함수
# ------------------------------------------------------------
def find_latest_checkpoint(repo_dir: str):
    """
    local_repo_dir 아래에 'checkpoint-stepXXX' 폴더가 있으면,
    그중 가장 큰 step 번호를 가진 폴더명을 반환합니다.
    없으면 None을 반환합니다.
    """
    candidates = []
    for name in os.listdir(repo_dir):
        if name.startswith("checkpoint-step"):
            try:
                step_num = int(name.replace("checkpoint-step", ""))
                candidates.append((step_num, name))
            except:
                continue
    if not candidates:
        return None
    latest = max(candidates, key=lambda x: x[0])
    return latest[1]  # 폴더명, 예: "checkpoint-step30"

# ------------------------------------------------------------
# 모델 / 옵티마이저 / 토크나이저 초기화
# ------------------------------------------------------------
# (이미 env, policy_model, tokenizer, critic, opt_policy, opt_critic 정의되어 있다고 가정)

# 학습할 총 epochs와 배치 사이즈
epochs, batch_size = Config.EPOCHS, Config.BATCH_SIZE

# 전역 step 카운터 초기화
start_global_step = 0


# 마지막 저장된 체크포인트 디렉토리 이름
last_ckpt_dirname = find_latest_checkpoint(local_repo_dir)

if last_ckpt_dirname is not None:
    # 체크포인트가 있으면 로드
    ckpt_path = os.path.join(local_repo_dir, last_ckpt_dirname)
    print(f"Resuming from checkpoint: {last_ckpt_dirname}")

    # 1) policy_model 및 tokenizer 로드
    policy_model = policy_model.from_pretrained(ckpt_path)
    tokenizer.from_pretrained(ckpt_path)

    # 2) critic 로드
    critic.load_state_dict(torch.load(os.path.join(ckpt_path, "critic.pt"), map_location=Config.DEVICE))

    # 3) 옵티마이저 로드
    opt_policy.load_state_dict(torch.load(os.path.join(ckpt_path, "opt_policy.pt"), map_location=Config.DEVICE))
    opt_critic.load_state_dict(torch.load(os.path.join(ckpt_path, "opt_critic.pt"), map_location=Config.DEVICE))

    # 4) global_step 정보 로드
    # checkpoint 폴더명 형식이 "checkpoint-step{N}" 이므로 N을 읽어서 start_global_step 설정
    try:
        start_global_step = int(last_ckpt_dirname.replace("checkpoint-step", "")) + 1
    except:
        start_global_step = 0
else:
    print("No checkpoint found, starting from scratch.")

# ------------------------------------------------------------
# 학습 루프 (N 배치마다 체크포인트 저장)
# ------------------------------------------------------------
def compute_returns(rs, gamma=None):
    if gamma is None:
        gamma = Config.GAMMA
    R, rets = 0, []
    for r in reversed(rs.tolist()):
        R = r + gamma * R
        rets.insert(0, R)
    return torch.tensor(rets, device=Config.DEVICE)

global_step = start_global_step
saved_ckpt_path = None  # 마지막으로 저장된 체크포인트 경로

for epoch in range(epochs):
    print(f"\n=== Epoch {epoch+1}/{epochs} ===")
    sampled = gpt4o_unique_records  # records 리스트
    idxs = list(range(len(sampled)))
    qs   = [r['question'] for r in sampled]
    ans  = [r['answer']   for r in sampled]

    # 배치 개수
    num_batches = (len(sampled) + batch_size - 1) // batch_size

    # tqdm을 활용해 배치 진행률과 시간 예측 표시
    for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}", ncols=80):
        batch_start_time = time.time()
        # start_global_step 이전 스텝 건너뛰기
        if global_step < start_global_step:
            global_step += 1
            continue

        start = batch_idx * batch_size
        b_idx = idxs[start:start+batch_size]
        b_qs  = qs[start:start+batch_size]
        b_as  = ans[start:start+batch_size]

        # RAGEnv step 및 PPO 업데이트
        rewards, values, logp_joint = env.step_batch(b_idx, b_qs, b_as)
        rets   = compute_returns(rewards)
        adv    = (rets - values).detach()
        adv    = (adv - adv.mean()) / (adv.std() + 1e-8)

        ratio_s    = torch.exp(logp_joint)
        loss_actor = -torch.min(ratio_s * adv, torch.clamp(ratio_s, 1 - Config.EPSILON, 1 + Config.EPSILON) * adv).mean()
        loss_critic = F.mse_loss(values, rets)

        opt_policy.zero_grad()
        opt_critic.zero_grad()
        (loss_actor + 0.5 * loss_critic).backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), Config.GRAD_CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(critic.parameters(),      Config.GRAD_CLIP_NORM)
        opt_policy.step()
        opt_critic.step()

        del rewards, values, logp_joint, rets, adv, ratio_s
        torch.cuda.empty_cache()

        # ------------------------------------------------------------------------------------
        # N 배치마다 체크포인트 저장 및 이전 체크포인트 삭제
        # ------------------------------------------------------------------------------------
        if global_step % Config.CHECKPOINT_SAVE_INTERVAL == 0:
            # 이전 체크포인트 삭제
            if saved_ckpt_path is not None and os.path.isdir(saved_ckpt_path):
                shutil.rmtree(saved_ckpt_path)

            # 새로운 체크포인트 경로
            ckpt_dirname = f"checkpoint-step{global_step}"
            ckpt_dir = os.path.join(local_repo_dir, ckpt_dirname)
            os.makedirs(ckpt_dir, exist_ok=True)

            # 1) policy_model & tokenizer 저장
            policy_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

            # 2) critic 저장
            torch.save(critic.state_dict(), os.path.join(ckpt_dir, "critic.pt"))

            # 3) 옵티마이저 상태 저장
            torch.save(opt_policy.state_dict(), os.path.join(ckpt_dir, "opt_policy.pt"))
            torch.save(opt_critic.state_dict(), os.path.join(ckpt_dir, "opt_critic.pt"))

            # 4) 커밋 및 푸시
            #repo.push_to_hub(commit_message=f"Checkpoint at global_step {global_step}")

            print(f"Saved & pushed checkpoint: {ckpt_dirname}")
            saved_ckpt_path = ckpt_dir


        global_step += 1
        batch_end_time = time.time()
        batch_time_sec = batch_end_time - batch_start_time
        print(f"[Batch {batch_idx}] Time: {batch_time_sec:.2f} sec")

    pd.DataFrame(env.logs).to_csv(Config.LOG_OUTPUT_PATH.replace("epoch_3", f"epoch_{epoch+1}"), index=False)
    env.logs = []
    print(f"Epoch {epoch+1} complete")
    break

print("Training finished.")
