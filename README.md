# BOAZ MP2 - AFA (AI for AI)

BOAZ(Big Data Academy) 24기 Mini Project 2 - AI 학습 도우미
Multi-Agent Proximal Policy Optimization (MAPPO)를 활용한 Retrieval-Augmented Generation (RAG) 시스템

## 프로젝트 개요

이 프로젝트는 강화학습(MAPPO)을 활용하여 RAG 시스템의 성능을 개선한다는 논문을 통해 AI 논문에 대한 공부를 쉽게 할 수 있도록 저희만의 데이터셋으로 MARL을 진행한 프로젝트입니다!
Query Rewriting, Document Selection 단계를 최적화하여 더 정확한 답변 생성을 목표로 합니다.

## 시스템 아키텍처

```
Query → QR (Query Rewriting) → BM25 Retrieval → Selector → Generator → Answer
         ↑                                        ↑
         └────────────── MAPPO Training ──────────┘
```

### 주요 구성 요소
- **Query Rewriter (QR)**: 입력 질문을 검색에 최적화된 형태로 1개 재작성
- **Document Selector**: BM25로 검색된 문서 중 가장 관련성 높은 3개의 문서 선택
- **Answer Generator**: 선택된 문서를 바탕으로 최종 답변 생성
- **Critic Network**: 각 단계의 성능을 평가하고 학습 신호 제공

## 📁 프로젝트 구조

```
BOAZ_MP2/
├── code/                           # 소스 코드
│   ├── streamlit/                  # Streamlit 웹 애플리케이션
│   │   ├── streamlit_app.py        # 메인 웹 애플리케이션
│   │   ├── rag_pipeline.py         # RAG 추론 파이프라인
│   │   ├── streamlit_config.py     # Streamlit 전용 설정
│   │   ├── streamlit_config.yaml   # 기본 설정 파일
│   │   └── run_streamlit.py        # Streamlit 실행 헬퍼 스크립트
│   ├── run_mappo.py               # 메인 MAPPO 학습 스크립트
│   ├── config.py                  # 설정 파일 (PY)
│   ├── config.yaml                # 설정 파일 (YAML)
│   ├── get_unique_df.py           # 데이터 전처리
│   └── README.md                  # 코드 사용법 가이드
├── data/                          # 데이터 파일들 (gitignore로 제외)
│   ├── gpt4o_unique_df.csv       # gpt4o_qa_top10.jsonl 파일의 paper_id가 unique한 데이터
│   ├── final_qa.jsonl            # 원본 QA 데이터 (용량이 커 제외)
│   ├── gpt4o_qa_top10.jsonl      # 원본 QA 데이터를 GPT-4o-mini를 이용하여 질문을 다시 생성한 데이터 (용량이 커 제외)
│   └── MetaDB_with_date_id.jsonl # 논문 데이터베이스
├── requirements.txt              # Python 패키지
├── .env                          # 환경 변수 (HF_TOKEN 설정)
├── .gitignore                    # Git 제외 파일 목록
└── README.md                     # 현재 파일
```

## 시작 (Streamlit 웹 앱)

**권장 환경**: Google Colab Pro (A100 GPU)

### 1. 저장소 클론
```bash
!git clone <repo-url>
cd BOAZ_MP2
```

### 2. 환경변수 설정
```bash
# .env 파일
HF_TOKEN=your_huggingface_token_here
```

### 3. 의존성 설치 및 앱 실행
```bash
!pip install -r requirements.txt
```

### 4. Streamlit 앱 실행
```bash
!python code/streamlit/run_streamlit.py
```

## 웹 인터페이스 사용법

1. **질문 입력**: AI 관련 질문을 입력
2. **답변 생성**: "질문하기" 버튼 클릭
3. **결과 확인**: 
   - **Rewritten Question**: QR 모델이 재작성한 질문
   - **Selected Document IDs**: 선택된 논문 ID들
   - **Final Answer**: 최종 답변
   - **Used Documents**: 답변 생성에 사용된 문서들


## 학습 및 개발

### MAPPO 학습 실행

```bash
!python code/run_mappo.py
```

### 학습 설정

MAPPO 학습 설정은 `code/config.yaml`에서 관리됩니다:

```yaml
# 모델 설정
models:
  policy_model_name: "meta-llama/Llama-3.2-1B-Instruct"
  generator_model_name: "meta-llama/Llama-3.1-8B-Instruct"

# RAG 설정
rag:
  k_retrieve: 10      # BM25에서 검색할 문서 수
  k_select: 3         # 최종 선택할 문서 수

# 하이퍼파라미터
hyperparameters:
  epochs: 3
  batch_size: 128
  policy_learning_rate: 1.0e-5
```

### Streamlit 설정

Streamlit 앱의 설정은 `code/streamlit/streamlit_config.yaml`에서 관리됩니다:

```yaml
# 모델 설정 (추론용)
models:
  policy_model_name: "peter520416/llama1b-MMOA_RAG_Final_cp180"  # 사전 훈련된 모델
  generator_model_name: "meta-llama/Llama-3.1-8B-Instruct"
  sbert_model_name: "all-MiniLM-L6-v2"
```

## 기술 스택

### Streamlit 웹 앱
- **Frontend**: Streamlit
- **Backend**: PyTorch, Transformers
- **Models**: 
  - Policy: `peter520416/llama1b-MMOA_RAG_Final_cp180` (사전 훈련됨)
  - Generator: `meta-llama/Llama-3.1-8B-Instruct`
  - Retrieval: BM25 + Sentence-BERT

### 학습 환경
- **Framework**: PyTorch
- **Reinforcement Learning**: Multi-Agent PPO
- **Quantization**: 8-bit via BitsAndBytes
- **Model Hosting**: Hugging Face Hub

## 주요 기능

### Multi-Agent 학습
- **Policy Model**: Query Rewriting과 Document Selection 담당
- **Critic Network**: 각 에이전트의 행동 평가
- **Shared Parameters**: 효율적인 학습을 위한 파라미터 공유

### RAG 파이프라인
- **BM25 Retrieval**: 빠르고 효과적인 문서 검색
- **Document Selection**: 강화학습으로 최적화된 문서 선택
- **Answer Generation**: 선택된 문서 기반 답변 생성

### 웹 인터페이스
- **실시간 추론**: 사전 훈련된 모델로 즉시 답변 생성
- **단계별 결과**: QR, 문서 선택, 최종 답변 과정 시각화
- **사용자 친화적**: 직관적인 웹 인터페이스

### 실험 및 모니터링
- **체크포인트 자동 저장**: 학습 중단 시 복구 가능
- **상세 로깅**: 각 단계별 성능 메트릭 추적
- **Hugging Face 통합**: 모델 자동 업로드 및 공유

## 주의사항

- **첫 실행**: 모델 다운로드로 인해 10-15분 소요 가능
- **GPU 메모리**: 최소 12GB GPU 메모리 권장
- **인터넷 연결**: 모델 다운로드를 위한 안정적인 인터넷 필요
- **Colab 제한**: 무료 Colab에서는 메모리 부족 가능성

## 참고 자료

- [MAPPO 논문](https://arxiv.org/abs/2501.15228)
- [훈련된 모델](https://huggingface.co/peter520416/llama1b-MMOA_RAG_Final_cp180)

## 참여자

- **박소연**: 깃허브 링크
- **원서현**: 깃허브 링크
- **정명훈**: 깃허브 링크
- **정주현**: https://github.com/peter520416

--- 
