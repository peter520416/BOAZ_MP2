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
│   ├── run_mappo.py               # 메인 MAPPO 학습 스크립트
│   ├── config.py                  # 설정 파일 (PY)
│   ├── config.yaml                # 설정 파일 (YAML)
│   ├── get_unique_df.py           # 데이터 전처리
│   └── README.md                  # 코드 사용법 가이드
├── data/                          # 데이터 파일들 (gitignore로 제외)
│   ├── gpt4o_unique_df.csv       # 고유 질문-답변 쌍
│   ├── final_qa.jsonl            # 원본 QA 데이터 (용량이 커 제외)
│   ├── gpt4o_qa_top10.jsonl      # 원본 QA 데이터를 GPT-4o-mini를 이용하여 질문을 다시 생성한 데이터 (용량이 커 제외)
│   └── MetaDB_with_date_id.jsonl # 논문 데이터베이스
├── requirements.txt               # Python 패키지
├── .gitignore                    # Git 제외 파일 목록
└── README.md                     # 현재 파일
```

## 코드 사용법
!!!본 코드는 colab pro a100 환경에서 실행하는 것을 권고드립니다!!!

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repo-url>
cd BOAZ_MP2

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

BOAZ_MP2 경로에 `.env` 파일을 생성하고 Hugging Face 토큰을 설정:

```bash
HF_TOKEN=your_huggingface_token_here
```


### 3. 학습 실행

```bash
python code/run_mappo.py
```

## ⚙️ 설정

모든 하이퍼파라미터와 프롬프트 등 설정은 `code/config.yaml` 파일에서 관리됩니다:

```yaml
# 모델 설정
models:
  policy_model_name: "meta-llama/Llama-3.2-1B-Instruct"
  generator_model_name: "meta-llama/Llama-3.1-8B-Instruct"

# 하이퍼파라미터
hyperparameters:
  epochs: 3
  batch_size: 128
  policy_learning_rate: 1.0e-5
```

자세한 설정 방법은 `code/README.md`를 참조하세요.

## 주요 기능

### Multi-Agent 학습
- **Policy Model**: Query Rewriting과 Document Selection 담당
- **Critic Network**: 각 에이전트의 행동 평가
- **Shared Parameters**: 효율적인 학습을 위한 파라미터 공유

### RAG 파이프라인
- **BM25 Retrieval**: 빠르고 효과적인 문서 검색
- **Multi-step Reasoning**: 단계별 최적화

### 실험 및 모니터링
- **체크포인트 자동 저장**: 학습 중단 시 복구 가능
- **상세 로깅**: 각 단계별 성능 메트릭 추적
- **Hugging Face 통합**: 모델 자동 업로드 및 공유

## 참고 자료

- [MAPPO 논문](https://arxiv.org/abs/2501.15228)

## 참여자

- **박소연**: 깃허브 링크
- **원서현**: 깃허브 링크
- **정명훈**: 깃허브 링크
- **정주현**: https://github.com/peter520416

--- 