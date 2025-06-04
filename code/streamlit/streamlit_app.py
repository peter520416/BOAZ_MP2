"""
MAPPO RAG 스트림릿 웹 애플리케이션
사전 훈련된 모델을 사용한 논문 질의응답 시스템
"""

import streamlit as st
import time
from rag_pipeline import infer_attention_question

# 페이지 설정
st.set_page_config(
    page_title="MAPPO RAG 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .example-question {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
        cursor: pointer;
        border: 1px solid #e0e0e0;
    }
    
    .example-question:hover {
        background-color: #e0e4ea;
        border-color: #667eea;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .status-loading {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .status-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .document-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .highlight {
        background-color: #ffffcc;
        padding: 0.2rem;
        border-radius: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# 메인 타이틀
st.markdown('<h1 class="main-title">🤖 MAPPO RAG 챗봇</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">사전 훈련된 MAPPO 모델을 사용한 논문 질의응답 시스템</p>', 
           unsafe_allow_html=True)

# 세션 상태 초기화
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.results = None

# 사이드바 설정
with st.sidebar:
    st.header("📋 모델 정보")
    st.info("""
    **Policy Model**: peter520416/llama1b-MMOA_RAG_Final_cp180  
    **Generator**: meta-llama/Llama-3.1-8B-Instruct  
    **Retrieval**: BM25 + Sentence-BERT  
    **Mode**: 추론 전용 (사전 훈련된 모델)
    """)
    
    st.header("🎯 사용법")
    st.write("""
    1. 질문을 입력하거나 예시 질문 선택
    2. '답변 생성' 버튼 클릭
    3. 4개 탭에서 결과 확인:
       - 📄 최종 답변
       - 🔄 재작성된 질문
       - 📚 선택된 논문
       - 📖 사용된 문서
    """)
    
    st.header("💡 팁")
    st.write("""
    - 영어 질문이 더 정확함
    - 구체적인 질문일수록 좋음
    - 강화학습/딥러닝 분야 특화
    - 첫 실행 시 시간이 걸릴 수 있음
    """)

# 예시 질문들
example_questions = [
    "What is PPO used for in reinforcement learning?",
    "How does attention mechanism work in transformers?",
    "What are the advantages of MAPPO over other RL algorithms?",
    "Explain the difference between actor-critic and Q-learning",
    "What is the role of value function in policy gradient methods?",
    "How does multi-agent reinforcement learning differ from single-agent?",
    "What are the challenges in training large language models?",
    "How does BERT improve upon previous language models?"
]

# 메인 컨테이너
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🤔 질문을 입력하세요")
    
    # 질문 입력
    question = st.text_area(
        "질문:",
        height=100,
        placeholder="예: What is reinforcement learning and how does it work?",
        help="강화학습, 딥러닝, 머신러닝과 관련된 질문을 영어로 입력해주세요."
    )
    
    # 답변 생성 버튼
    generate_button = st.button(
        "🤖 답변 생성",
        type="primary",
        disabled=not question.strip(),
        use_container_width=True
    )

with col2:
    st.subheader("💡 예시 질문")
    
    # 예시 질문 선택
    for i, example in enumerate(example_questions):
        if st.button(f"📌 {example[:50]}...", key=f"example_{i}", use_container_width=True):
            st.session_state.selected_question = example
            # 자동으로 질문 입력란에 설정하기 위해 rerun
            st.rerun()

# 선택된 예시 질문을 질문 입력란에 반영
if 'selected_question' in st.session_state:
    question = st.session_state.selected_question
    # 세션 상태 초기화
    del st.session_state.selected_question
    st.rerun()

# 답변 생성 처리
if generate_button and question.strip():
    # 로딩 상태 표시
    with st.container():
        st.markdown('<div class="status-box status-loading">🔄 모델이 답변을 생성 중입니다... 잠시만 기다려주세요.</div>', 
                   unsafe_allow_html=True)
        
        # 진행 상태 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1단계: 초기화
            progress_bar.progress(10)
            status_text.text("🚀 RAG 파이프라인 초기화 중...")
            time.sleep(0.5)
            
            # 2단계: 추론 실행
            progress_bar.progress(30)
            status_text.text("🔍 질문 분석 및 문서 검색 중...")
            
            # 실제 추론 실행
            results = infer_attention_question(question)
            
            progress_bar.progress(100)
            status_text.text("✅ 답변 생성 완료!")
            time.sleep(1)
            
            # 성공 메시지
            st.markdown('<div class="status-box status-success">✅ 답변이 성공적으로 생성되었습니다!</div>', 
                       unsafe_allow_html=True)
            
            # 결과 저장
            st.session_state.results = results
            st.session_state.current_question = question
            
        except Exception as e:
            progress_bar.progress(100)
            st.markdown(f'<div class="status-box status-error">❌ 오류가 발생했습니다: {str(e)}</div>', 
                       unsafe_allow_html=True)
            st.error("다시 시도해주세요.")

# 결과 표시
if st.session_state.results:
    st.divider()
    st.subheader(f"📊 결과: {st.session_state.current_question}")
    
    # 탭으로 결과 구성
    tab1, tab2, tab3, tab4 = st.tabs(["📄 최종 답변", "🔄 재작성된 질문", "📚 선택된 논문", "📖 사용된 문서"])
    
    results = st.session_state.results
    
    with tab1:
        st.subheader("🤖 AI 답변")
        st.markdown(f'<div class="document-box">{results["final_answer"]}</div>', 
                   unsafe_allow_html=True)
        
        # 답변 품질 피드백
        st.subheader("📝 이 답변이 도움이 되었나요?")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 좋음", use_container_width=True):
                st.success("피드백 감사합니다!")
        with col2:
            if st.button("👎 나쁨", use_container_width=True):
                st.info("더 나은 서비스를 위해 노력하겠습니다.")
        with col3:
            if st.button("🔄 다시 생성", use_container_width=True):
                st.session_state.results = None
                st.rerun()
    
    with tab2:
        st.subheader("🔄 QR 모듈이 재작성한 질문")
        st.markdown(f'<div class="highlight">{results["qr_text"]}</div>', 
                   unsafe_allow_html=True)
        
        st.subheader("📈 비교")
        st.write("**원본 질문:**")
        st.write(st.session_state.current_question)
        st.write("**재작성된 질문:**")
        st.write(results["qr_text"])
    
    with tab3:
        st.subheader("🎯 Selector가 선택한 논문 ID들")
        
        for i, paper_id in enumerate(results["sel_ids"], 1):
            col1, col2 = st.columns([1, 4])
            with col1:
                st.metric(f"논문 {i}", paper_id)
            with col2:
                arxiv_url = f"https://arxiv.org/abs/{paper_id}"
                st.markdown(f"[📄 ArXiv에서 보기]({arxiv_url})")
        
        if len(results["sel_ids"]) == 0:
            st.warning("선택된 논문이 없습니다.")
    
    with tab4:
        st.subheader("📖 최종 답변 생성에 사용된 문서들")
        
        for i, doc in enumerate(results["documents"], 1):
            with st.expander(f"📄 문서 {i}", expanded=(i == 1)):
                st.markdown(f'<div class="document-box">{doc}</div>', 
                           unsafe_allow_html=True)
        
        if len(results["documents"]) == 0:
            st.warning("사용된 문서가 없습니다.")

# 하단 정보
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **🔧 기술 스택**  
    - MAPPO (Multi-Agent PPO)
    - Llama 모델 기반
    - BM25 검색
    - Streamlit UI
    """)

with col2:
    st.success("""
    **✨ 특징**  
    - 사전 훈련된 모델 사용
    - 실시간 논문 검색
    - 4단계 RAG 파이프라인
    - 직관적인 웹 인터페이스
    """)

with col3:
    st.warning("""
    **⚠️ 주의사항**  
    - 첫 실행 시 모델 로딩 시간 소요
    - GPU 메모리 사용량 높음
    - 인터넷 연결 필요
    - HF_TOKEN 설정 필수
    """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; margin-top: 2rem;'>
    <p>🤖 <strong>MAPPO RAG 챗봇</strong> | Made with ❤️ by BOAZ 20기</p>
    <p>사전 훈련된 <code>peter520416/llama1b-MMOA_RAG_Final_cp180</code> 모델 사용</p>
</div>
""", unsafe_allow_html=True) 