import streamlit as st
from rag_pipeline import infer_attention_question

st.set_page_config(page_title="MMOA-RAG 챗봇", layout="wide")
st.title("MMOA-RAG 논문 챗봇")

question = st.text_input("질문을 입력하세요", placeholder="예: What is PPO used for in reinforcement learning?")

if st.button("질문하기") and question:
    with st.spinner("문서를 검색하고 응답을 생성 중입니다..."):
        result = infer_attention_question(question)

        st.subheader("Rewritten Question")
        st.markdown(f"> {result['qr_text']}")

        st.subheader("Selected Document IDs")
        st.code(", ".join(result["sel_ids"]))

        st.subheader("📄 Final Answer")
        st.markdown(result["final_answer"])

        with st.expander("🔍 Used Documents"):
            for doc in result["documents"]:
                st.markdown(doc[:500] + "...") 