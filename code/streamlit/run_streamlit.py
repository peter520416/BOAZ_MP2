"""
스트림릿 앱 실행 스크립트
MAPPO RAG 챗봇을 실행하기 위한 헬퍼 스크립트
사전 훈련된 모델을 사용한 추론 전용

프로젝트 루트(BOAZ_MP2)에서 실행: python ./code/streamlit/run_streamlit.py
"""

import os
import sys
import subprocess
from pathlib import Path

def get_project_root():
    """프로젝트 루트 디렉토리를 찾습니다."""
    # 현재 스크립트의 위치에서 프로젝트 루트 추정
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # code/streamlit -> code -> BOAZ_MP2
    return project_root

def check_requirements():
    """필요한 환경이 설정되어 있는지 확인합니다."""
    
    print("🔍 환경 검사 중...")
    
    project_root = get_project_root()
    streamlit_dir = project_root / "code" / "streamlit"
    
    # 1. 필요한 파일들 존재 확인
    required_files = [
        "streamlit_config.yaml",
        "streamlit_config.py", 
        "rag_pipeline.py",
        "streamlit_app.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not (streamlit_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ 필요한 파일들이 없습니다: {missing_files}")
        print(f"   위치: {streamlit_dir}")
        return False
    
    # 2. 환경변수 확인 (.env 파일이 프로젝트 루트에 있는지)
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  .env 파일이 프로젝트 루트에 없습니다.")
        print(f"   {env_file} 에 HF_TOKEN=your_token_here 형태로 추가하세요.")
        print("   Hugging Face에서 토큰을 발급받아 사용하세요: https://huggingface.co/settings/tokens")
        return False
    
    # 환경변수 로드 (프로젝트 루트 기준)
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    if not os.getenv("HF_TOKEN"):
        print("⚠️  HF_TOKEN 환경변수가 설정되지 않았습니다.")
        print(f"   {env_file} 파일에 HF_TOKEN=your_token_here 형태로 추가하세요.")
        return False
    
    # 3. 데이터 파일 확인
    try:
        # streamlit 디렉토리를 sys.path에 추가
        sys.path.insert(0, str(streamlit_dir))
        from streamlit_config import StreamlitConfig
        
        # 프로젝트 루트 기준으로 config 로드
        config = StreamlitConfig(str(streamlit_dir / "streamlit_config.yaml"))
        
        data_files = [
            project_root / config.METADB_PATH.replace("./", ""),
            project_root / config.GPT4O_DATA_PATH.replace("./", "")
        ]
        
        missing_data_files = []
        for file_path in data_files:
            if not file_path.exists():
                missing_data_files.append(str(file_path))
        
        if missing_data_files:
            print("⚠️  다음 데이터 파일들이 없습니다:")
            for file_path in missing_data_files:
                print(f"   - {file_path}")
            print("   해당 위치에 데이터 파일을 배치하세요.")
            return False
            
    except ImportError as e:
        print(f"⚠️  streamlit_config 모듈을 불러올 수 없습니다: {e}")
        print("   streamlit_config.py 파일이 올바른지 확인하세요.")
        return False
    except Exception as e:
        print(f"⚠️  설정 확인 중 오류: {e}")
        return False
    
    print("✅ 환경 검사 완료!")
    return True

def check_models():
    """모델 접근 가능성을 확인합니다."""
    print("🤖 모델 접근성 확인 중...")
    
    try:
        from transformers import AutoTokenizer
        
        project_root = get_project_root()
        streamlit_dir = project_root / "code" / "streamlit"
        
        # streamlit 디렉토리를 sys.path에 추가
        sys.path.insert(0, str(streamlit_dir))
        from streamlit_config import StreamlitConfig
        
        # 프로젝트 루트 기준으로 config 로드
        config = StreamlitConfig(str(streamlit_dir / "streamlit_config.yaml"))
        
        # Policy 모델 접근 확인
        print(f"🔄 Policy 모델 확인: {config.POLICY_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            config.POLICY_MODEL_NAME,
            use_fast=False
        )
        print("✅ Policy 모델 접근 가능")
        
        # Generator 모델 접근 확인
        print(f"🔄 Generator 모델 확인: {config.GENERATOR_MODEL_NAME}")
        generator_tokenizer = AutoTokenizer.from_pretrained(
            config.GENERATOR_MODEL_NAME,
            use_fast=False
        )
        print("✅ Generator 모델 접근 가능")
        
        print("✅ 모든 모델 접근 가능!")
        return True
        
    except Exception as e:
        print(f"❌ 모델 접근 실패: {e}")
        print("   다음을 확인해주세요:")
        print("   1. 인터넷 연결 상태")
        print("   2. HF_TOKEN의 유효성")
        print("   3. 모델에 대한 접근 권한")
        return False

def install_streamlit():
    """스트림릿이 설치되어 있지 않으면 설치합니다."""
    try:
        import streamlit
        print("✅ Streamlit이 이미 설치되어 있습니다.")
        return True
    except ImportError:
        print("📦 Streamlit을 설치 중...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
            print("✅ Streamlit 설치 완료!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Streamlit 설치 실패!")
            return False

def run_streamlit():
    """스트림릿 앱을 실행합니다."""
    
    print("\n" + "="*50)
    print("🚀 MAPPO RAG 챗봇 시작!")
    print("   사전 훈련된 모델 사용 (추론 전용)")
    print("="*50)
    
    project_root = get_project_root()
    streamlit_dir = project_root / "code" / "streamlit"
    streamlit_app_path = streamlit_dir / "streamlit_app.py"
    
    # 스트림릿 명령어 구성 (streamlit 디렉토리에서 실행)
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(streamlit_app_path),
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--server.headless", "false"
    ]
    
    try:
        print(f"📁 프로젝트 루트: {project_root}")
        print(f"📁 스트림릿 앱: {streamlit_app_path}")
        print("🌐 브라우저에서 http://localhost:8501 을 열어주세요.")
        print("🛑 종료하려면 Ctrl+C를 눌러주세요.")
        print("\n📝 첫 실행 시 모델 로딩에 시간이 걸릴 수 있습니다...")
        print("💾 모델이 로드되면 캐시되어 이후 실행이 빨라집니다.\n")
        
        # 작업 디렉토리를 streamlit 폴더로 설정하여 실행
        subprocess.run(cmd, cwd=str(streamlit_dir))
        
    except KeyboardInterrupt:
        print("\n👋 챗봇이 종료되었습니다.")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")

def main():
    """메인 실행 함수"""
    
    project_root = get_project_root()
    
    print("🤖 MAPPO RAG 챗봇 시작 준비")
    print("📋 사전 훈련된 모델 사용 (peter520416/llama1b-MMOA_RAG_Final_cp180)")
    print(f"📁 프로젝트 루트: {project_root}")
    print("-" * 60)
    
    # 프로젝트 루트에서 실행되고 있는지 확인
    current_dir = Path.cwd()
    if current_dir.name != "BOAZ_MP2":
        print("⚠️  이 스크립트는 BOAZ_MP2 프로젝트 루트에서 실행해야 합니다.")
        print(f"   현재 위치: {current_dir}")
        print("   올바른 실행 방법:")
        print("   cd BOAZ_MP2")
        print("   python ./code/streamlit/run_streamlit.py")
        return
    
    # 1. 환경 검사
    if not check_requirements():
        print("\n❌ 환경 설정이 완료되지 않았습니다.")
        print("문제를 해결한 후 다시 실행해주세요.")
        return
    
    # 2. Streamlit 설치 확인
    if not install_streamlit():
        print("\n❌ Streamlit 설치에 실패했습니다.")
        return
    
    # 3. 모델 접근성 확인 (선택적 - 실패해도 진행)
    print("\n" + "="*30)
    if not check_models():
        print("⚠️  모델 접근성 확인에 실패했지만 앱을 시작합니다.")
        print("   실제 실행 중에 모델 로딩이 실패할 수 있습니다.")
        response = input("계속 진행하시겠습니까? (y/N): ")
        if response.lower() != 'y':
            print("👋 실행이 취소되었습니다.")
            return
    
    # 4. 앱 실행
    run_streamlit()

if __name__ == "__main__":
    main() 