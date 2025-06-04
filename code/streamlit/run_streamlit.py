"""
스트림릿 앱 실행 스크립트
MAPPO RAG 챗봇을 실행하기 위한 헬퍼 스크립트
사전 훈련된 모델을 사용한 추론 전용

프로젝트 루트(BOAZ_MP2)에서 실행: python ./code/streamlit/run_streamlit.py
코랩에서 실행: !python code/streamlit/run_streamlit.py
"""

import os
import sys
import subprocess
from pathlib import Path

def is_colab():
    """Google Colab 환경인지 확인합니다."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

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

def get_external_ip():
    """코랩에서 외부 IP 주소를 확인합니다."""
    try:
        import subprocess
        result = subprocess.run(['wget', '-q', '-O', '-', 'ipv4.icanhazip.com'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            ip = result.stdout.strip()
            print(f"🌐 외부 IP 주소: {ip}")
            print(f"💡 이 IP를 Tunnel Password로 사용하세요: {ip}")
            return ip
        else:
            print("⚠️  IP 주소 확인 실패")
            return None
    except Exception as e:
        print(f"⚠️  IP 주소 확인 중 오류: {e}")
        return None

def install_localtunnel():
    """localtunnel을 설치합니다 (npm 필요)."""
    try:
        # npm이 설치되어 있는지 확인
        subprocess.run(['npm', '--version'], capture_output=True, check=True)
        print("✅ npm이 이미 설치되어 있습니다.")
        
        # localtunnel 설치
        print("📦 localtunnel 설치 중...")
        subprocess.run(['npm', 'install', '-g', 'localtunnel'], check=True)
        print("✅ localtunnel 설치 완료!")
        return True
    except subprocess.CalledProcessError:
        print("❌ localtunnel 설치 실패!")
        print("💡 Node.js와 npm이 설치되어 있는지 확인하세요.")
        return False
    except FileNotFoundError:
        print("❌ npm이 설치되어 있지 않습니다.")
        print("💡 Node.js를 설치하면 npm도 함께 설치됩니다.")
        return False

def run_streamlit_with_localtunnel(streamlit_app_path, project_root):
    """localtunnel을 사용하여 Streamlit을 실행합니다."""
    import threading
    import time
    import sys
    import select
    
    # 외부 IP 확인
    external_ip = get_external_ip()
    
    print("\n🚀 Streamlit 앱을 시작합니다...")
    print("⏳ 앱이 시작될 때까지 잠시 기다려주세요...")
    
    # Streamlit 프로세스 시작
    streamlit_cmd = [
        sys.executable, "-m", "streamlit", "run", str(streamlit_app_path),
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    def run_streamlit():
        subprocess.run(streamlit_cmd, cwd=str(project_root / "code" / "streamlit"))
    
    # Streamlit을 별도 스레드에서 실행
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Streamlit이 시작될 때까지 대기
    print("🔄 Streamlit 서버 시작 대기 중...")
    time.sleep(10)  # 10초 대기
    
    # localtunnel 실행
    print("🌐 localtunnel로 외부 접속 URL 생성 중...")
    try:
        localtunnel_process = subprocess.Popen(
            ['npx', 'localtunnel', '--port', '8501'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # localtunnel 출력 모니터링
        print("🌍 외부 접속 URL이 생성되었습니다!")
        print("📋 접속 방법:")
        print("   1. 아래에 표시되는 URL을 클릭하세요")
        print("   2. 'Tunnel Password' 입력란에 위에서 확인한 IP 주소를 입력하세요")
        if external_ip:
            print(f"   3. Tunnel Password: {external_ip}")
        print("\n" + "="*60)
        
        try:
            # 실시간으로 localtunnel 출력 표시
            while True:
                output = localtunnel_process.stdout.readline()
                if output:
                    print("🌐", output.strip())
                    if "your url is:" in output.lower():
                        # URL 추출 및 강조 표시
                        url_line = output.strip()
                        print("🎉 " + "="*50)
                        print(f"✨ 접속 URL: {url_line}")
                        print("🎉 " + "="*50)
                
                # 프로세스가 종료되었는지 확인
                if localtunnel_process.poll() is not None:
                    break
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n👋 사용자가 중단했습니다.")
        finally:
            print("\n🛑 localtunnel 프로세스를 종료합니다...")
            localtunnel_process.terminate()
            localtunnel_process.wait()
            
    except Exception as e:
        print(f"❌ localtunnel 실행 실패: {e}")
        print("💡 수동으로 다음 명령어를 실행해보세요:")
        print("   !streamlit run app.py & npx localtunnel --port 8501")

def run_streamlit():
    """스트림릿 앱을 실행합니다."""
    
    print("\n" + "="*50)
    print("🚀 MAPPO RAG 챗봇 시작!")
    print("   사전 훈련된 모델 사용 (추론 전용)")
    if is_colab():
        print("   🔬 Google Colab 환경 감지됨")
    print("="*50)
    
    project_root = get_project_root()
    streamlit_dir = project_root / "code" / "streamlit"
    streamlit_app_path = streamlit_dir / "streamlit_app.py"
    
    # 스트림릿 명령어 구성
    if is_colab():
        print("🔬 Google Colab 환경에서 localtunnel 방식으로 실행합니다...")
        print("\n📋 실행 순서:")
        print("   1. Streamlit 앱 시작")
        print("   2. 외부 IP 주소 확인")
        print("   3. localtunnel로 외부 접속 URL 생성")
        print("   4. 브라우저에서 접속")
        
        # localtunnel 설치 확인
        if not install_localtunnel():
            print("\n❌ localtunnel 설치에 실패했습니다.")
            print("💡 대안: 수동으로 다음 명령어들을 순서대로 실행하세요:")
            print("   1. !streamlit run code/streamlit/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &")
            print("   2. !wget -q -O - ipv4.icanhazip.com  # IP 주소 확인")
            print("   3. !npx localtunnel --port 8501  # 외부 URL 생성")
            return
        
        # localtunnel 방식으로 실행
        run_streamlit_with_localtunnel(streamlit_app_path, project_root)
        
    else:
        # 로컬 환경용 설정
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(streamlit_app_path),
            "--server.address", "0.0.0.0",
            "--server.port", "8501",
            "--server.headless", "false"
        ]
        
        print("🌐 브라우저에서 http://localhost:8501 을 열어주세요.")
        print("🛑 종료하려면 Ctrl+C를 눌러주세요.")
        print("\n📝 첫 실행 시 모델 로딩에 시간이 걸릴 수 있습니다...")
        print("💾 모델이 로드되면 캐시되어 이후 실행이 빨라집니다.\n")
        
        try:
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
    if is_colab():
        print("🔬 Google Colab 환경에서 실행 중")
        print("🌐 localtunnel 방식으로 외부 접속을 지원합니다")
    print("-" * 60)
    
    # 프로젝트 루트에서 실행되고 있는지 확인 (로컬 환경만)
    if not is_colab():
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
    
    # 3. 코랩용 pyngrok 설치 제거 (localtunnel 사용)
    # install_pyngrok_if_needed() 호출 제거
    
    # 4. 모델 접근성 확인 (선택적 - 실패해도 진행)
    print("\n" + "="*30)
    if not check_models():
        print("⚠️  모델 접근성 확인에 실패했지만 앱을 시작합니다.")
        print("   실제 실행 중에 모델 로딩이 실패할 수 있습니다.")
        if not is_colab():
            response = input("계속 진행하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                print("👋 실행이 취소되었습니다.")
                return
        else:
            print("   코랩에서 자동으로 진행합니다...")
    
    # 5. 앱 실행
    run_streamlit()

if __name__ == "__main__":
    main() 