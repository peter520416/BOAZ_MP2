"""
스트림릿 전용 설정 관리자
추론 전용 YAML 파일에서 설정을 읽어와서 클래스 속성으로 제공
"""

import os
import yaml
import torch
from pathlib import Path
from dotenv import load_dotenv

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

# 프로젝트 루트에서 환경 변수 로드
project_root = find_project_root()
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()  # 기본 위치에서 로드

class StreamlitConfig:
    def __init__(self, config_path="streamlit_config.yaml"):
        """스트림릿 전용 YAML 설정 파일을 로드하여 클래스 속성으로 설정."""
        
        # 설정 파일 경로 처리 (절대경로가 아닌 경우 현재 디렉토리 기준)
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), config_path)
        
        # YAML 파일 로드
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # 환경 변수에서 HF_TOKEN 로드
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        
        # 설정값들을 클래스 속성으로 매핑
        self._setup_attributes()
    
    def _setup_attributes(self):
        """YAML 설정값들을 클래스 속성으로 매핑."""
        
        # ============================================================================
        # 모델 설정
        # ============================================================================
        models = self.config['models']
        self.POLICY_MODEL_NAME = models['policy_model_name']
        self.GENERATOR_MODEL_NAME = models['generator_model_name']
        self.SBERT_MODEL_NAME = models['sbert_model_name']
        
        # 디바이스 설정
        if models['device'] == "auto":
            self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.DEVICE = models['device']
        
        # ============================================================================
        # 파일 경로 설정
        # ============================================================================
        paths = self.config['paths']
        self.CORPUS_INPUT_PATH = paths['MetaDB_path']
        self.METADB_PATH = paths['MetaDB_path']  # 별칭
        self.GPT4O_DATA_PATH = paths['gpt4o_data_path']
        self.LOG_OUTPUT_PATH = paths['log_output_path']
        
        # ============================================================================
        # RAG 모델 설정
        # ============================================================================
        rag = self.config['rag']
        self.K_RETRIEVE = rag['k_retrieve']
        self.K_SELECT = rag['k_select']
        
        # 생성 관련 설정
        generation = rag['generation']
        self.MAX_NEW_TOKENS_QR = generation['max_new_tokens_qr']
        self.MAX_NEW_TOKENS_GEN = generation['max_new_tokens_gen']
        self.MAX_INPUT_LENGTH = generation['max_input_length']
        self.MAX_SEL_INPUT_LENGTH = generation['max_sel_input_length']
        self.MAX_GEN_INPUT_LENGTH = generation['max_gen_input_length']
        self.GEN_BATCH_SIZE = generation['gen_batch_size']
        self.TEMPERATURE = generation['temperature']
        
        # ============================================================================
        # 시스템 설정
        # ============================================================================
        system = self.config['system']
        self.CUDA_LAUNCH_BLOCKING = system['cuda_launch_blocking']
        self.LOAD_IN_8BIT = system['load_in_8bit']
        self.LOW_CPU_MEM_USAGE = system['low_cpu_mem_usage']
        
        # ============================================================================
        # 프롬프트 템플릿
        # ============================================================================
        prompts = self.config['prompts']
        
        # QR 프롬프트
        qr_prompts = prompts['qr']
        self.QR_SYSTEM_PROMPT = qr_prompts['system_prompt']
        self.QR_INSTRUCTION = qr_prompts['instruction']
        
        # Selector 프롬프트
        sel_prompts = prompts['selector']
        self.SEL_SYSTEM_PROMPT = sel_prompts['system_prompt']
        self.SEL_INSTRUCTION = sel_prompts['instruction']
        
        # Generator 프롬프트
        gen_prompts = prompts['generator']
        self.DEFAULT_SYSTEM = gen_prompts['default_system']
        
        # ============================================================================
        # 실패 감지 문구
        # ============================================================================
        self.FAILURE_PHRASES = self.config['failure_phrases']
    
    def validate(self):
        """설정값들의 유효성을 검증."""
        assert self.HF_TOKEN, "HF_TOKEN 환경변수가 설정되지 않았습니다."
        
        # 파일 경로 검증
        if not os.path.exists(self.METADB_PATH):
            print(f"⚠️  MetaDB 파일이 존재하지 않습니다: {self.METADB_PATH}")
        
        if not os.path.exists(self.GPT4O_DATA_PATH):
            print(f"⚠️  GPT4O 데이터 파일이 존재하지 않습니다: {self.GPT4O_DATA_PATH}")
        
        print("✓ 스트림릿 설정 검증 완료")
    
    def print_config(self):
        """현재 설정값들을 출력."""
        print("=" * 60)
        print("MAPPO RAG 스트림릿 설정 정보")
        print("=" * 60)
        print(f"Policy Model: {self.POLICY_MODEL_NAME}")
        print(f"Generator Model: {self.GENERATOR_MODEL_NAME}")
        print(f"Device: {self.DEVICE}")
        print(f"K Retrieve: {self.K_RETRIEVE}")
        print(f"K Select: {self.K_SELECT}")
        print(f"8bit Loading: {self.LOAD_IN_8BIT}")
        print(f"Low CPU Memory: {self.LOW_CPU_MEM_USAGE}")
        print("=" * 60)

# 전역 StreamlitConfig 인스턴스 생성
Config = StreamlitConfig() 