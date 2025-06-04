"""
MAPPO RAG 모델 설정 관리자
YAML 파일에서 설정을 읽어와서 클래스 속성으로 제공.
"""

import os
import yaml
import torch
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(".env")

class Config:
    def __init__(self, config_path="code/config.yaml"):
        """YAML 설정 파일을 로드하여 클래스 속성으로 설정."""
        
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
        self.LOCAL_REPO_DIR = paths['local_repo_dir']
        self.LOG_OUTPUT_PATH = paths['log_output_path']
        self.REPO_ID = paths['repo_id']
        
        # ============================================================================
        # 하이퍼파라미터 설정
        # ============================================================================
        hyperparams = self.config['hyperparameters']
        self.GAMMA = hyperparams['gamma']
        self.EPSILON = hyperparams['epsilon']
        self.C1 = hyperparams['c1']
        self.EPOCHS = hyperparams['epochs']
        self.BATCH_SIZE = hyperparams['batch_size']
        self.POLICY_LEARNING_RATE = hyperparams['policy_learning_rate']
        self.CRITIC_LEARNING_RATE = hyperparams['critic_learning_rate']
        self.GRAD_CLIP_NORM = hyperparams['grad_clip_norm']
        
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
        # 체크포인트 설정
        # ============================================================================
        checkpoint = self.config['checkpoint']
        self.CHECKPOINT_SAVE_INTERVAL = checkpoint['save_interval']
        
        # ============================================================================
        # 시스템 설정
        # ============================================================================
        system = self.config['system']
        self.CUDA_LAUNCH_BLOCKING = system['cuda_launch_blocking']
        
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
        corpus_dir = os.path.dirname(self.CORPUS_INPUT_PATH)
        if corpus_dir and not os.path.exists(corpus_dir):
            print(f"⚠️  코퍼스 디렉토리가 존재하지 않습니다: {corpus_dir}")
        
        data_dir = os.path.dirname(self.GPT4O_DATA_PATH)
        if data_dir and not os.path.exists(data_dir):
            print(f"⚠️  데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        
        print("✓ 설정 검증 완료")
    
    def print_config(self):
        """현재 설정값들을 출력."""
        print("=" * 60)
        print("MAPPO RAG 설정 정보")
        print("=" * 60)
        print(f"Policy Model: {self.POLICY_MODEL_NAME}")
        print(f"Generator Model: {self.GENERATOR_MODEL_NAME}")
        print(f"Device: {self.DEVICE}")
        print(f"Epochs: {self.EPOCHS}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Learning Rate (Policy): {self.POLICY_LEARNING_RATE}")
        print(f"Learning Rate (Critic): {self.CRITIC_LEARNING_RATE}")
        print(f"K Retrieve: {self.K_RETRIEVE}")
        print(f"K Select: {self.K_SELECT}")
        print("=" * 60)
    
    def get_config_dict(self):
        """전체 설정을 딕셔너리로 반환."""
        return self.config
    
    def update_config(self, key_path, value):
        """
        설정값을 업데이트.
        key_path: 점으로 구분된 키 경로 (예: "hyperparameters.learning_rate")
        value: 새로운 값
        """
        keys = key_path.split('.')
        config = self.config
        
        # 마지막 키를 제외하고 경로를 따라가기
        for key in keys[:-1]:
            config = config[key]
        
        # 마지막 키의 값 업데이트
        config[keys[-1]] = value
        
        # 속성 다시 설정
        self._setup_attributes()
        print(f"✓ 설정 업데이트: {key_path} = {value}")

# 전역 Config 인스턴스 생성
Config = Config() 