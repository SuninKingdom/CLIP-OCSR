import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # Data paths
    dataset_dir: str = ""
    labels_path: str = ""
    output_dir: str = ""

    # CLIP-OCSR model paths
    stage1_ckpt_path: str = ""
    stage2_ckpt_path: str = ""
    tokenizer_path: str = "assets/tokenizer_smiles.json"
    abbrev_group_path: str = "assets/abbrev_group.json"

    # LLM provider: "deepseek" or "mimo"
    llm_provider: str = "deepseek"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = ""
    deepseek_model: str = ""

    # Mimo
    mimo_api_key: str = ""
    mimo_base_url: str = ""
    mimo_model: str = ""

    # MinerU API
    mineru_token: str = ""

    # Cropping thresholds
    text_y_threshold: float = 0.45

    # LLM settings
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    llm_timeout: int = 60

    def __post_init__(self):
        self.dataset_dir = self.dataset_dir or os.getenv("MARKUSH_DATASET_DIR", "")
        self.labels_path = self.labels_path or os.getenv("MARKUSH_LABELS_PATH", "")
        self.output_dir = self.output_dir or os.getenv("MARKUSH_OUTPUT_DIR", "./results")

        self.stage1_ckpt_path = self.stage1_ckpt_path or os.getenv("STAGE1_CKPT_PATH", "")
        self.stage2_ckpt_path = self.stage2_ckpt_path or os.getenv("STAGE2_CKPT_PATH", "")
        self.tokenizer_path = os.getenv("TOKENIZER_PATH", self.tokenizer_path)
        self.abbrev_group_path = os.getenv("ABBREV_GROUP_PATH", self.abbrev_group_path)

        self.deepseek_api_key = self.deepseek_api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.deepseek_base_url = self.deepseek_base_url or os.getenv("DEEPSEEK_BASE_URL", "")
        self.deepseek_model = self.deepseek_model or os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
        self.mimo_api_key = self.mimo_api_key or os.getenv("MIMO_API_KEY", "")
        self.mimo_base_url = self.mimo_base_url or os.getenv("MIMO_BASE_URL", "")
        self.mimo_model = self.mimo_model or os.getenv("MIMO_MODEL", "mimo-v2.5")
        self.mineru_token = self.mineru_token or os.getenv("MINERU_TOKEN", "")

        # Separate output dir per LLM provider
        self.output_dir = os.path.join(self.output_dir, self.llm_provider)

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        return self.mimo_api_key

    @property
    def llm_base_url(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_base_url
        return self.mimo_base_url

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        return self.mimo_model
