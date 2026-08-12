import os
from dataclasses import dataclass
from pathlib import Path
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

    # Local MinerU output directory (pre-computed)
    mineru_output_dir: str = ""

    # Cropping thresholds
    text_y_threshold: float = 0.45

    # LLM settings
    llm_temperature: float = 0.0
    llm_max_retries: int = 3
    llm_timeout: int = 60

    # Optional concrete-product generation. It is disabled by default so the
    # established parsing and evaluation outputs remain unchanged.
    enable_instantiation: bool = False
    use_fragment_library: bool = True
    fragment_library_path: str = ""
    instantiation_max_products: int = 256
    instantiation_max_assignment_attempts: int = 10000
    instantiation_max_position_variants: int = 64
    instantiation_max_frequency_variants: int = 64
    instantiation_max_repeat_count: int = 100
    instantiation_max_candidates_per_value: int = 64

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
        self.mineru_output_dir = self.mineru_output_dir or os.getenv("MINERU_OUTPUT_DIR", "")

        if self.use_fragment_library:
            self.fragment_library_path = (
                self.fragment_library_path
                or os.getenv("MARKUSH_FRAGMENT_LIBRARY", "")
                or str(
                    Path(__file__).resolve().parent
                    / "resources"
                    / "markush_fragment_library.json"
                )
            )
        else:
            self.fragment_library_path = ""

        for field_name in (
            "instantiation_max_products",
            "instantiation_max_assignment_attempts",
            "instantiation_max_position_variants",
            "instantiation_max_frequency_variants",
            "instantiation_max_candidates_per_value",
        ):
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        if int(self.instantiation_max_repeat_count) < 0:
            raise ValueError("instantiation_max_repeat_count cannot be negative")

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
