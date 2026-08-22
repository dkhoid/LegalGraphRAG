import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """Model configuration"""

    model_name: str = "qwen3"
    device: str = "cuda:0"
    prompt_language: str = "en"
    # OpenAI-type model configuration
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    # Generation parameters
    max_length: int = 4096
    temperature: float = 0.1

    def __post_init__(self):
        """Validate model name"""
        valid_models = [
            "qwen3",
            "qwen2_5",
            "gemma3",
            "internlm3",
            "glm4",
            "deepseek_v3",
            "gpt4o_mini",
            "gemini_flash",
            "gemini_flash_lite",
        ]
        if self.model_name not in valid_models:
            raise ValueError(
                f"Invalid model_name: {self.model_name}. Must be one of {valid_models}"
            )
        valid_prompt_languages = ["en", "zh", "cn", "chinese", "english"]
        if self.prompt_language.lower() not in valid_prompt_languages:
            raise ValueError(
                f"Invalid prompt_language: {self.prompt_language}. "
                f"Must be one of {valid_prompt_languages}"
            )


@dataclass
class DataConfig:
    """Data path configuration"""

    case_db_path: str = "./data/clean/cases_clean.json"
    law_to_dispute_path: str = "./data/clean/law_to_dispute_clean.json"
    datasets_path: Optional[str] = None  # Dataset root directory
    output_dir: str = "./data/outputs"

    def __post_init__(self):
        """Create output directory"""
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class RetrieveConfig:
    """Retrieval configuration"""

    top_retrieve: bool = True
    direct_retrieve: bool = True
    augment_retrieve: bool = True
    top_retrieve_top_k: int = 10
    direct_retrieve_top_k: int = 10
    batch_judge: bool = True

    # New advanced pipeline features
    use_reranker: bool = True
    reranker_top_k: int = 8
    use_self_consistent: bool = True
    self_consistent_n: int = 3
    judge_chatbot: str = "deepseek_v3"
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    mmr_top_k: int = 5
    min_rrf_score: float = 0.0
    law_desc_cap: int = 600
    max_judge_laws: int = 8
    max_applicable_laws: int = 8
    expand_abbreviations: bool = True
    rrf_k: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "top_retrieve": self.top_retrieve,
            "direct_retrieve": self.direct_retrieve,
            "augment_retrieve": self.augment_retrieve,
            "top_retrieve_top_k": self.top_retrieve_top_k,
            "direct_retrieve_top_k": self.direct_retrieve_top_k,
            "batch_judge": self.batch_judge,
            "use_reranker": self.use_reranker,
            "reranker_top_k": self.reranker_top_k,
            "use_self_consistent": self.use_self_consistent,
            "self_consistent_n": self.self_consistent_n,
            "judge_chatbot": self.judge_chatbot,
            "use_mmr": self.use_mmr,
            "mmr_lambda": self.mmr_lambda,
            "mmr_top_k": self.mmr_top_k,
            "min_rrf_score": self.min_rrf_score,
            "law_desc_cap": self.law_desc_cap,
            "max_judge_laws": self.max_judge_laws,
            "max_applicable_laws": self.max_applicable_laws,
            "expand_abbreviations": self.expand_abbreviations,
            "rrf_k": self.rrf_k,
        }


@dataclass
class GraphConfig:
    """Graph database configuration"""

    graph_db_path: Optional[str] = None  # Graph database save/load path
    embedding_api_url: str = "http://localhost:11434/api/embed"
    embedding_model: str = "bge-m3"
    auto_save: bool = True  # Whether to auto-save graph database
    auto_build: bool = True  # Whether to auto-build if graph doesn't exist


@dataclass
class LegalGraphRAGConfig:
    """LegalGraphRAG complete configuration"""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    @classmethod
    def from_env_file(cls, dotenv_path: str = ".env") -> "LegalGraphRAGConfig":
        """
        Load configuration from .env file.
        Dataclass field defaults are the single source of truth.
        Only values explicitly defined in the .env file will override them.

        Args:
            dotenv_path: Path to .env file

        Returns:
            LegalGraphRAGConfig instance
        """
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path, override=True)

        defaults = cls()

        def _env_str(key: str, default: str) -> str:
            return os.getenv(key) if os.getenv(key) is not None else default

        def _env_bool(key: str, default: bool) -> bool:
            v = os.getenv(key)
            return v.lower() in ("1", "true", "yes") if v is not None else default

        def _env_int(key: str, default: int) -> int:
            v = os.getenv(key)
            return int(v) if v is not None else default

        def _env_float(key: str, default: float) -> float:
            v = os.getenv(key)
            return float(v) if v is not None else default

        model_config = ModelConfig(
            model_name=_env_str("model_name", defaults.model.model_name),
            device=_env_str("device", defaults.model.device),
            prompt_language=_env_str("prompt_language", defaults.model.prompt_language),
            api_key=os.getenv("api_key", defaults.model.api_key),
            base_url=os.getenv("base_url", defaults.model.base_url),
            max_length=_env_int("max_length", defaults.model.max_length),
            temperature=_env_float("temperature", defaults.model.temperature),
        )

        data_config = DataConfig(
            case_db_path=_env_str("case_db_path", defaults.data.case_db_path),
            law_to_dispute_path=_env_str("law_to_dispute_path", defaults.data.law_to_dispute_path),
            datasets_path=os.getenv("datasets_path", defaults.data.datasets_path),
            output_dir=_env_str("output_dir", defaults.data.output_dir),
        )

        retrieve_config = RetrieveConfig(
            top_retrieve=_env_bool("top_retrieve", defaults.retrieve.top_retrieve),
            direct_retrieve=_env_bool("direct_retrieve", defaults.retrieve.direct_retrieve),
            augment_retrieve=_env_bool("augment_retrieve", defaults.retrieve.augment_retrieve),
            top_retrieve_top_k=_env_int("top_retrieve_top_k", defaults.retrieve.top_retrieve_top_k),
            direct_retrieve_top_k=_env_int(
                "direct_retrieve_top_k", defaults.retrieve.direct_retrieve_top_k
            ),
            batch_judge=_env_bool("batch_judge", defaults.retrieve.batch_judge),
            use_reranker=_env_bool("use_reranker", defaults.retrieve.use_reranker),
            reranker_top_k=_env_int("reranker_top_k", defaults.retrieve.reranker_top_k),
            use_self_consistent=_env_bool(
                "use_self_consistent", defaults.retrieve.use_self_consistent
            ),
            self_consistent_n=_env_int("self_consistent_n", defaults.retrieve.self_consistent_n),
            judge_chatbot=_env_str("judge_chatbot", defaults.retrieve.judge_chatbot),
            use_mmr=_env_bool("use_mmr", defaults.retrieve.use_mmr),
            mmr_lambda=_env_float("mmr_lambda", defaults.retrieve.mmr_lambda),
            mmr_top_k=_env_int("mmr_top_k", defaults.retrieve.mmr_top_k),
            min_rrf_score=_env_float("min_rrf_score", defaults.retrieve.min_rrf_score),
            law_desc_cap=_env_int("law_desc_cap", defaults.retrieve.law_desc_cap),
            max_judge_laws=_env_int("max_judge_laws", defaults.retrieve.max_judge_laws),
            max_applicable_laws=_env_int(
                "max_applicable_laws", defaults.retrieve.max_applicable_laws
            ),
            expand_abbreviations=_env_bool(
                "expand_abbreviations", defaults.retrieve.expand_abbreviations
            ),
            rrf_k=_env_int("rrf_k", defaults.retrieve.rrf_k),
        )

        graph_config = GraphConfig(
            graph_db_path=os.getenv("graph_db_path", defaults.graph.graph_db_path),
            embedding_api_url=_env_str("embedding_api_url", defaults.graph.embedding_api_url),
            embedding_model=_env_str("embedding_model", defaults.graph.embedding_model),
            auto_save=_env_bool("auto_save", defaults.graph.auto_save),
            auto_build=_env_bool("auto_build", defaults.graph.auto_build),
        )

        return cls(
            model=model_config,
            data=data_config,
            retrieve=retrieve_config,
            graph=graph_config,
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LegalGraphRAGConfig":
        """
        Create configuration from dictionary

        Args:
            config_dict: Configuration dictionary

        Returns:
            LegalGraphRAGConfig instance
        """
        model_config = ModelConfig(**config_dict.get("model", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        retrieve_config = RetrieveConfig(**config_dict.get("retrieve", {}))
        graph_config = GraphConfig(**config_dict.get("graph", {}))

        return cls(
            model=model_config,
            data=data_config,
            retrieve=retrieve_config,
            graph=graph_config,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model": {
                "model_name": self.model.model_name,
                "device": self.model.device,
                "prompt_language": self.model.prompt_language,
                "api_key": self.model.api_key,
                "base_url": self.model.base_url,
                "max_length": self.model.max_length,
                "temperature": self.model.temperature,
            },
            "data": {
                "case_db_path": self.data.case_db_path,
                "law_to_dispute_path": self.data.law_to_dispute_path,
                "datasets_path": self.data.datasets_path,
                "output_dir": self.data.output_dir,
            },
            "retrieve": self.retrieve.to_dict(),
            "graph": {
                "graph_db_path": self.graph.graph_db_path,
                "embedding_api_url": self.graph.embedding_api_url,
                "embedding_model": self.graph.embedding_model,
                "auto_save": self.graph.auto_save,
                "auto_build": self.graph.auto_build,
            },
        }

    def save(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "LegalGraphRAGConfig":
        """Load configuration from JSON file"""
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
