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
    top_retrieve_top_k: int = 5
    direct_retrieve_top_k: int = 5
    batch_judge: bool = True

    # New advanced pipeline features
    use_reranker: bool = False
    reranker_top_k: int = 8
    use_self_consistent: bool = False
    self_consistent_n: int = 3
    judge_chatbot: str = "gemini_flash_lite"
    use_mmr: bool = False
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
        Load configuration from .env file

        Args:
            dotenv_path: Path to .env file

        Returns:
            LegalGraphRAGConfig instance
        """
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=dotenv_path)

        # Model configuration
        model_config = ModelConfig(
            model_name=os.getenv("model_name", "qwen3"),
            device=os.getenv("device", "cuda:0"),
            prompt_language=os.getenv("prompt_language", "en"),
            api_key=os.getenv("api_key"),
            base_url=os.getenv("base_url"),
            max_length=int(os.getenv("max_length", 4096)),
            temperature=float(os.getenv("temperature", 0.1)),
        )

        # Data configuration
        data_config = DataConfig(
            case_db_path=os.getenv("case_db_path", "./data/clean/cases_clean.json"),
            law_to_dispute_path=os.getenv(
                "law_to_dispute_path", "./data/clean/law_to_dispute_clean.json"
            ),
            datasets_path=os.getenv("datasets_path"),
            output_dir=os.getenv("output_dir", "./data/outputs"),
        )

        # Retrieval configuration
        retrieve_config = RetrieveConfig(
            top_retrieve=os.getenv("top_retrieve", "True") == "True",
            direct_retrieve=os.getenv("direct_retrieve", "True") == "True",
            augment_retrieve=os.getenv("augment_retrieve", "True") == "True",
            top_retrieve_top_k=int(os.getenv("top_retrieve_top_k", 5)),
            direct_retrieve_top_k=int(os.getenv("direct_retrieve_top_k", 5)),
            batch_judge=os.getenv("batch_judge", "True") == "True",
            use_reranker=os.getenv("use_reranker", "False") == "True",
            reranker_top_k=int(os.getenv("reranker_top_k", 8)),
            use_self_consistent=os.getenv("use_self_consistent", "False") == "True",
            self_consistent_n=int(os.getenv("self_consistent_n", 3)),
            judge_chatbot=os.getenv("judge_chatbot", "gemini_flash_lite"),
            use_mmr=os.getenv("use_mmr", "False") == "True",
            mmr_lambda=float(os.getenv("mmr_lambda", 0.5)),
            mmr_top_k=int(os.getenv("mmr_top_k", 5)),
            min_rrf_score=float(os.getenv("min_rrf_score", 0.0)),
            law_desc_cap=int(os.getenv("law_desc_cap", 600)),
            max_judge_laws=int(os.getenv("max_judge_laws", 8)),
            max_applicable_laws=int(os.getenv("max_applicable_laws", 8)),
            expand_abbreviations=os.getenv("expand_abbreviations", "True") == "True",
            rrf_k=int(os.getenv("rrf_k", 60)),
        )

        # Graph configuration
        graph_config = GraphConfig(
            graph_db_path=os.getenv("graph_db_path"),
            embedding_api_url=os.getenv("embedding_api_url", "http://localhost:11434/api/embed"),
            embedding_model=os.getenv("embedding_model", "bge-m3"),
            auto_save=os.getenv("auto_save", "True") == "True",
            auto_build=os.getenv("auto_build", "True") == "True",
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
