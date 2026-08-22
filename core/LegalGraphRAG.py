"""LegalGraphRAG main class"""

import os
import json
import uuid

from typing import Optional, Dict, Any, List
from tqdm import tqdm

from core.models import BaseModel
from core.utils.util import analyze_case
from core.graph_construct.graph_db import GraphDBManager
from core.utils.logger import logger
from core.utils.formatting import concat_feature_descriptions


from core.config import LegalGraphRAGConfig


class LegalGraphRAG:
    """LegalGraphRAG main class"""

    def __init__(self, config: Optional[LegalGraphRAGConfig] = None):
        """
        Initialize LegalGraphRAG

        Args:
            config: Configuration object, if None use default configuration
        """
        self.config = config or LegalGraphRAGConfig()
        from core.prompt import set_prompt_language

        set_prompt_language(self.config.model.prompt_language)

        # Initialize model
        self.model = self._init_model()

        # Internal storage for lazy-loaded datasets
        self._cases_db = None
        self._law_to_dispute = None

        # Initialize graph database connection (Neo4j)
        from core.graph_construct.neo4j_manager import neo4j_manager

        if not neo4j_manager.driver:
            logger.warning("Neo4j driver is not initialized. Vector search may fail.")
        else:
            logger.info("Using Neo4j for Graph RAG retrieval.")

    def _init_model(self) -> BaseModel:
        """Initialize model"""
        from core.models import (
            QwenChatbot,
            Qwen2Chatbot,
            GemmaChatbot,
            InternlmChatbot,
            GlmChatbot,
            DeepSeekChatbot,
            GPT4OMiniChatbot,
            GeminiChatbot,
        )

        model_map = {
            "qwen3": QwenChatbot,
            "qwen2_5": Qwen2Chatbot,
            "gemma3": GemmaChatbot,
            "internlm3": InternlmChatbot,
            "glm4": GlmChatbot,
            "deepseek_v3": DeepSeekChatbot,
            "gpt4o_mini": GPT4OMiniChatbot,
            "gemini_flash": GeminiChatbot,
            "gemini_flash_lite": GeminiChatbot,
        }

        # Gemini model name mapping (SDK model IDs)
        gemini_model_ids = {
            "gemini_flash": "gemini-2.0-flash",
            "gemini_flash_lite": "gemini-2.0-flash-lite",
        }

        model_class = model_map[self.config.model.model_name]

        # Gemini uses its own SDK, not OpenAI-compatible
        if self.config.model.model_name in gemini_model_ids:
            return GeminiChatbot(
                model_name=gemini_model_ids[self.config.model.model_name],
                device=self.config.model.device,
                api_key=self.config.model.api_key,
            )

        # OpenAI-type models need special handling
        if self.config.model.model_name in ["deepseek_v3", "gpt4o_mini"]:
            # OpenAI-type models need model_name, api_key, base_url
            init_kwargs = {
                "device": self.config.model.device,
            }
            if self.config.model.api_key:
                init_kwargs["api_key"] = self.config.model.api_key
            if self.config.model.base_url:
                init_kwargs["base_url"] = self.config.model.base_url
            return model_class(**init_kwargs)
        else:
            # Transformers-type models only need device
            return model_class(device=self.config.model.device)

    def _load_cases_db(self) -> List[Dict[str, Any]]:
        """Load case database"""
        if not os.path.exists(self.config.data.case_db_path):
            raise FileNotFoundError(f"Case database not found: {self.config.data.case_db_path}")
        with open(self.config.data.case_db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_law_to_dispute(self) -> List[Dict[str, Any]]:
        """Load law to crime mapping"""
        if not os.path.exists(self.config.data.law_to_dispute_path):
            raise FileNotFoundError(
                f"Law to crime mapping not found: {self.config.data.law_to_dispute_path}"
            )
        with open(self.config.data.law_to_dispute_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @property
    def cases_db(self) -> List[Dict[str, Any]]:
        """Lazy-loaded case database"""
        if self._cases_db is None:
            self._cases_db = self._load_cases_db()
        return self._cases_db

    @cases_db.setter
    def cases_db(self, value: Optional[List[Dict[str, Any]]]):
        self._cases_db = value

    @property
    def law_to_dispute(self) -> List[Dict[str, Any]]:
        """Lazy-loaded law to dispute mapping"""
        if self._law_to_dispute is None:
            self._law_to_dispute = self._load_law_to_dispute()
        return self._law_to_dispute

    @law_to_dispute.setter
    def law_to_dispute(self, value: Optional[List[Dict[str, Any]]]):
        self._law_to_dispute = value

    def analyze_case(self, case: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze a single case

        Args:
            case: Case dictionary containing "fact" and "name" fields

        Returns:
            List of analysis results, each element corresponds to a defendant's analysis result
        """
        retrieve_config = self.config.retrieve.to_dict()
        return analyze_case(self.model, case, self.law_to_dispute, self.cases_db, retrieve_config)

    def analyze_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch analyze cases

        Args:
            cases: List of cases

        Returns:
            List of analysis results
        """
        results = []
        for case in cases:
            case_result = self.analyze_case(case)
            results.append(
                {
                    "case_id": case.get("id"),
                    "fact": case.get("fact"),
                    "analysis": case_result,
                }
            )
        return results

    def save_graph_db(self, filepath: Optional[str] = None):
        """Save graph database"""
        save_path = filepath or self.config.graph.graph_db_path
        if not save_path:
            raise ValueError("Graph database path not specified")
        GraphDBManager.save(save_path)
        logger.info(f"Graph database saved to {save_path}")

    def load_graph_db(self, filepath: Optional[str] = None):
        """Load graph database"""
        load_path = filepath or self.config.graph.graph_db_path
        if not load_path:
            raise ValueError("Graph database path not specified")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Graph database not found: {load_path}")
        GraphDBManager.load(load_path)
        logger.info(f"Graph database loaded from {load_path}")

    def _prepare_nodes_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Prepare node data

        Returns:
            Dictionary containing 'case', 'law', 'dispute' keys
        """
        case_nodes_data = []
        law_nodes_data = []
        dispute_nodes_data = []

        # Process case nodes
        for case in tqdm(self.cases_db, desc="Preparing case nodes"):
            # Use 'fact' as the primary description to ensure distinct embeddings
            description = case.get("fact", "")
            if not description and "features" in case:
                description = concat_feature_descriptions(case["features"])

            # Normalize law field
            laws = case.get("law", case.get("laws", []))

            case_nodes_data.append(
                {
                    "id": str(uuid.uuid4()),
                    "description": description,
                    "caseId": case.get("id", ""),
                    "dispute": case.get("dispute", []),
                    "law": laws,
                    "type": "case",
                }
            )

        # Process law nodes and crime nodes
        crimes = set()
        for law in tqdm(self.law_to_dispute, desc="Preparing law and crime nodes"):
            text_id = law.get("id")

            # Process items field
            if "items" in law:
                for item in law["items"]:
                    # Collect crimes
                    if "dispute" in item:
                        if isinstance(item["dispute"], list):
                            crimes.update(item["dispute"])
                        else:
                            crimes.add(item["dispute"])

                    # Create law node
                    law_nodes_data.append(
                        {
                            "id": str(uuid.uuid4()),
                            "entry": text_id,
                            "description": item.get("text", ""),
                            "disputes": item.get("dispute", []),
                            "judge_dep": item.get("judge_dep", []),
                            "related_laws": item.get("related_laws", []),
                            "type": "law",
                        }
                    )
            else:
                # If no items field, process law object directly
                if "dispute" in law:
                    if isinstance(law["dispute"], list):
                        crimes.update(law["dispute"])
                    else:
                        crimes.add(law["dispute"])

                law_nodes_data.append(
                    {
                        "id": str(uuid.uuid4()),
                        "entry": text_id,
                        "description": law.get("text", law.get("description", "")),
                        "disputes": law.get("dispute", []),
                        "judge_dep": law.get("judge_dep", []),
                        "related_laws": law.get("related_laws", []),
                        "type": "law",
                    }
                )

        # Create crime nodes
        crimes = list(crimes)
        for crime in crimes:
            if crime and crime != "":
                dispute_nodes_data.append(
                    {"id": str(uuid.uuid4()), "description": crime, "type": "dispute"}
                )

        return {
            "case": case_nodes_data,
            "law": law_nodes_data,
            "dispute": dispute_nodes_data,
        }

    def build_graph(self, force_rebuild: bool = False):
        """
        Build graph structure

        Args:
            force_rebuild: If True, rebuild even if graph database already exists
        """
        from core.graph_construct.graph_builder import construct_feature_graph

        # Check if graph database already exists
        if (
            not force_rebuild
            and self.config.graph.graph_db_path
            and os.path.exists(self.config.graph.graph_db_path)
        ):
            logger.info(f"Graph database already exists at {self.config.graph.graph_db_path}")
            logger.info("Use force_rebuild=True to rebuild the graph")
            return

        logger.info("Starting graph construction...")

        # Prepare node data
        nodes_data = self._prepare_nodes_data()

        logger.info(
            f"Prepared {len(nodes_data['case'])} case nodes, "
            f"{len(nodes_data['law'])} law nodes, "
            f"{len(nodes_data['dispute'])} crime nodes"
        )

        # Build graph structure
        construct_feature_graph(self.model, nodes_data)

        # Save graph database
        if self.config.graph.graph_db_path:
            self.save_graph_db()
            logger.info(
                f"Graph construction completed and saved to {self.config.graph.graph_db_path}"
            )
        else:
            logger.info("Graph construction completed (not saved, graph_db_path not specified)")

    def __del__(self):
        """Destructor, auto-save graph database"""
        try:
            import sys

            if sys is None or getattr(sys, "meta_path", None) is None:
                return
            if hasattr(self, "config") and self.config and getattr(self.config, "graph", None):
                if self.config.graph.auto_save and self.config.graph.graph_db_path:
                    self.save_graph_db()
        except Exception:
            pass
