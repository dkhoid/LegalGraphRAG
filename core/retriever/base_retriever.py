from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
from core.models import BaseModel


class BaseRetriever(ABC):
    """Base abstract class for all retrievers"""

    def __init__(self, model: BaseModel):
        self.model = model

    @abstractmethod
    def retrieve(
        self,
        case: Dict[str, Any],
        law_to_dispute: List[Dict[str, Any]],
        cases_db: List[Dict[str, Any]],
        retrieve_config: Dict[str, Any] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Retrieve similar facts and laws for a given case.

        Args:
            case: The case features dictionary (contains feature text, name, etc.)
            law_to_dispute: The mapping of laws to disputes
            cases_db: The full database of cases
            retrieve_config: Optional configuration for retrieval

        Returns:
            Tuple of:
            - original_retrieved_res: Raw retrieval metrics/debug data
            - final_retrieved_laws: List of retrieved laws
            - retrieved_facts: List of retrieved case facts
        """
        pass
