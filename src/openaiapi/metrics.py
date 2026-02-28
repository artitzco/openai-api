import copy
import pandas as pd
from typing import List, Dict, Any, Optional


class Metrics:
    """
    Registra y gestiona las métricas de cada solicitud realizada al modelo de OpenAI.

    Cada registro incluye:
    - El diccionario puro de uso de tokens reportado por la API.
    - El nombre del modelo utilizado en esa solicitud.
    - La lista de IDs de nodos activos que se usaron para construir el prompt.
    """

    def __init__(self):
        self._records: List[Dict[str, Any]] = []

    def log(self, usage_dict: Dict[str, Any], model: str, active_node_ids: List[int]) -> None:
        """
        Registra las métricas de una solicitud al modelo.

        Args:
            usage_dict: Diccionario puro de tokens reportado por la API de OpenAI.
            model: Nombre del modelo utilizado en esta solicitud.
            active_node_ids: Lista de IDs de nodos activos al momento de esta solicitud.
        """
        record = {
            "model": model,
            "active_nodes": active_node_ids,
            "usage": usage_dict
        }
        self._records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Transforma los registros de métricas en un DataFrame de pandas.
        Las columnas de uso de tokens se aplanarán en un MultiIndex.
        Las columnas 'model' y 'active_nodes' se mantienen como columnas regulares.
        """
        if not self._records:
            return pd.DataFrame()

        rows = []
        for record in self._records:
            flat_row = {}
            flat_row[("model",)] = record["model"]
            flat_row[("active_nodes",)] = record["active_nodes"]

            # Aplanar el diccionario de uso de tokens recursivamente
            def _flatten(source: dict, parent_key: tuple = ()) -> None:
                for k, v in source.items():
                    new_key = parent_key + (k,)
                    if isinstance(v, dict):
                        _flatten(v, new_key)
                    else:
                        flat_row[new_key] = v

            _flatten(record.get("usage", {}))
            rows.append(flat_row)

        df = pd.DataFrame(rows)

        # Emparejar las tuplas al máximo nivel de profundidad
        max_len = max(len(t) for t in df.columns)
        padded_columns = [
            t + ("",) * (max_len - len(t))
            for t in df.columns
        ]

        df.columns = pd.MultiIndex.from_tuples(padded_columns)
        return df

    def __str__(self) -> str:
        return f"Metrics(total_requests={len(self._records)})"

    def __repr__(self) -> str:
        display(self.to_dataframe())
        return self.__str__()

    def deepcopy(self) -> "Metrics":
        """Devuelve una copia profunda e independiente de las métricas."""
        new_metrics = Metrics()
        new_metrics._records = copy.deepcopy(self._records)
        return new_metrics

    def clear(self) -> None:
        """Reinicia todos los registros de métricas."""
        self._records.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el estado completo de las métricas como un diccionario."""
        return {"records": self._records}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metrics":
        """Crea una instancia de métricas a partir de un diccionario."""
        metrics = cls()
        metrics._records = data["records"]
        return metrics
