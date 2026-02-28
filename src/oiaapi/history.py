import copy
import pandas as pd
from typing import List, Dict, Optional, Any, Union


class ConversationHistory:
    """
    Gestiona el historial de la conversación con un sistema de nodos identificados por ID.

    Reglas:
    - Cada nodo tiene un ID incremental.
    - El content del rol 'user' y el 'assistant' que le proceda comparten el mismo ID.
    - Cada cambio de 'system' prompt recibe su propio ID.
    - Los nodos se pueden activar o desactivar por ID (user/assistant siempre en par).
    - Solo puede haber a lo más un nodo 'system' activo a la vez.
    """

    def __init__(self):
        self._nodes: List[Dict[str, Any]] = []
        self._next_id: int = 0

    def _new_id(self) -> int:
        """Genera y devuelve el siguiente ID disponible."""
        current = self._next_id
        self._next_id += 1
        return current

    # ------------------------------------------------------------------ #
    #  Registro de nodos                                                   #
    # ------------------------------------------------------------------ #

    def add_system(self, content: str) -> int:
        """
        Registra un nuevo nodo de sistema con su propio ID.
        Desactiva cualquier nodo de sistema previamente activo.
        """
        # Desactivar cualquier system anterior activo
        for node in self._nodes:
            if node["role"] == "system" and node["active"]:
                node["active"] = False

        node_id = self._new_id()
        self._nodes.append({
            "id": node_id,
            "role": "system",
            "content": content,
            "active": True
        })
        return node_id

    def add_user(self, content: Union[str, List[Dict[str, Any]]]) -> int:
        """
        Registra un nuevo nodo de usuario con un nuevo ID.
        Se mantiene inactivo hasta que su assistant correspondiente sea registrado.
        """
        node_id = self._new_id()
        self._nodes.append({
            "id": node_id,
            "role": "user",
            "content": content,
            "active": False  # Se activará cuando el assistant sea registrado
        })
        return node_id

    def add_assistant(self, node_id: int, content: str) -> None:
        """
        Registra el nodo assistant con el mismo ID que el nodo user correspondiente.
        Al completarse el par, ambos se activan.
        """
        self._nodes.append({
            "id": node_id,
            "role": "assistant",
            "content": content,
            "active": True
        })

        # Activar también el nodo user correspondiente
        for node in self._nodes:
            if node["id"] == node_id and node["role"] == "user":
                node["active"] = True
                break

    # ------------------------------------------------------------------ #
    #  Activación / Desactivación                                          #
    # ------------------------------------------------------------------ #

    def toggle(self, node_id: int, active: Optional[bool] = None) -> None:
        """
        Activa o desactiva un nodo del historial por su ID.

        - Si el nodo es de tipo 'system': activa/desactiva solo ese nodo.
          Si se activa, desactiva cualquier otro system activo.
        - Si el nodo es de tipo 'user'/'assistant': activa/desactiva el par completo,
          solo si el par está completo (existe tanto el user como el assistant para ese ID).

        Args:
            node_id: El ID del nodo a modificar.
            active: True para activar, False para desactivar. 
                    Si es None, invierte el estado actual (toggle).
        """
        nodes_with_id = [n for n in self._nodes if n["id"] == node_id]

        if not nodes_with_id:
            raise ValueError(f"No existe ningún nodo con el ID {node_id}.")

        first_node = nodes_with_id[0]

        if first_node["role"] == "system":
            new_state = active if active is not None else not first_node["active"]
            first_node["active"] = new_state

            # Si se activa este system, desactivar los demás
            if new_state:
                for node in self._nodes:
                    if node["role"] == "system" and node["id"] != node_id:
                        node["active"] = False
        else:
            # Para user/assistant: verificar que el par esté completo
            roles_in_id = {n["role"] for n in nodes_with_id}
            if not {"user", "assistant"}.issubset(roles_in_id):
                raise ValueError(
                    f"El nodo {node_id} no tiene un par user/assistant completo. No puede activarse/desactivarse."
                )

            # Determinar el nuevo estado a partir de cualquiera de los dos
            reference = next(n for n in nodes_with_id if n["role"] == "user")
            new_state = active if active is not None else not reference["active"]

            for node in nodes_with_id:
                node["active"] = new_state

    # ------------------------------------------------------------------ #
    #  Construcción de mensajes para la API                                #
    # ------------------------------------------------------------------ #

    def build_messages(self) -> List[Dict[str, Any]]:
        """
        Construye la lista de mensajes que se enviará a la API de OpenAI.
        - Incluye el último system activo (si existe).
        - Incluye en orden todos los nodos activos de user/assistant.
        """
        messages = []

        # 1. Buscar el último nodo system activo
        last_system = None
        for node in self._nodes:
            if node["role"] == "system" and node["active"]:
                last_system = node

        if last_system:
            messages.append({
                "role": "system",
                "content": last_system["content"]
            })

        # 2. Agregar todos los nodos user/assistant activos en orden
        for node in self._nodes:
            if node["role"] in ("user", "assistant") and node["active"]:
                messages.append({
                    "role": node["role"],
                    "content": node["content"]
                })

        return messages

    def get_active_node_ids(self) -> List[int]:
        """Devuelve la lista de IDs de nodos activos (sin duplicados, en orden)."""
        seen = set()
        active_ids = []
        for node in self._nodes:
            if node["active"] and node["id"] not in seen:
                seen.add(node["id"])
                active_ids.append(node["id"])
        return active_ids

    # ------------------------------------------------------------------ #
    #  Representación                                                      #
    # ------------------------------------------------------------------ #

    def to_dataframe(self) -> pd.DataFrame:
        """Devuelve el historial completo como un DataFrame de pandas."""
        if not self._nodes:
            return pd.DataFrame(columns=["id", "role", "content", "active"])

        records = []
        for node in self._nodes:
            records.append({
                "id": node["id"],
                "role": node["role"],
                "content": node["content"],
                "active": node["active"]
            })
        return pd.DataFrame(records)

    def __str__(self) -> str:
        total = len(self._nodes)
        active = len([n for n in self._nodes if n["active"]])
        return f"ConversationHistory(total_nodes={total}, active_nodes={active})"

    def __repr__(self) -> str:
        display(self.to_dataframe())
        return self.__str__()

    def deepcopy(self) -> "ConversationHistory":
        """Devuelve una copia profunda e independiente del historial."""
        new_history = ConversationHistory()
        new_history._nodes = copy.deepcopy(self._nodes)
        new_history._next_id = self._next_id
        return new_history

    def clear(self, include_system: bool = False) -> None:
        """
        Desactiva todos los nodos del historial sin eliminarlos.

        Args:
            include_system: Si es True, también desactiva los nodos de sistema.
                            Si es False (por defecto), conserva el system activo actual.
        """
        for node in self._nodes:
            if node["role"] == "system" and not include_system:
                continue
            node["active"] = False

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve el estado completo del historial como un diccionario."""
        return {
            "nodes": self._nodes,
            "next_id": self._next_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationHistory":
        """Crea una instancia de historial a partir de un diccionario."""
        history = cls()
        history._nodes = data["nodes"]
        history._next_id = data["next_id"]
        return history
