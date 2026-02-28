import os
import json
import copy
from openai import OpenAI
from typing import List, Dict, Optional, Any, Union

from .content import Image, ContentPart
from .history import ConversationHistory
from .metrics import Metrics


class Chat:
    """
    Clase para manejar una conversación (chat) con un modelo de OpenAI.
    Gestiona el historial mediante nodos con IDs y delega las métricas
    de uso a una clase especializada.
    """

    def __init__(self, openia: OpenAI, model: str = "gpt-5-mini", system_prompt: Optional[str] = None):
        """
        Inicializa la instancia de chat.

        Args:
            openia: Instancia de OpenAI a utilizar para este chat.
            model: Identificador del modelo a utilizar por defecto.
            system_prompt: Instrucciones base opcionales para definir el comportamiento del asistente.
        """
        self.openia = openia
        self.model = model

        # Instancias delegadas
        self.history = ConversationHistory()
        self.metrics = Metrics()

        # Si se provee un system prompt, lo registramos como nodo
        if system_prompt:
            self.history.add_system(system_prompt)

    def set_openia(self, openia: OpenAI) -> None:
        """Cambia la instancia activa de OpenAI utilizada para conectarse a la API."""
        self.openia = openia

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        """
        Establece un nuevo prompt de sistema en el historial.
        Si se recibe None, desactiva el system prompt activo actual (si existe).
        Cada cambio de system prompt genera un nuevo nodo con su propio ID.
        """
        if system_prompt is None:
            # Desactivar el system activo actual
            for node in self.history._nodes:
                if node["role"] == "system" and node["active"]:
                    node["active"] = False
            return

        self.history.add_system(system_prompt)

    def set_model(self, model: str) -> None:
        """
        Cambia el modelo de lenguaje de OpenAI utilizado para las siguientes interacciones.
        """
        self.model = model

    def copy(self, openia: Optional[OpenAI] = None) -> "Chat":
        """
        Crea y devuelve una copia exacta e independiente del chat actual.
        Alterar la copia no afectará el historial ni las métricas del chat original.

        Args:
            openia: Instancia opcional de OpenAI para usar en la copia. 
                    Si es None, se usa la misma instancia del chat original.
        """
        new_chat = Chat(
            openia=openia if openia else self.openia,
            model=self.model
        )

        # Copiamos profundamente las instancias delegadas
        new_chat.history = self.history.deepcopy()
        new_chat.metrics = self.metrics.deepcopy()

        return new_chat

    def chat(self, *messages: Any) -> str:
        """
        Recibe uno o múltiples mensajes del usuario, los envía al modelo manteniendo el flujo
        de la conversación y devuelve la respuesta.

        Args:
            *messages: Los componentes del mensaje del usuario de forma posicional.

        Returns:
            La respuesta en texto generada por el modelo.
        """
        if not messages:
            raise ValueError("Debes proporcionar al menos un mensaje.")

        # Construir el content (string puro o lista multimodal)
        if len(messages) == 1 and isinstance(messages[0], str):
            content = messages[0]
        else:
            content = []
            for msg in messages:
                if isinstance(msg, str):
                    content.append({"type": "text", "text": msg})
                elif isinstance(msg, ContentPart):
                    content.append(msg.encode())
                else:
                    raise ValueError(
                        f"Tipo de mensaje no soportado: {type(msg)}")

        # Registrar el nodo de usuario (inactivo hasta que el assistant responda)
        user_node_id = self.history.add_user(content)

        try:
            # Construir los mensajes a partir de los nodos activos + el mensaje actual
            api_messages = self.history.build_messages()

            # El mensaje del usuario actual aún no está activo, lo agregamos manualmente
            api_messages.append({"role": "user", "content": content})

            # Capturar los IDs activos de esta solicitud (incluyendo el del usuario actual)
            active_ids = self.history.get_active_node_ids() + [user_node_id]

            # Petición a la API usando self.openia
            response = self.openia.chat.completions.create(
                model=self.model,
                messages=api_messages
            )

            # Extraer y registrar la respuesta del asistente
            assistant_reply = response.choices[0].message.content
            self.history.add_assistant(user_node_id, assistant_reply)

            # Registrar métricas
            if response.usage:
                usage_dict = response.usage.model_dump(exclude_none=True)
                self.metrics.log(
                    usage_dict=usage_dict,
                    model=self.model,
                    active_node_ids=active_ids
                )

            return assistant_reply

        except Exception as e:
            raise RuntimeError(
                f"Error al comunicarse con la API de OpenAI: {e}")

    def clear(self, include_system: bool = False) -> None:
        """
        Desactiva los nodos del historial sin eliminarlos y reinicia las métricas.

        Args:
            include_system: Si es True, también desactiva los nodos de sistema.
                            Si es False (por defecto), conserva el system activo actual.
        """
        self.history.clear(include_system=include_system)
        self.metrics.clear()

    def __str__(self) -> str:
        return f"Chat(model='{self.model}', history={self.history}, metrics={self.metrics})"

    def __repr__(self) -> str:
        return self.__str__()

    def save(self, path: str) -> None:
        """
        Guarda el estado completo de la conversación y las métricas en un archivo JSON.
        """
        data = {
            "model": self.model,
            "history": self.history.to_dict(),
            "metrics": self.metrics.to_dict()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


class Client:
    """
    Clase principal que administra la instancia persistente del motor de OpenAI y 
    nos permite iniciar nuevas conversaciones (chats) o recuperar conversaciones guardadas.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Inicializa un cliente maestro conectándose a OpenAI una sola vez.

        Args:
            api_key: Opcional, llave de acceso a OpenAI.
            **kwargs: Otras configuraciones que acepta la clase nativa openai.OpenAI.
        """
        self.openia = OpenAI(api_key=api_key, **kwargs)

    def chat(self, model: str = "gpt-4o-mini", system_prompt: Optional[str] = None) -> Chat:
        """
        Inicia y devuelve un entorno completamente nuevo de Chat vinculado a este cliente.

        Args:
            model: Identificador del modelo (ej. 'gpt-4o', 'gpt-3.5-turbo').
            system_prompt: Propósito raíz del Chat que se va a iniciar.
        """
        return Chat(
            openia=self.openia,
            model=model,
            system_prompt=system_prompt
        )

    def load_chat(self, path: str) -> Chat:
        """
        Carga una instancia de un Chat usando un archivo JSON, y la conecta a este cliente.

        Args:
            path: La ruta donde se ubica el JSON exportado previamente.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No se encontró el archivo en la ruta: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Usar la instancia madre de OpenAI contenida en self.openia
        restored_chat = Chat(openia=self.openia, model=data["model"])

        # Restaurar estado del historial y métricas
        restored_chat.history = ConversationHistory.from_dict(data["history"])
        restored_chat.metrics = Metrics.from_dict(data["metrics"])

        return restored_chat


__all__ = ["Client", "Chat", "Image",
           "ContentPart", "ConversationHistory", "Metrics"]
