import os
import json
from openai import OpenAI
from typing import Optional

from .core import Chat
from .content import Image, ContentPart
from .history import ConversationHistory
from .metrics import Metrics


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

    def chat(self, model: str = "gpt-5-mini", system_prompt: Optional[str] = None) -> Chat:
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
