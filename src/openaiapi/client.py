import os
from openai import OpenAI
from typing import List, Dict, Optional, Any


class ChatClient:
    """
    Clase minimalista para manejar una conversación (chat) con un modelo de OpenAI.
    Mantiene un registro de la conversación internamente.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-5-mini", system_prompt: Optional[str] = None, **kwargs):
        """
        Inicializa la instancia de chat.

        Args:
            api_key: Clave de API de OpenAI. Si es None, intentará usar la variable de entorno OPENAI_API_KEY.
            model: Identificador del modelo a utilizar por defecto.
            system_prompt: Instrucciones base opcionales para definir el comportamiento del asistente.
            **kwargs: Otros parámetros opcionales para la instancia del cliente.
        """
        self.client = OpenAI(api_key=api_key, **kwargs)
        self.model = model
        self.history: List[Dict[str, str]] = []

        # Diccionario para mantener el registro interno completo del uso de tokens
        self.usage_details: Dict[str, Any] = {}

        # Si se provee un system prompt, lo agregamos al inicio de la conversación
        if system_prompt:
            self._add_to_history("system", system_prompt)

    def _add_to_history(self, role: str, content: str) -> None:
        """
        Método interno para registrar la información importante de la conversación.
        """
        self.history.append({"role": role, "content": content})

    def chat(self, message: str) -> str:
        """
        Recibe un mensaje del usuario, lo envía al modelo manteniendo el flujo
        de la conversación y devuelve la respuesta.

        Args:
            message: El mensaje de texto del usuario.

        Returns:
            La respuesta en texto generada por el modelo.
        """
        # Registrar el mensaje del usuario
        self._add_to_history("user", message)

        try:
            # Petición a la API usando el historial completo
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history
            )

            # Extraer y registrar la respuesta del asistente
            assistant_reply = response.choices[0].message.content
            self._add_to_history("assistant", assistant_reply)

            # Acumular dinámicamente todo el uso reportado por la API (incluyendo detalles anidados)
            if response.usage:
                usage_dict = response.usage.model_dump(exclude_none=True)

                def _accumulate(target: dict, source: dict) -> None:
                    for k, v in source.items():
                        if isinstance(v, int) or isinstance(v, float):
                            target[k] = target.get(k, 0) + v
                        elif isinstance(v, dict):
                            if k not in target:
                                target[k] = {}
                            _accumulate(target[k], v)

                _accumulate(self.usage_details, usage_dict)

            return assistant_reply

        except Exception as e:
            # Si ocurre un error, deshacemos el registro del mensaje del usuario
            # para no manchar el historial con un mensaje que no fue procesado
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            raise RuntimeError(
                f"Error al comunicarse con la API de OpenAI: {e}")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Devuelve el registro completo de la conversación actual.
        """
        return self.history

    def set_conversation_history(self, history: List[Dict[str, str]]) -> None:
        """
        Sobrescribe el historial de conversación actual con uno nuevo.
        Recomendado al cargar conversaciones previamente guardadas.

        Args:
            history: Una lista de diccionarios con las claves "role" y "content".
        """
        self.history = history

    def get_usage_details(self) -> Dict[str, Any]:
        """
        Devuelve el resumen completo y acumulado del consumo reportado por la API 
        durante esta sesión con la instancia.
        """
        return self.usage_details

    def clear_history(self) -> None:
        """
        Reinicia el historial de la conversación. 
        Conserva el system prompt si fue definido originalmente.
        """
        # Guardar el primer mensaje si es un prompt de sistema
        system_msg = None
        if self.history and self.history[0]["role"] == "system":
            system_msg = self.history[0]

        self.history.clear()

        if system_msg:
            self.history.append(system_msg)
