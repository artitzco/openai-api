import os
import copy
import pandas as pd
from openai import OpenAI
from typing import List, Dict, Optional, Any, Union
from .content import ContentPart


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

        # Lista para mantener el registro puro del uso de tokens por cada respuesta
        self.usage_details: List[Dict[str, Any]] = []

        # Si se provee un system prompt, lo agregamos al inicio de la conversación
        if system_prompt:
            self._add_to_history("system", system_prompt)

    def _add_to_history(self, role: str, content: Union[str, List[Dict[str, Any]]]) -> None:
        """
        Método interno para registrar la información importante de la conversación.
        """
        self.history.append({"role": role, "content": content})

    def set_system_prompt(self, system_prompt: Optional[str]) -> None:
        """
        Actualiza o establece el prompt del sistema para la conversación.
        Si ya existe un mensaje de sistema (siempre debe ser el primero), lo reemplaza.
        Si se recibe None, elimina el prompt del sistema actual si existe.
        Si no existe y se recibe un string válido, lo inserta al inicio del historial.
        """
        has_system = self.history and self.history[0]["role"] == "system"

        if system_prompt is None:
            if has_system:
                self.history.pop(0)
            return

        if has_system:
            self.history[0]["content"] = system_prompt
        else:
            self.history.insert(
                0, {"role": "system", "content": system_prompt})

    def set_model(self, model: str) -> None:
        """
        Cambia el modelo de lenguaje de OpenAI utilizado para las siguientes interacciones.
        """
        self.model = model

    def copy(self) -> "ChatClient":
        """
        Crea y devuelve una copia exacta e independiente del cliente actual.
        Altera la copia no afectará el historial ni el registro de uso del cliente original.
        """
        # Creamos una instancia nueva vacía
        new_client = ChatClient(
            api_key=self.client.api_key,
            model=self.model
        )

        # Copiamos profundamente los registros para que sean totalmente independientes
        new_client.history = copy.deepcopy(self.history)
        new_client.usage_details = copy.deepcopy(self.usage_details)

        return new_client

    def chat(self, *messages: Any) -> str:
        """
        Recibe uno o múltiples mensajes del usuario, los envía al modelo manteniendo el flujo
        de la conversación y devuelve la respuesta. Puede recibir un simple string, múltiples
        strings o mezclar clases multimodales como 'Image'.

        Args:
            *messages: Los componentes del mensaje del usuario de forma posicional.

        Returns:
            La respuesta en texto generada por el modelo.
        """
        if not messages:
            raise ValueError("Debes proporcionar al menos un mensaje.")

        # Si el usuario mandó un único mensaje y es un texto, conservamos el comportamiento original y más puro
        if len(messages) == 1 and isinstance(messages[0], str):
            content = messages[0]
        else:
            # Si hay varios fragmentos o elementos complejos, construimos la lista multimodal
            content = []
            for msg in messages:
                if isinstance(msg, str):
                    content.append({"type": "text", "text": msg})
                elif isinstance(msg, ContentPart):
                    content.append(msg.encode())
                else:
                    raise ValueError(
                        f"Tipo de mensaje no soportado: {type(msg)}")

        # Registrar el mensaje unificado del usuario
        self._add_to_history("user", content)

        try:
            # Petición a la API usando el historial completo
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history
            )

            # Extraer y registrar la respuesta del asistente
            assistant_reply = response.choices[0].message.content
            self._add_to_history("assistant", assistant_reply)

            # Guardar el uso reportado de forma pura
            if response.usage:
                usage_dict = response.usage.model_dump(exclude_none=True)
                self.usage_details.append(usage_dict)

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

    def get_usage_details(self) -> pd.DataFrame:
        """
        Transforma el historial puro de tokens y devuelve el dataframe con el consumo de cada interacción.
        Las columnas son un MultiIndex construido a partir de los subniveles del diccionario puro.
        """
        if not self.usage_details:
            return pd.DataFrame()

        flattened_list = []
        for usage_dict in self.usage_details:
            flat_usage = {}

            def _flatten(source: dict, parent_key: tuple = ()) -> None:
                for k, v in source.items():
                    new_key = parent_key + (k,)
                    if isinstance(v, dict):
                        _flatten(v, new_key)
                    else:
                        flat_usage[new_key] = v

            _flatten(usage_dict)
            flattened_list.append(flat_usage)

        df = pd.DataFrame(flattened_list)

        # Obtener la profundidad máxima y emparejar las tuplas
        max_len = max(len(t) for t in df.columns)
        padded_columns = [t + tuple([""] * (max_len - len(t)))
                          for t in df.columns]

        df.columns = pd.MultiIndex.from_tuples(padded_columns)
        return df

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
