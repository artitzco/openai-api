import os
import base64
import mimetypes
from typing import Dict, Any


class ContentPart:
    """
    Clase base para todos los componentes de un mensaje multimodal (imágenes, audio, archivos, etc.).
    Garantiza que el cliente pueda codificarlos e iterarlos de forma segura usando OOP.
    """

    def encode(self) -> Dict[str, Any]:
        """
        Codifica la instancia y construye la estructura de diccionario requerida por OpenAI.
        Debe ser sobrescrito por las clases hijas.
        """
        raise NotImplementedError(
            "Cada ContentPart debe implementar el método encode().")


class Image(ContentPart):
    """
    Clase para representar una imagen y procesarla en un formato multimodal compatible con OpenAI.
    """

    def __init__(self, path_or_url: str, detail: str = "auto"):
        self.path_or_url = path_or_url
        self.detail = detail

    def encode(self) -> Dict[str, Any]:
        """
        Codifica la instancia y construye la estructura de diccionario en el momento en el 
        que el cliente lo requiera para su envío.
        """
        if self.path_or_url.startswith("http://") or self.path_or_url.startswith("https://"):
            url = self.path_or_url
        else:
            # Es una ruta local, así que lo convertimos a base64
            if not os.path.exists(self.path_or_url):
                raise FileNotFoundError(
                    f"No se pudo encontrar la imagen en la ruta local: {self.path_or_url}")

            mime_type, _ = mimetypes.guess_type(self.path_or_url)
            if not mime_type:
                mime_type = "image/jpeg"  # Fallback por defecto

            with open(self.path_or_url, "rb") as image_file:
                encoded_string = base64.b64encode(
                    image_file.read()).decode('utf-8')

            # Formatear a la sintaxis que pide OpenAI (Data URL)
            url = f"data:{mime_type};base64,{encoded_string}"

        return {
            "type": "image_url",
            "image_url": {
                "url": url,
                "detail": self.detail
            }
        }
