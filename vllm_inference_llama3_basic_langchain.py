"""
Script: vllm_inference_llama3_basic_langchain.py

Descripción:
------------
Este script permite realizar inferencias utilizando un modelo LLaMA 3 desplegado en Amazon SageMaker mediante la biblioteca `langchain-aws`.

Utiliza un endpoint ya activo y configurado con DJL + vLLM en SageMaker. La clase personalizada `ContentHandler_inference` permite transformar el input y output del modelo para que sea compatible con los formatos esperados por el endpoint SageMaker (tipo chat-style similar a OpenAI).

Requisitos:
-----------
- Tener un endpoint activo en SageMaker con un modelo de lenguaje tipo LLaMA 3 compatible con la API de DJL.
- Haber instalado previamente `langchain_aws`, `boto3` y sus dependencias.

"""

import json
import re
from typing import Dict
from langchain_aws.llms import SagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler

# --------------------------------------------------------------------------------
# Configuración del endpoint
# --------------------------------------------------------------------------------

ENDPOINT_NAME = "endpoint-vllm-model3B-trained2"
REGION_NAME = "us-east-1"

# --------------------------------------------------------------------------------
# Prompts: sistema y usuario
# --------------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Eres un asistente de IA especializado en responder Preguntas. "
    "Responde con una oración clara, precisa y sin explicaciones ni pensamientos."
)

USER_PROMPT = "¿Cuál es la capital de Perú?"

# --------------------------------------------------------------------------------
# ContentHandler personalizado para el endpoint de SageMaker
# --------------------------------------------------------------------------------

class ContentHandler_inference(LLMContentHandler):
    """Clase que transforma la entrada y salida para el endpoint de inferencia SageMaker vLLM"""
    
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        """
        Convierte el prompt y parámetros adicionales en un payload JSON codificado.
        """
        payload = {
            "messages": [
                {"role": "system", "content": model_kwargs.get("system_prompt", "Eres un asistente útil que da respuestas concisas y breves.")},
                {"role": "user", "content": prompt}
            ],
            "max_length": model_kwargs.get("max_length", 650),
            "temperature": model_kwargs.get("temperature", 0.2),
            "max_new_tokens": model_kwargs.get("max_new_tokens", 50),
            "do_sample": model_kwargs.get("do_sample", True),
            "top_p": model_kwargs.get("top_p", 0.9),
            "repetition_penalty": model_kwargs.get("repetition_penalty", 1),
            "top_k": model_kwargs.get("top_k", 50),
            "num_return_sequences": model_kwargs.get("num_return_sequences", 1)
        }
        return json.dumps(payload).encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        """
        Procesa la respuesta JSON del endpoint y devuelve el texto limpio.
        """
        response_json = json.loads(output.read().decode("utf-8"))
        raw_text = response_json["choices"][0]["message"]["content"]

        # Limpiar etiquetas HTML si las hubiera
        cleaned_text = re.sub(r"</?\w+>", "", raw_text).strip()
        return cleaned_text

# --------------------------------------------------------------------------------
# Inicializar el modelo LLM usando LangChain
# --------------------------------------------------------------------------------

content_handler_inference = ContentHandler_inference()

llm_inference = SagemakerEndpoint(
    endpoint_name=ENDPOINT_NAME,
    region_name=REGION_NAME,
    content_handler=content_handler_inference
)

# Para este ejemplo simple, no se utiliza una chain compleja de LangChain
chain_inference = llm_inference

# --------------------------------------------------------------------------------
# Ejecutar la inferencia
# --------------------------------------------------------------------------------

result = chain_inference.invoke(USER_PROMPT, system_prompt=SYSTEM_PROMPT)

# --------------------------------------------------------------------------------
# Mostrar el resultado
# --------------------------------------------------------------------------------

print("\n\033[1;33m>>> Respuesta final del modelo:\033[0m", result)
