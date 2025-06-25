from langchain_aws.llms import SagemakerEndpoint
from langchain_aws.llms.sagemaker_endpoint import LLMContentHandler
import json

class ChatMessageContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        messages = [
            {"role": "system", "content": model_kwargs.get("system_prompt", "Eres un asistente de IA.")},
            {"role": "user", "content": prompt}
        ]
        body = {
            "messages": messages,
            "temperature": model_kwargs.get("temperature", 0.7),
            "max_tokens": model_kwargs.get("max_tokens", 1236), # Maximo numero de tokens en la respuesta
            "top_p": model_kwargs.get("top_p", 0.9),  # Sampling nucleus (probabilidad acumulada)
            "top_k": model_kwargs.get("top_k", 20),    # Número máximo de tokens para top-k sampling
            "presence_penalty": model_kwargs.get("presence_penalty", 0.0),  # Penaliza tokens ya usados (fomenta nuevos temas)
            "frequency_penalty": model_kwargs.get("frequency_penalty", 1.2),# Penaliza tokens repetidos (menos redundancia)
            "stream": True
        }
        return json.dumps(body).encode("utf-8")

    def transform_output(self, chunk: bytes) -> str:
        try:
            line = chunk.decode("utf-8").strip()
            if line.startswith("data:"):
                line = line.replace("data:", "").strip()
            data = json.loads(line)
            return data["choices"][0]["delta"].get("content", "")
        except Exception:
            return ""

# LLM con soporte streaming usando tu contenedor
llm = SagemakerEndpoint(
    endpoint_name="endpoint-vllm-model3B-trained2",
    region_name="us-east-1",
    content_handler=ChatMessageContentHandler(),
    streaming=True
)

# Test del stream
prompt = "Nombra lugares populares para visitar en Londres?"
for chunk in llm.stream(prompt, system_prompt=""):
    print(chunk, end="", flush=True)
