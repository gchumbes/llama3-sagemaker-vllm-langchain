## 📦 LLaMA 3.2 3B Instruct en SageMaker con vLLM (Streaming Enabled)

Este repositorio documenta el despliegue del modelo **LLaMA 3.2 3B Instruct** en **Amazon SageMaker** utilizando **vLLM** como servidor de inferencia, con soporte para **streaming de respuestas al estilo OpenAI**.\
Está diseñado para facilitar la implementación de modelos LLM eficientes en la nube, optimizando costos y latencia en instancias GPU como `g5.xlarge`.

El modelo puede ser invocado desde clientes como **LangChain**, simulando el comportamiento del endpoint `/v1/chat/completions` de OpenAI, permitiendo integraciones con agentes, herramientas y workflows conversacionales avanzados.

---

## ⚙️ Requisitos

### 🔸 Infraestructura

- AWS account con permisos para usar SageMaker
- IAM Role con acceso a SageMaker, S3 y ECR (por ejemplo, `SageMakerTestRole2`)
- Bucket S3 con el modelo (`model3B_trained2`) descomprimido
- Instancia SageMaker compatible con GPU:
  ```
  ml.g5.xlarge
  ```

### 🔸 Contenedor base

- Contenedor oficial de Amazon DJL Inference con soporte LMI v15:
  ```bash
  763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128
  ```
  Ejecuta vLLM mediante:
  ```
  OPTION_ENTRYPOINT=djl_python.lmi_vllm.vllm_async_service
  ```

### 🔸 Variables de entorno configuradas

| Variable                        | Descripción                                     |
| ------------------------------- | ----------------------------------------------- |
| `HF_MODEL_ID`                   | Ruta S3 con el modelo descomprimido             |
| `HF_TOKEN`                      | Token (si aplica)                               |
| `OPTION_MAX_MODEL_LEN`          | Longitud máxima de contexto en tokens (`80000`) |
| `OPTION_ASYNC_MODE`             | Habilita modo asincrónico (`true`)              |
| `OPTION_ROLLING_BATCH`          | Control de lotes (`disable`)                    |
| `OPTION_MAX_ROLLING_BATCH_SIZE` | Tamaño máximo de batch (`8`)                    |
| `OPTION_ENTRYPOINT`             | Inicia servidor vLLM (`vllm_async_service`)     |
| `SERVING_FAIL_FAST`             | Detiene carga si falla configuración            |

---

## 📂 Estructura del repositorio

```
📁
├── deploy_sagemaker_endpoint_uncompressed_model.py  # Despliegue del endpoint
├── vllm_inference_llama3_basic_langchain_stream.py  # Inferencia con LangChain
├── README.md
```

### 📄 Descripción de los scripts

- ``\
  Despliega un endpoint en SageMaker configurado con vLLM sobre GPU `ml.g5.xlarge`.

- ``\
  Conecta con LangChain al endpoint y recibe respuestas por stream al estilo ChatGPT.

---

## 🚀 Despliegue paso a paso

### 1. Subir el modelo a S3

Modelo descomprimido en bucket S3:

```
s3://test-bucket-llm-local3/model3B_trained2/
```

Debe contener archivos Hugging Face: `config.json`, `model.safetensors`, `tokenizer_config.json`, etc.

---

### 2. Desplegar el endpoint

Ejecutar [`deploy_sagemaker_endpoint_uncompressed_model.py`](./deploy_sagemaker_endpoint_uncompressed_model.py). Este script:

- Usa el contenedor DJL LMI v15
- Apunta al modelo en S3
- Lanza una instancia `ml.g5.xlarge`

---

### 3. Probar la inferencia (streaming)

Ejecutar [`vllm_inference_llama3_basic_langchain_stream.py`](./vllm_inference_llama3_basic_langchain_stream.py) para:

- Enviar un prompt estilo Chat
- Recibir tokens en stream
- Simular comportamiento del endpoint `/v1/chat/completions`

---

## 💬 Ejemplo de inferencia esperada

Prompt de prueba:

```python
prompt = "Nombra lugares populares para visitar en Londres?"
```

Salida (streaming):

```
Londres ofrece una gran variedad de lugares icónicos para visitar, entre ellos:
1. El Big Ben y el Palacio de Westminster
2. El London Eye
3. El Museo Británico
4. La Torre de Londres
...
```

> El contenido puede variar según el fine-tuning aplicado al modelo.

---


