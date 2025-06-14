# llama3-sagemaker-vllm-langchain
modelo LLaMA 3 desplegado en SageMaker con vLLM y LangChain.
Este repositorio contiene dos scripts principales para desplegar e interactuar con un modelo de lenguaje grande (LLM), como LLaMA 3, en Amazon SageMaker usando contenedores compatibles con DJL y vLLM. Se utiliza `LangChain` para hacer inferencias desde Python de manera eficiente.

---

## 🚀 Estructura del repositorio

```
.
├── deploy_sagemaker_endpoint_compressed_model.py   # Despliega el modelo en SageMaker
├── vllm_inference_llama3_basic_langchain.py        # Realiza inferencias vía LangChain
└── README.md                                       # Este archivo
```

---

## 🧠 Requisitos

### AWS
- Tener acceso a una cuenta de AWS con permisos para usar Amazon SageMaker, S3 y ECR.
- Rol IAM con las siguientes políticas:
  - `AmazonSageMakerFullAccess`
  - `AmazonS3ReadOnlyAccess`
  - `AmazonEC2ContainerRegistryReadOnly`

### Python
- Python 3.8 o superior
- Instalar los siguientes paquetes:

```bash
pip install boto3 sagemaker langchain_aws
```

---

## 📌 Despliegue del modelo en SageMaker

El script `deploy_sagemaker_endpoint_compressed_model.py` realiza lo siguiente:

1. Crea un objeto `Model` apuntando a un archivo `.tar.gz` del modelo almacenado en S3.
2. Configura el entorno de ejecución para DJL + vLLM.
3. Despliega el modelo en un endpoint de SageMaker con una instancia tipo `ml.g5.xlarge`.

### Personaliza antes de ejecutar:
- `role`: nombre de tu rol de SageMaker
- `model_data`: ruta S3 de tu modelo `.tar.gz`
- `endpoint_name`: nombre deseado para el endpoint

### Ejecuta el script:

```bash
python deploy_sagemaker_endpoint_compressed_model.py
```

---

## 🤖 Inferencia con LangChain

El script `vllm_inference_llama3_basic_langchain.py` hace inferencias a través del endpoint creado.

1. Usa `langchain_aws.SagemakerEndpoint` para conectarse al endpoint.
2. Transforma los prompts de entrada/salida para formato chat.
3. Muestra la respuesta generada por el modelo.

### Puedes modificar:
- `USER_PROMPT`: la pregunta que deseas hacerle al modelo.
- `SYSTEM_PROMPT`: las instrucciones al asistente (ej. tono, tipo de respuesta).

### Ejecuta el script:

```bash
python vllm_inference_llama3_basic_langchain.py
```

---

## 🧪 Ejemplo de salida

```text
>>> Respuesta final del modelo: Lima.
```

---
