"""
Script: deploy_sagemaker_endpoint_compressed_model.py

Descripción:
------------
Este script despliega un modelo de lenguaje grande (LLM), como LLaMA 3, previamente entrenado y empaquetado en un archivo `.tar.gz` almacenado en Amazon S3, utilizando Amazon SageMaker.

Se utiliza la imagen de inferencia de DJL (Deep Java Library) para VLLM, compatible con el modelo, y se define una configuración personalizada a través de variables de entorno.

Requisitos:
-----------
- El modelo debe estar empaquetado y almacenado en un bucket S3 accesible.
- El rol especificado debe tener permisos adecuados para acceso a SageMaker, S3 y ECR.
- AWS CLI y SDK (`boto3` y `sagemaker`) deben estar instalados y configurados correctamente.

Autor: Gustavo Chumbes
"""

import boto3
import sagemaker
from sagemaker.model import Model

# --------------------------------------------------------------------------------
# Configuración general
# --------------------------------------------------------------------------------

REGION = 'us-east-1'
INSTANCE_TYPE = 'ml.g5.xlarge'
ROLE = 'SageMakerTestRole2'  # Rol IAM con permisos de acceso a SageMaker y S3

# Ruta del modelo empaquetado (.tar.gz) almacenado en S3
MODEL_DATA = 's3://test-bucket-llm-local3/model3B_trained2.tar.gz'

# Nombre del endpoint que se creará
ENDPOINT_NAME = 'endpoint-vllm-model3B-trained2'

# Versión del contenedor DJL VLLM (ajustar según compatibilidad CUDA/modelo)
CONTAINER_VERSION = '0.33.0-lmi15.0.0-cu128'
CONTAINER_URI = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/djl-inference:{CONTAINER_VERSION}'

# --------------------------------------------------------------------------------
# Variables de entorno para configurar el contenedor de inferencia VLLM
# --------------------------------------------------------------------------------

VLLM_CONFIG = {
    "HF_TOKEN": "entertoken",  # Token de acceso a HuggingFace si se requiere
    "OPTION_MAX_MODEL_LEN": "80000",
    "OPTION_MAX_ROLLING_BATCH_SIZE": "1",
    "OPTION_MODEL_LOADING_TIMEOUT": "1500",
    "SERVING_FAIL_FAST": "true",
    "OPTION_ROLLING_BATCH": "disable",
    "OPTION_ASYNC_MODE": "true",
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service"
}

# --------------------------------------------------------------------------------
# Crear el modelo y desplegar el endpoint
# --------------------------------------------------------------------------------

# Instanciar el objeto Model de SageMaker
model = Model(
    image_uri=CONTAINER_URI,
    role=ROLE,
    model_data=MODEL_DATA,
    env=VLLM_CONFIG
)

print(f"Desplegando el endpoint: {ENDPOINT_NAME}")

# Desplegar el modelo como un endpoint en SageMaker
predictor = model.deploy(
    initial_instance_count=1,
    instance_type=INSTANCE_TYPE,
    endpoint_name=ENDPOINT_NAME,
    container_startup_health_check_timeout=1800  # tiempo extendido por modelos grandes
)

print("Despliegue completado exitosamente.")
