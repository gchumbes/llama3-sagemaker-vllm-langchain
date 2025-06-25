import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.model import Model

vllm_config = {
    "HF_MODEL_ID": "s3://test-bucket-llm-local3/model3B_trained2/",
    "HF_TOKEN": "entertoken",
    "OPTION_MAX_MODEL_LEN": "80000",
    "OPTION_MAX_ROLLING_BATCH_SIZE": "8",
    "OPTION_MODEL_LOADING_TIMEOUT": "1500",
    "SERVING_FAIL_FAST": "true",
    "OPTION_ROLLING_BATCH": "disable",
    "OPTION_ASYNC_MODE": "true",
    "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service"
}

CONTAINER_VERSION = '0.33.0-lmi15.0.0-cu128'
REGION = 'us-east-1'
# Construct container URI
container_uri = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/djl-inference:{CONTAINER_VERSION}'
role="SageMakerTestRole2"
# Select instance type
instance_type = "ml.g5.xlarge"

model = Model(image_uri=container_uri,
              role=role,
              env=vllm_config)
#endpoint_name = sagemaker.utils.name_from_base("endpoint-vllm-model3B-trained2")
endpoint_name = "endpoint-vllm-model3B-trained2"
print(endpoint_name)
model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name,
    container_startup_health_check_timeout = 1800
)
