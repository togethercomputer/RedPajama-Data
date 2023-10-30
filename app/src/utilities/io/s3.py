import boto3
from botocore import UNSIGNED


def init_client(
        endpoint_url: str,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        signature_version: str = UNSIGNED
):
    return boto3.client(
        service_name='s3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
        config=boto3.session.Config(
            signature_version=signature_version,
            retries={
                'max_attempts': 5,  # this is the default in standard mode
                'mode': 'standard'
            }
        )
    )
