"""Abstracted storage layer for media upload/download.

Uploads to S3-compatible storage (AWS, R2, B2, MinIO).
Returns direct public URLs.

Required env vars:
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    S3_REGION (default: eu-west-2)
    S3_BUCKET (required)
    S3_ENDPOINT_URL (optional, for R2/B2/MinIO)
"""

import mimetypes
import os
import urllib.request
import uuid


def _get_s3_client():
    """Create S3 client from environment variables."""
    from botocore.config import Config
    import boto3

    kwargs = {
        "region_name": os.environ.get("S3_REGION", "eu-west-2"),
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "config": Config(signature_version="s3v4"),
    }
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url

    return boto3.client("s3", **kwargs)


def _get_bucket() -> str:
    bucket = os.environ.get("S3_BUCKET", "")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is required")
    return bucket


def _upload_s3(local_path: str, key: str = "") -> str:
    """Upload a local file to S3 and return a direct URL."""
    if not key:
        ext = os.path.splitext(local_path)[1]
        key = f"comfy-gen/outputs/{uuid.uuid4().hex[:12]}{ext}"

    client = _get_s3_client()
    bucket = _get_bucket()
    region = os.environ.get("S3_REGION", "eu-west-2")
    endpoint_url = os.environ.get("S3_ENDPOINT_URL", "")

    content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
    client.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": content_type})

    # Generate pre-signed URL (works for all providers including private R2 buckets)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=604800,  # 7 days
    )
    return url


def upload(local_path: str, key: str = "") -> str:
    """Upload a local file to S3 and return a public URL.

    Args:
        local_path: Path to the local file.
        key: S3 key. If empty, auto-generates one.

    Returns:
        Direct public URL.
    """
    return _upload_s3(local_path, key)


def download(url: str, local_path: str) -> None:
    """Download a file from a URL to a local path.

    Handles both pre-signed S3 URLs and regular HTTP URLs.
    """
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, local_path)


def upload_input(local_path: str) -> str:
    """Upload an input file (image/video) and return a URL.

    Used by the CLI to upload inputs before submitting a job.
    """
    ext = os.path.splitext(local_path)[1]
    key = f"comfy-gen/inputs/{uuid.uuid4().hex[:12]}{ext}"
    return upload(local_path, key)
