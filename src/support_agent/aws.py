"""Shared AWS credential-chain, region and request-limit configuration."""

import os

import boto3
from botocore.config import Config


def bedrock_client(settings, *, timeout, max_retries, endpoint_url=None, service="bedrock-runtime"):
    session = boto3.Session(
        profile_name=settings.aws_profile,
        region_name=settings.aws_region
        or os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION"),
    )
    return session.client(
        service,
        endpoint_url=endpoint_url,
        config=Config(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={"mode": "standard", "total_max_attempts": max_retries + 1},
        ),
    )
