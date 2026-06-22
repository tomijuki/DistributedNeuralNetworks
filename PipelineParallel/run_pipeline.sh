#!/bin/bash
set -e

IMAGE_NAME="mnist-pipeline"
JOB_NAME="mnist-pipeline-worker"
SERVICE_NAME="mnist-pipeline-service"

# Delete previous job and service if they exist
echo "Deleting previous job and service..."
#kubectl delete job "$JOB_NAME" --ignore-not-found
#kubectl delete service "$SERVICE_NAME" --ignore-not-found

# Rebuild Docker image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:latest" .

# Apply the job
# kubectl apply -f pipeline.yaml
