#!/bin/bash
set -e

IMAGE_NAME="mnist-baseline"
JOB_NAME="mnist-baseline-worker"
SERVICE_NAME="mnist-baseline-service"

# Delete previous job and service if they exist
echo "Deleting previous job and service..." 
kubectl delete job "$JOB_NAME" --ignore-not-found
kubectl delete service "$SERVICE_NAME" --ignore-not-found

# Rebuild Docker image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME:latest" .

# Load image into kind cluster
echo "Loading image into kind..."
kind load docker-image "$IMAGE_NAME:latest"

# Apply the job
# kubectl apply -f job.yaml
# kubectl get pods -w
# kubectl get logs podname