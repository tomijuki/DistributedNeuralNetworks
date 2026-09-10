FROM python:3.10.12-slim

#set working directory in the container
WORKDIR /app

# iproute2 provides `tc`, needed for netem fault injection (fault_inject.sh).
# Harmless in normal runs; only used when you actively inject a fault.
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

#copy requirements file to the container
COPY requirements.txt .

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Bake MNIST into the image so pods need no internet/DNS at runtime.
# Kept BEFORE `COPY main.py` so editing code doesn't re-download the dataset.
RUN python -c "from torchvision import datasets; \
    datasets.MNIST(root='data', train=True, download=True); \
    datasets.MNIST(root='data', train=False, download=True)"

#copy the rest of the application code to the container
COPY main.py .

# Overridden by the StatefulSet command; kept for local `docker run`.
CMD ["python", "main.py"]