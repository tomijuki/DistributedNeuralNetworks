FROM python:3.10.12-slim

#set working directory in the container
WORKDIR /app

#copy requirements file to the container
COPY requirements.txt .

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

#copy the rest of the application code to the container
COPY main.py .

#command to run the training script
CMD ["python", "main.py"]