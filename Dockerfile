# First stage: Training
FROM python:3.9-slim AS train

# Set the working directory
WORKDIR /app

# Copy pipeline script and requirements
COPY train_requirements.txt .
COPY ticket-helpdesk-multi-lang.csv .
COPY pipeline.py .

# Install dependencies
RUN pip install --no-cache-dir -r train_requirements.txt

# Run the training script to train and save the model
RUN python pipeline.py

# Second stage: Inference
FROM python:3.9-slim AS inference

# Set the working directory
WORKDIR /app

COPY serve_requirements.txt .

RUN pip install --no-cache-dir -r serve_requirements.txt

# Copy the trained model artifacts  from the train stage
COPY --from=train /app/intelligence /app

# Copy the FastAPI app script
COPY server.py .

# Expose the port
# EXPOSE 8000

# Run the FastAPI app
CMD ["python", "server.py"]