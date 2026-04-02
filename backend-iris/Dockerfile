# Use a lightweight Python base image
FROM python:3.9-slim

# Create a working directory
WORKDIR /fastapi

# Copy requirements and install them
COPY requirements.txt /fastapi/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /fastapi

# 1) Run the training script to generate 'iris_model.joblib'
# It is retrieved from Hugging Face Hub in main.py, so this step is optional if the model is already uploaded there.

# Expose port 8000
EXPOSE 8000

# 2) Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
