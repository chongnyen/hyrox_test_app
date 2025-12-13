# Use a robust, common base image for Python, which includes essential development tools
# We use a slim Debian variant to keep the image small.
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install system dependencies needed for some Python packages (like scientific libraries)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install all Python dependencies from requirements.txt
# This includes FastAPI, Uvicorn, Pandas, Transformers, etc.
RUN pip install --no-cache-dir -r requirements.txt

# --- NOTE on PyTorch / Torch Dependencies ---
# If your fine-tuned Mistral model (via 'peft' and 'transformers') requires PyTorch,
# it is currently commented out in your requirements.txt.
# You might need to uncomment 'torch' or add a specific CPU version installation command here
# after your initial test, as the ML part will fail without the model library.
# Example if needed: RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy the rest of your application code into the container
# This includes main.py, pipeline files, and the Hybrid_App directory
COPY . /app

# Expose the ports used by FastAPI (8000) and Streamlit (8501)
# This helps Codespaces automatically map them for you.
EXPOSE 8000
EXPOSE 8501

# The default command when the container runs is often a shell for development,
# but you can define default launch services here if needed (not strictly required for Codespaces)

# CMD ["/bin/bash"]