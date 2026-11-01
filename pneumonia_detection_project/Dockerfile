
# Use a minimal base image with Python 3.11 installed (Keras 3.12 needs Python >= 3.10)
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install Python dependencies first so this layer is cached when only app code changes
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Define the command to run the Streamlit app on port 8501 and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]

# NOTE: Disable XSRF protection for easier external access in order to make batch predictions
