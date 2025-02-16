FROM python:3.12-slim-bookworm

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates gnupg

# Add Node.js official repository and install Node.js + npm
RUN apt-get update && apt-get install -y nodejs npm

# Verify installations
RUN node -v && npm -v && git --version

# Install GitPython
RUN pip install gitpython

# Download the latest UV installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Copy application files
COPY app.py /app

# Run the application
CMD ["uv", "run", "app.py"]
