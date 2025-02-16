FROM python:3.12-slim-bookworm

# Install required dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates gnupg

# Add Node.js official repository and install Node.js + npm
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | tee /etc/apt/keyrings/nodesource.asc && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.asc] https://deb.nodesource.com/node_18.x bookworm main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y nodejs

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
