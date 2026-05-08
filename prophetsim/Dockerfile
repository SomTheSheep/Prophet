FROM ubuntu:22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install python3, pip, and build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libpq-dev \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python (often expected by scripts)
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy our custom script to the image
COPY prophetsim.py .

CMD ["sh", "-c", "python prophetsim.py && tail -f /dev/null"]