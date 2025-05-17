FROM pytorch/pytorch:latest

# package install
RUN apt-get update && apt-get install -y \
    python3-pip \
    libx11-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# pip install
RUN pip install --upgrade pip

# Copy requirements.txt
COPY requirements.txt /workspace/requirements.txt

# Install pip packages
RUN pip install -r /workspace/requirements.txt

# set working directory
WORKDIR /workspace

CMD ["/bin/bash"]