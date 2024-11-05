# Use Ubuntu 22.04 as the base image
FROM ubuntu:24.04

# Update package lists and install dependencies
RUN apt-get -y update \
    && apt-get install -y --no-install-recommends \
    software-properties-common \
    wget \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

RUN mkdir /segment-anything

COPY . /segment-anything/

# Set the working directory
WORKDIR /segment-anything

RUN python3 -m venv ./venv \
    && chmod +x ./venv/bin/activate \
    && ./venv/bin/pip install --upgrade pip \
    && ./venv/bin/pip install -r requirements.txt

RUN mkdir /segment-anything/src/weights \
    && wget -P /segment-anything/src/weights/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Set the command to run the application
ENV PATH="/segment_anything/venv/bin:$PATH"
CMD ["/segment-anything/venv/bin/python", "serve.py"]