# Segment Anything for GeoTiff Images

This repository contains a version of SAM for GTiff images.

More details on how to use the model are available in `test_image_processor.py`.

TODO: better documentation.

## Key features

SAM allows segmenting objects with spatial prompts that are in a form of bounding boxes and points. Note that the model can work only with bounding boxes. Points can be used to better identify objects inside the bounding box by defining foreground and background pixels.

<div style="display: flex; justify-content: space-between;">
    <img src="imgs/img_1.png" alt="example with bounding box" style="width: 45%;"/>
    <img src="imgs/img_2.png" alt="Result" style="width: 45%;"/>
</div>

## Image format

The input images can be of any size. However, do not use very large images to avoid saturating the machine as the model consumes a lot of resources.

The model accepts input images in Geotiff format.

The output of the model is the segmentation mask of the detected object and it comes in two formats: a png image and GeoTiff image.

## Repository content

```markdown
.
├── Dockerfile
├── README.md
├── imgs
│   ├── img_1.png
│   └── img_2.png
├── model.proto
├── model_pb2.py
├── model_pb2_grpc.py
├── pyproject.toml
├── requirements.txt
├── serve.py
├── src
│   ├── app
│   └── sam
├── test-data
│   ├── T40RBN_20230607T064629_RGB.tif
│   ├── bbox.cpg
│   ├── bbox.dbf
│   ├── bbox.prj
│   ├── bbox.shp
│   ├── bbox.shx
│   ├── palm_roi.cpg
│   ├── palm_roi.dbf
│   ├── palm_roi.prj
│   ├── palm_roi.shp
│   └── palm_roi.shx
└── test_image_processor.py
```

## Local Development

- In a terminal, clone the repository

```powershell
git clone https://github.com/AlbughdadiM/depai-sam.git
```

- Go to the repository directory

```powershell
cd depai-sam
```

- If the files `model_pb2_grpc.py` and `model_pb2.py` are not there, generate them using

```powershell
python3.10 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. model.proto
```

- Build the docker image

```powershell
docker build . -t sam:v0.1
```

- Create a container from the built image

```powershell
docker run --name=test -v ./test-data:/data -p 8061:8061 sam:v0.1
```

- Run the pytest

```powershell
pytest test_image_processor.py
```

## Container Registry

- Generate a personal access token: Github account settings > Developer settings > Personal access tokens (classic). Generate a token with the `read:package` scope.

- In a terminal, login to container registry using

```powershell
docker login ghcr.io -u USERNAME -p PAT
```

- Pull the image

```powershell
docker pull ghcr.io/albughdadim/depai-sam:v0.1
```

- Create a container

```powershell
docker run --name=test -p 8061:8061 ghcr.io/albughdadim/depai-sam:v0.1
```

>[!IMPORTANT]
> Please note how the input is formatted in `test_image_processor.py`. It must respect the specification in `model.proto`.
