# InferenceStore

[![deploy status](https://github.com/janwytze/inferencestore/workflows/test%20suite/badge.svg)](https://github.com/janwytze/inferencestore/actions)

A lightweight Inference Protocol GRPC service that caches inference requests.

InferenceStore can be used as a CI/CD and local development tool, so for example not a full
[NVIDIA Triton](https://developer.nvidia.com/triton-inference-server) instance needs to be started to do inference.
The service acts like an Inference Protocol service, but internally proxies the requests to an actual Inference Protocol
service. Once a request has completed, the inputs and outputs are cached. When a request with the same input occurs
again, the corresponding outputs are returned, without actually using the model.

This tool is still under active development, which means that breaking changes can occur.
The tool is currently in an early stage, so not all features are available yet.

## Why InferenceStore?

InferenceStore has been developed to make testing of applications that use Triton models more efficient.
If you run integrations tests for every commit for an application that depends on Triton, the tests can take up a lot of
resources and cost a lot of time. The Triton Docker image is several GBs, and a decent model also adds a few more GBs.

This tool works in the magnitude of MBs instead of GBs. This reduces time and resources. It is not needed to run a full
Triton Docker image to test your application end-to-end, but just a small InferenceStore Docker image. The full ONNX
model is also not needed, since the model outputs are cached.

This tool is fully compatible with the Open Inference Protocol (V2 Inference Protocol), like Triton, Seldon, OpenVINO
and TorchServe.

## Getting started

Currently, this service only has a Docker image available.
Using it in your project can be done using the following `docker-compose.yml` file:

```yaml
version: '3.8'
services:
  inference_store:
    image: ghcr.io/janwytze/inferencestore:0.0.0
    ports:
      - "50051:50051"
    environment:
      - APP__MODE=collect # Change to "serve" to use the cache.
      - APP__TARGET_SERVER__HOST=http://triton:8001
    volumes:
      - "./inferencestore:/app/inferencestore" # Directory to store the cache.

  triton:
    image: nvcr.io/nvidia/tritonserver:23.02-py3
    ports:
      - "8001:8001"
    volumes:
      - "./models:/models"
    command: [ "tritonserver", "--model-repository=/models" ]
```

This `docker-compose.yml` file starts an InferenceStore service and a Triton service.
Doing inference requests to `inference_store` service will cache the outputs in the `./inferencestore` directory.
When the Triton service is down, the InferenceStore service will return the cached outputs.

## How it works

InferenceStore out-of-the-box Docker image that can be used to run the tool locally, or in CI/CD.
In a later development stadium executables will also be available.
The tool uses [tonic](https://github.com/hyperium/tonic) for the GRPC service.
The [Triton](https://github.com/triton-inference-server/common/tree/00b3a71519e32e3bc954e9f0d067e155ef8f1a6c/protobuf) protobuf definitions are used.

When an inference request comes in, it will check if a request with the same inputs has already been cached.
If not, the call is redirected to a target server (e.g. a Triton server), the response will be cached in the directory supplied in the settings (`./inferencestore` by default).
