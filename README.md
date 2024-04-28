# InferenceStore

[![deploy status](https://github.com/janwytze/inferencestore/workflows/test%20suite/badge.svg)](https://github.com/janwytze/inferencestore/actions)

A lightweight Inference Protocol GRPC service that caches inference requests.

InferenceStore can be used as a CI/CD and local development tool, so for example not a full
[NVIDIA Triton](https://developer.nvidia.com/triton-inference-server) instance needs to be started.
The service acts like an Inference Protocol service, but internally proxies the requests to an actual Inference Protocol
service. Once a request has completed, the inputs and outputs are cached. When a request with the same input occurs
again, the corresponding outputs are returned, without actually using the model.
