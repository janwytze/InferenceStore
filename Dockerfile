FROM rust:1-alpine3.19 as build

ENV RUSTFLAGS="-C target-feature=-crt-static"
RUN apk add --no-cache musl-dev protoc

WORKDIR /app
COPY ./ /app

RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/app/target cargo build --release
RUN --mount=type=cache,target=/app/target strip target/release/inference-store
RUN --mount=type=cache,target=/app/target mv target/release/inference-store inference-store

FROM alpine:3.19

RUN apk add --no-cache libgcc

COPY --from=build /app/inference-store .

ENTRYPOINT ["/inference-store"]