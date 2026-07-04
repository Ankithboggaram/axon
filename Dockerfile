# Stage 1: build
# Note: several dependencies (e.g. ort-sys 2.0.0-rc.12) now require a newer
# rustc than the crate's declared rust-version = "1.85" MSRV floor. 1.85 is
# the minimum Axon's own code needs, not a ceiling; this pins a toolchain new
# enough to actually build the current dependency tree.
#
# trixie (not bookworm): the ONNX Runtime binary ort's "download-binaries"
# feature fetches at build time is linked against glibc symbols
# (__isoc23_strtoll and friends) that bookworm's glibc doesn't have. The
# runtime image must match (see stage 2) since the same .so runs there too.
FROM rust:1.88-trixie AS builder

WORKDIR /app

ENV RUSTFLAGS="-C strip=symbols"

# rdkafka's cmake-build feature compiles librdkafka from source (vendored, so
# no system librdkafka package is needed) but still needs a C build toolchain
# plus zlib/SSL headers to configure against. protobuf-compiler provides
# `protoc`, which both axon's own build.rs (tonic-build) and cortex-contract's
# (prost-build) shell out to at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    pkg-config \
    libssl-dev \
    zlib1g-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Build dependencies first for better layer caching.
# benches/ is required even for a plain build: Cargo.toml declares [[bench]]
# targets by name, so it needs benches/*.rs on disk just to parse the manifest.
COPY Cargo.toml Cargo.lock build.rs ./
COPY proto ./proto
COPY benches ./benches
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN cargo build --release --locked || true
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release --locked

# Collect librdkafka's dynamically-linked dependencies for the runtime image.
# librdkafka itself is statically vendored (cmake-build) and, as of the
# current `ort`, so is ONNX Runtime (ort's "download-binaries" now fetches a
# static libonnxruntime.a, linked directly into the axon binary) -- verified
# via `ldd target/release/axon`, which shows no libonnxruntime/libgomp
# dependency at all, only these:
RUN mkdir -p /app/lib && \
    find /usr/lib -name "libssl.so*" -o -name "libcrypto.so*" -o -name "libz.so*" -o -name "libzstd.so*" 2>/dev/null | xargs -I{} cp -n {} /app/lib/ && \
    test -f /app/target/release/axon || (echo "ERROR: axon binary not found; cannot build runtime image" && exit 1)

# Stage 2: runtime
# Distroless nonroot: no shell, no package manager, runs as UID 65532.
# debian13 (trixie), matching the builder's glibc — see stage 1's note.
FROM gcr.io/distroless/cc-debian13:nonroot AS runtime

WORKDIR /app

COPY --from=builder /app/target/release/axon ./axon
COPY --from=builder /app/lib ./lib
COPY examples/config.toml ./config.toml

ENV LD_LIBRARY_PATH=/app/lib

EXPOSE 50051
EXPOSE 9090

USER nonroot:nonroot

ENTRYPOINT ["./axon"]
CMD ["serve", "--config", "config.toml"]
