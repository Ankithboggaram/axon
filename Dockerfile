# Stage 1: build
FROM rust:1.85-bookworm AS builder

WORKDIR /app

ENV RUSTFLAGS="-C strip=symbols"

# Build dependencies first for better layer caching.
COPY Cargo.toml Cargo.lock build.rs ./
COPY proto ./proto
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN cargo build --release --locked || true
RUN rm -rf src

COPY src ./src
RUN touch src/main.rs && cargo build --release --locked

# Collect ORT and OpenMP shared libraries for the runtime image.
RUN mkdir -p /app/lib && \
    find /app/target /root -name "libonnxruntime.so*" 2>/dev/null | head -1 | xargs -I{} cp {} /app/lib/libonnxruntime.so && \
    find /usr/lib -name "libgomp.so.1" 2>/dev/null | head -1 | xargs -I{} cp {} /app/lib/libgomp.so.1 && \
    test -f /app/lib/libonnxruntime.so || (echo "ERROR: libonnxruntime.so not found; cannot build runtime image" && exit 1)

# Stage 2: runtime
# Distroless nonroot: no shell, no package manager, runs as UID 65532.
FROM gcr.io/distroless/cc-debian12:nonroot AS runtime

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
