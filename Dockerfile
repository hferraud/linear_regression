FROM rust:1.67 as planner
WORKDIR /usr/src/linreg
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src/bin && \
    echo "fn main() {println!(\"dummy\")}" > src/bin/train.rs && \
    echo "fn main() {println!(\"dummy\")}" > src/bin/predict.rs && \
    touch src/lib.rs && \
    cargo build --release
#RUN cargo add serde
#RUN cargo add csv

FROM rust:1.67 as builder
WORKDIR /usr/src/linreg
COPY --from=planner /usr/src/linreg/target target
COPY --from=planner /usr/local/cargo /usr/local/cargo

VOLUME /usr/src/linreg
WORKDIR /usr/src/linreg

CMD ["cargo", "run", "--bin", "train"]
