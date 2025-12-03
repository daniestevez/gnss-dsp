# print just recipies
default:
    just -l

# run maturin develop
develop:
    uvx -w patchelf maturin develop --release -E matplotlib

# run cargo format and ruff format
format:
    cargo fmt
    uvx ruff format

# run cargo clippy and ruff check
lint:
    cargo clippy
    uvx ruff check

# profile acquisition algorithm with nsight-systems
profile-acquisition:
    cargo build --release --examples
    nsys profile target/release/examples/acquisition
