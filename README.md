# gnss-dsp

`gnss-dsp` is a set of signal processing tools for GNSS. It is mainly
implemented in Rust, with some Python for scripting and plotting. It can be used
as a Python package, which contains some CLI tools, and as a Rust crate, for
access to most of the functionality within Rust projects.

Currently this package contains a CUDA high-sensitivity acquisition algorithm
for GPS L1 C/A.

## Development

This repository uses [just](github.com/casey/just) to run common development
tasks. Run `just` to print the available recipes. The recommended way of doing
development is to create a Python virtual environment, activate it, and run
`just develop`, which calls `maturin develop` to build the package and install
it into the virtual environment.

[uv](https://docs.astral.sh/uv/) is required to run several development tasks.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
