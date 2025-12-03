use std::path::PathBuf;

fn main() {
    let out_path =
        PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR env variable not defined"));

    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let cuda_path =
        PathBuf::from(std::env::var("CUDA_PATH").expect("CUDA_PATH env variable not defined"));
    println!(
        "cargo:rustc-link-search={}",
        cuda_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=cufft");

    let cufft_bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", cuda_path.join("include").display()))
        // these cause multiple definition errors if not blocklisted
        .blocklist_item("FP_NAN")
        .blocklist_item("FP_ZERO")
        .blocklist_item("FP_INFINITE")
        .blocklist_item("FP_NORMAL")
        .blocklist_item("FP_SUBNORMAL")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("unable to generate cufft bindings");
    cufft_bindings
        .write_to_file(out_path.join("cufft_bindings.rs"))
        .expect("could not write cufft_bindings.rs");
}
