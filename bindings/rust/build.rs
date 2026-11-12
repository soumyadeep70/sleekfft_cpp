use std::{env, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Repo-level include/ directory (contains sleekfft/fft.h and the SIMD headers).
    let cpp_include = manifest.join("../../include");

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++23")
        .opt_level(3)
        .warnings(true)
        .include("csrc")
        .include(&cpp_include)
        .file("csrc/sleekfft_c.cpp");

    // The library is header-only, so only the shim needs compiling.
    // SIMD kernels need target flags unless they use per-function target attributes;
    // mirror whatever the top-level CMakeLists.txt passes.
    if env::var_os("CARGO_FEATURE_NATIVE").is_some() {
        build.flag_if_supported("-march=native");
    }

    build.compile("sleekfft_c");

    println!("cargo:rerun-if-changed=csrc");
    println!("cargo:rerun-if-changed=../../include");
    println!("cargo:rerun-if-env-changed=CXX");
}
