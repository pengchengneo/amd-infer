use std::{env, path::PathBuf, process::Command};
fn main() {
    println!("cargo:rerun-if-changed=kernels/iq4_xs.hip");
    println!("cargo:rerun-if-changed=kernels/execution_graph.h");
    println!("cargo:rerun-if-changed=kernels/gdn_profile.h");
    println!("cargo:rerun-if-changed=kernels/attention_profile.h");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-changed=kernels/ggml_reference");
    if env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let rocm = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm-7.2.3".into());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let result = Command::new(format!("{rocm}/bin/hipcc"))
        .args([
            "-O3",
            "-std=c++17",
            "--offload-arch=gfx1201",
            "-shared",
            "-fPIC",
            "kernels/iq4_xs.hip",
            "-o",
        ])
        .arg(out.join("libamdinfer_kernels.so"))
        .status()
        .expect("could not run hipcc");
    assert!(result.success(), "HIP compilation failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=amdinfer_kernels");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{rocm}/lib");
}
