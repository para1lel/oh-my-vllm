//! Bind the executable to its Cargo inputs and project Rust source tree.

use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};

fn visit(directory: &Path, paths: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(directory).expect("read Rust source directory") {
        let path = entry.expect("read source entry").path();
        if path.is_dir() {
            visit(&path, paths);
        } else if path.extension().is_some_and(|extension| extension == "rs")
            || path.file_name().is_some_and(|name| name == "Cargo.toml")
        {
            paths.push(path);
        }
    }
}

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("Cargo manifest directory");
    let root = Path::new(&manifest).parent().unwrap().parent().unwrap();
    let mut paths = vec![root.join("Cargo.toml"), root.join("Cargo.lock")];
    visit(&root.join("crates"), &mut paths);
    paths.sort();
    let mut hash = Sha256::new();
    println!("cargo:rerun-if-changed={}", root.join("crates").display());
    for path in paths {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path
            .strip_prefix(root)
            .unwrap()
            .to_str()
            .unwrap()
            .as_bytes();
        hash.update((relative.len() as u64).to_le_bytes());
        hash.update(relative);
        hash.update(Sha256::digest(
            fs::read(&path).expect("read Rust build input"),
        ));
    }
    println!(
        "cargo:rustc-env=OH_MY_VLLM_RUST_SOURCES_SHA256={:x}",
        hash.finalize()
    );
}
