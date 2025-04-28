use std::process::Command;

fn main() {
    // note: add error checking yourself.
    let output_result = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output();

    let git_hash = match output_result{
        Ok(output) => {
            match String::from_utf8(output.stdout){
                Ok(git_hash) => {
                    git_hash
                },
                Err(_) => {
                    "GitHash:PARSE_ERROR".to_owned()
                }
            }
        },
        Err(_) => {
            "GitHash:ERROR_git_not_installed?".to_owned()
        }
    };
    println!("cargo:rustc-env=GIT_HASH={}", git_hash);
    println!("cargo:rustc-env=BUILD_TIME_CHRONO={}", chrono::offset::Local::now());
}