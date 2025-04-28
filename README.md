# 

## Compiling

Note: This was tested on Ubuntu 22.04, but other Linux distributions should 
also work.

If you do not have Rust (and cargo) installed, install it via [rustup](https://doc.rust-lang.org/book/ch01-01-installation.html) 


For compiling, first clone this repository via 
```bash
git clone https://github.com/Pardoxa/firm_delay.git
```
or via 
```bash
git clone git@github.com:Pardoxa/firm_delay.git
```
or download the zip file of this repository from Github and unpack it.

### Option 1)

Compile the program by opening a terminal in the folder where this README.md file is and executing the command:

```bash
cargo build --release
```
This will download the dependencies and compiles the program with release optimizations.
The executable "firm_delay" will be now located in ./target/release
and you can call it by running the command (make sure you are still in the folder with the README file):
```bash
./target/release/firm_delay
```
or just add the executable to your PATH variable.

### Option 2)

you can also install it by opening a terminal in the folder of this README file and then running:
```bash
cargo install --path .
```
This will add the executable to .cargo/bin which is 
added to your PATH during the installation of Rust, 
at least with default settings, so you can skip the PATH adding 
step and still call the program firm_delay from anywhere.