# Critical demand in a stochastic model of flows in supply networks

This code was written by Yannick Feld for the paper "Critical demand in a stochastic model of flows in supply networks" by Yannick Feld and Marc Barthelemy in Physical Review Letters (DOI: To be added once I know the DOI).

## Requirements for compiling and using the program

The program was programmed on Ubuntu 22.04 (and Arch and NixOS), but other Linux distributions should also work. 
GCC or Clang are assumed to be installed.
Some of the subcommands (those that will create a little animation) of the program require ffmpeg and gnuplot to be installed and available in the PATH. 
Otherwise these subcommands will crash during runtime.


## Compiling

Note: The following constructions are for Ubuntu 22.04, but they should be the same for other linux distributions.


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
or just add the executable to your PATH variable to be able to call firm_delay from anywhere.

### Option 2)

you can also install it by opening a terminal in the folder of this README file and then running:
```bash
cargo install --path .
```
This will download the dependencies, compile the program and add the executable to .cargo/bin which is 
added to your PATH during the installation of Rust, 
at least with default settings, so you can skip the PATH adding 
step and still call the program firm_delay from anywhere.