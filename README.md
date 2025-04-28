# Critical demand in a stochastic model of flows in supply networks

This code was written by Yannick Feld for the paper "Critical demand in a stochastic model of flows in supply networks" by Yannick Feld and Marc Barthelemy in Physical Review Letters (DOI: To be added once I know the DOI).

## Requirements for compiling and using the program

The program was programmed on Ubuntu 22.04 (and Arch and NixOS), but other Linux distributions should also work. 
GCC or Clang are assumed to be installed.
Some of the subcommands (those that will create a little animation) of the program require ffmpeg and gnuplot to be installed and available in the PATH. 
Otherwise these subcommands will crash during runtime.


## Compiling

Note: The following instructions are for Ubuntu 22.04, but they should be the same for most other linux distributions.


If you do not have Rust (and cargo) installed, install it via [rustup](https://doc.rust-lang.org/book/ch01-01-installation.html).


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
This will download the dependencies and compile the program with release optimizations.
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


## Generating data from the paper:

### Fig 2)

I recommend to start in an empty folder, as the program will create a good number of temporary files.

Create a config file like this:
```json
{
    "opts": {
        "root_demand_rate_min": 0.0,
        "root_demand_rate_max": 1.0,
        "root_demand_samples": 100,
        "time": 4000000,
        "samples": 1,
        "chain_length": 2,
        "seed": 312211001,
        "threads": null,
        "num_chains": 1,
        "max_stock": 0.05,
        "initial_stock": "Full"
    },
    "chain_start": 1,
    "chain_end": 300,
    "chain_step": 10,
    "y_range": {
        "start": 0.0,
        "end": 1.0
    }
}
```

and save it as "config.json". 
"max_stock" corresponds to the stock "s", so adjust this to whatever you desire.
Adjust "chain_start" and "chain_end" to decide which "N" range to sample.
Adjust "chain_step" for the step size with which "N" will be sampled.
Next run:
```bash
firm_delay my chain-crit --json s0.05.json -o s0.05_a.dat -n
```
The "-n" skips the creation of an animation via ffmpeg, use this if ffmpeg is not installed or you don't want the animation. Remove the "-n" if you want the animation. Gnuplot is required.