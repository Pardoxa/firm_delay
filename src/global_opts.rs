use std::num::*;

use camino::Utf8PathBuf;
use clap::{Parser, ValueEnum, Subcommand};
use rand_distr::Uniform;
use serde::{Deserialize, Serialize};

use crate::{complexer_firms::NetworkStructure, config_helper::AutoBufSubJobOpts, correlations::CorOpts, MyDistr};


#[derive(Parser, Debug)]
pub struct SimpleOpt{
    #[arg(short, long)]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// print alternative json options
    pub print_alternatives: bool

}

#[derive(Parser, Debug)]
pub struct SubAutoOpt{
    #[arg(long, requires("output"))]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// output name
    pub output: Option<String>,

    /// Num threads
    #[arg(short)]
    pub j: Option<NonZeroUsize>,

    #[arg(long, short)]
    /// Print alternatives
    pub print_alternatives: bool

}


#[derive(Parser, Debug)]
pub struct SubstitutingMeanFieldOpt{
    #[arg(short, long, requires("out_stub"))]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// output stub
    pub out_stub: Option<String>,


    /// desired framerate in Hz
    #[arg(short, long, default_value_t=25)]
    pub framerate: u8,

    /// print alternatives for json creation
    #[arg(long, short)]
    pub print_alternatives: bool,

    #[arg(long, short)]
    /// Do not clean temporary files afterwards
    pub no_clean: bool,

    #[arg(long, short)]
    /// Also create a converted video file that 
    /// has better compatibility, i.e., with more video players
    pub convert_video: bool
}

#[derive(Parser, Debug)]
pub struct SubstitutingNetworkOpt{
    #[arg(short, long, requires("out_stub"))]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// output stub
    pub out_stub: Option<String>,


    /// desired framerate in Hz
    #[arg(short, long, default_value_t=25)]
    pub framerate: u8,

    #[command(subcommand)]
    /// network structure
    pub structure: NetworkStructure,

    /// print alternatives for json creation
    #[arg(long, short)]
    pub print_alternatives: bool,

    #[arg(long, short)]
    /// Do not clean temporary files afterwards
    pub no_clean: bool,

    #[arg(long, short)]
    /// Also create a converted video file that 
    /// has better compatibility, i.e., with more video players
    pub convert_video: bool
}

#[derive(Parser, Debug)]
pub struct SubstitutingMeanFieldSampleVeloOpt{
    #[arg(short, long, requires("out_stub"))]
    /// path to json file
    pub json: Option<camino::Utf8PathBuf>,

    #[arg(short, long)]
    /// output stub
    pub out_stub: Option<String>,

    /// print alternatives for json creation
    #[arg(long, short)]
    pub print_alternatives: bool,

    #[clap(flatten)]
    pub gnuplot: Gnuplot
}

#[derive(Parser, Debug)]
pub struct Gnuplot{
    #[arg(long)]
    /// create plot with gnuplot
    pub gnuplot: bool,

    #[arg(long, requires("y_max"))]
    /// Set min for y-range
    pub y_min: Option<f64>,

    #[arg(long)]
    /// set max for y-range
    pub y_max: Option<f64>,

    #[arg(long, value_enum, default_value_t)]
    /// Choose other gnuplot terminal
    pub gnuplot_terminal: GnuplotTerminal
}

#[derive(Default, Debug, ValueEnum, Clone, Copy)]
pub enum GnuplotTerminal{
    /// Set gnuplot terminal to produce png output
    Png,
    #[default]
    /// Set gnuplot terminal to produce pdf output
    Pdf
}

impl GnuplotTerminal{
    pub fn str(self) -> &'static str
    {
        match self {
            Self::Pdf => "pdf",
            Self::Png => "png"
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum SimpleCommand{
    SimpleFirmDifK(SimpleOpt),
    SimpleOtherFirmDifK(SimpleOpt),
    SimpleFirmPhase(SimpleOpt),
    SimpleFirmBufHist(SimpleOpt),
    SimpleFirmAverage(SimpleOpt),
    SimpleFirmAverageOrder(SimpleOpt),
    SimpleFirmAverageOrderMoran(SimpleOpt),
    SimpleFirmAverageOrderMoranAvalanch(SimpleOpt),
}

#[derive(Subcommand, Debug)]
pub enum SubstitutingCommand{
    /// Sample velocity with option to create plot
    #[clap(visible_alias="vel")]
    Velocity(SubstitutingMeanFieldSampleVeloOpt),
    /// Create video and measure critical B over substitution probability
    #[clap(visible_alias="video")]
    CritBVideo(SubstitutingMeanFieldOpt),
    /// Create video and measure critical B over substitution probability (or delta buffer)
    #[clap(visible_alias="vNet")]
    CritBVideoNetworks(SubstitutingNetworkOpt),
    /// Calculate the autocorrelation of the mean delay
    Auto(SubAutoOpt)
}

/// Created by Yannick Feld
/// Program to simulate the delay in firms
#[derive(Parser)]
#[command(author, version, about)]
pub enum CmdChooser{
    #[command(subcommand)]
    /// Contains single firm subcommands
    Single(SimpleCommand),

    #[command(subcommand)]
    #[clap(visible_alias="sub")]
    /// Contains subcommands of substituting firms
    Substituting(SubstitutingCommand),

    #[command(subcommand)]
    #[clap(visible_alias="my")]
    /// Contains subcommands of my model
    MyModel(MyModelCommand),

    /// Calculate autocorrelation from a file
    Auto(CorOpts),

    /// Various helper stuff
    #[command(subcommand)]
    #[clap(visible_alias="he")]
    Helper(Helper),

    Test
}

#[derive(Subcommand, Debug)]
pub enum Helper{
    /// Create jsons for Correlation of Substituting firms from buffer file + example json
    SubAutoBufSwap(SubAutoBufSwaper),
    /// Create kanta job for autocorrelation measurings
    SubAutoKanta(AutoBufSubJobOpts)
}

#[derive(Parser, Debug)]
pub struct SubAutoBufSwaper{
    /// File that contains all the buffer vals as lines
    #[arg(long, short)]
    pub buffer_file: Utf8PathBuf,

    #[arg(long, short)]
    /// File that contains the old config
    pub example_file: Utf8PathBuf,

    #[arg(long, short)]
    #[clap(visible_alias="sub")]
    /// Also change the substitution probability
    pub change_sub_prob: Option<f64>
}

#[derive(Parser, Debug)]
pub struct PathAndOut{
    #[arg(short, long, requires("out"))]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// print alternative json options
    pub out: Option<Utf8PathBuf>

}

#[derive(Parser, Debug)]
pub struct TreePrintOpts{
    #[arg(short, long)]
    /// print alternative json options
    pub dot_out: Utf8PathBuf,

    /// How many children per node
    pub chilren_per_node: NonZeroUsize,

    /// How deep should the tree go?
    pub depth: usize,

    #[arg(long,short)]
    /// Normaly parents point at children, this reverses the direction
    pub reverse_direction: bool
}

#[derive(Parser, Debug)]
pub struct RandTreePrintOpts{
    #[arg(short, long)]
    /// print alternative json options
    pub dot_out: Utf8PathBuf,

    /// How deep should the tree go? Counting from 0
    pub max_depth: usize,

    /// Seed for the tree
    #[arg(long, short)]
    pub seed: Option<u64>,

    #[arg(long,short)]
    /// Normaly parents point at children, this reverses the direction
    pub reverse_direction: bool,

    /// Which distribution to use for childcount
    #[command(subcommand)]
    pub which: WhichDistr
}

impl RandTreePrintOpts{
    pub fn get_distr(&self) -> Box<dyn MyDistr>
    {
        match &self.which{
            WhichDistr::Uniform(u) => {
                Box::new(
                    Uniform::new_inclusive(u.start, u.end)
                )
            },
            WhichDistr::Constant(c) => {
                Box::new(c.value)
            }
        }
    }
}

#[derive(Parser, Debug)]
pub struct ClosedChainPrintOpts{
    #[arg(short, long)]
    /// print alternative json options
    pub dot_out: Utf8PathBuf,

    /// How many children per node
    pub num_chains: NonZeroUsize,

    /// How deep should the tree go?
    pub other_chain_len: NonZeroUsize
}

#[derive(Subcommand, Debug, Serialize, Deserialize, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum WhichDistr{
    /// Uniform
    Uniform(UniformParser),
    /// Constant
    Constant(ConstantParser)
}

#[derive(Subcommand, Debug)]
#[allow(clippy::enum_variant_names)]
pub enum MyModelCommand{
    /// Sample velocity 
    #[clap(visible_alias="vel")]
    ChainVelocity(PathAndOut),
    /// scan critical point for changing chain length
    #[clap(visible_alias="chcrit")]
    ChainCrit(PathAndOut),
    /// quenched m: scan critical point for changing chain length
    QuenchedChainCrit(PathAndOut),
    /// create profile of chain
    #[clap(visible_alias="chprof")]
    ChainProfile(PathAndOut),
    /// Sample crit for Trees
    #[clap(visible_alias="trcrit")]
    TreeCrit(PathAndOut),
    /// Sample crit for random Trees
    #[clap(visible_alias="rtrcrit")]
    RandTreeCrit(PathAndOut),
    /// Print tree dot files
    #[clap(visible_alias="dotT")]
    DotTree(TreePrintOpts),
    /// Print dot file for random tree
    #[clap(visible_alias="dotrT")]
    DotRandTree(RandTreePrintOpts),
    /// Print closed multi chain dot files
    #[clap(visible_alias="dotcms")]
    DotClosedMultiChain(ClosedChainPrintOpts),
    /// Measure criticallity for closed chains by scanning through chain len
    #[clap(visible_alias="cmscrit")]
    ClosedMultiChainCrit(PathAndOut),
    /// Measure criticallity for closed chains by scanning through num chains
    #[clap(visible_alias="cmscrit2")]
    ClosedMultiChainCrit2(PathAndOut)
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
/// Create a uniform distribution
pub struct UniformParser{
    #[arg(short, long)]
    /// inclusive lower bound
    pub start: usize,

    #[arg(short, long)]
    /// inclusive upper bound
    pub end: usize

}

#[derive(Parser, Debug, Serialize, Deserialize, Clone)]
/// Use a constant as distribution
pub struct ConstantParser{
    /// the value
    pub value: usize

}