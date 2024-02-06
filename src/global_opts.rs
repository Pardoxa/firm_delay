use std::num::*;

use clap::{Parser, ValueEnum, Subcommand};

use crate::correlations::CorOpts;


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
    #[arg(long)]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// output name
    pub output: String,

    /// Num threads
    #[arg(short)]
    pub j: Option<NonZeroUsize>,

    #[arg(long, short)]
    /// Print alternatives
    pub print_alternatives: bool

}

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
pub enum RandomState{
    Quenched,
    #[default]
    Dynamic
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

    /// Quenched or not?
    #[arg(short, long, value_enum, default_value_t)]
    pub randomness: RandomState,

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

    /// Calculate autocorrelation from a file
    Auto(CorOpts)
}