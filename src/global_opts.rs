use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
pub struct SimpleOpt{
    #[arg(short, long)]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// print alternative json options
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
    pub randomness: RandomState
}

/// Created by Yannick Feld
/// Program to simulate the delay in firms
#[derive(Parser)]
#[command(author, version, about)]
pub enum CmdChooser{
    SimpleFirmDifK(SimpleOpt),
    SimpleOtherFirmDifK(SimpleOpt),
    SimpleFirmPhase(SimpleOpt),
    SimpleFirmBufHist(SimpleOpt),
    SimpleFirmAverage(SimpleOpt),
    SimpleFirmAverageOrder(SimpleOpt),
    SimpleFirmAverageOrderMoran(SimpleOpt),
    SimpleFirmAverageOrderMoranAvalanch(SimpleOpt),
    SubMean(SubstitutingMeanFieldOpt),
    SubMeanVideo(SubstitutingMeanFieldOpt)
}