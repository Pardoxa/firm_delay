use clap::Parser;

#[derive(Parser, Debug)]
pub struct SimpleOpt{
    #[arg(short, long)]
    /// path to json file
    pub json: Option<String>,

    #[arg(short, long)]
    /// print alternative json options
    pub print_alternatives: bool

}

/// Created by Yannick Feld
/// Program to simulate the delay in firms
#[derive(Parser)]
#[command(author, version, about)]
pub enum CmdChooser{
    SimpleFirmDifK(SimpleOpt),
    SimpleFirmPhase(SimpleOpt)
}