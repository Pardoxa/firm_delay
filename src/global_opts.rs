use clap::Parser;

#[derive(Parser, Debug)]
pub struct SimpleOpt{
    #[arg(short, long)]
    /// File Name of json file
    pub json: Option<String>,

}

#[derive(Parser)]
pub enum CmdChooser{
    SimpleFirmDifK(SimpleOpt)
}