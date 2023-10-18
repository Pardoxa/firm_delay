use {
    clap::Parser,
    global_opts::CmdChooser,
    misc::parse,
    simple_firm::{SimpleFirmDifferentKOpts, SimpleFirmPhase}
};

pub mod misc;
mod global_opts;
mod simple_firm;

fn main() {
    
   // measure_end()
    let command = CmdChooser::parse();

    match command{
        CmdChooser::SimpleFirmDifK(opt) => {
            let o: SimpleFirmDifferentKOpts = parse(opt.json);
            o.exec();
        },
        CmdChooser::SimpleFirmPhase(opt) => {
            let o: SimpleFirmPhase = parse(opt.json);
            o.exec();
        }
    }
    
}

