use {
    clap::Parser,
    global_opts::CmdChooser,
    misc::*,
    simple_firm::{SimpleFirmDifferentKOpts, SimpleFirmPhase}
};

pub mod misc;
mod global_opts;
mod simple_firm;
mod any_dist;

fn main() {
    
   // measure_end()
    let command = CmdChooser::parse();

    match command{
        CmdChooser::SimpleFirmDifK(opt) => {
            if opt.print_alternatives{
                SimpleFirmDifferentKOpts::print_alternatives(0);
            } else {
                let o: SimpleFirmDifferentKOpts = parse(opt.json);
                o.exec();
            }
        },
        CmdChooser::SimpleFirmPhase(opt) => {
            if opt.print_alternatives{
                SimpleFirmPhase::print_alternatives(0);
            } else {
                let o: SimpleFirmPhase = parse(opt.json);
                o.exec();
            }   
        }
    }
    
}

