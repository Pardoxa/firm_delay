use {
    clap::Parser,
    global_opts::CmdChooser,
    misc::*,
    simple_firm::{SimpleFirmDifferentKOpts, SimpleFirmPhase, SimpleFirmBufferHistogram, SimpleFirmAverageAfter}
};

pub mod misc;
mod global_opts;
mod simple_firm;
mod any_dist;

fn main() {
    
    let option = CmdChooser::parse();

    match option{
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
        },
        CmdChooser::SimpleFirmBufHist(opt) => {
            if opt.print_alternatives{
                SimpleFirmBufferHistogram::print_alternatives(0);
            } else {
                let o: SimpleFirmBufferHistogram = parse(opt.json);
                o.exec();
            }  
        },
        CmdChooser::SimpleOtherFirmDifK(opt) => {
            if opt.print_alternatives{
                SimpleFirmDifferentKOpts::print_alternatives(0);
            } else {
                let o: SimpleFirmDifferentKOpts = parse(opt.json);
                simple_firm::different_k_with_max(&o);
            }
        },
        CmdChooser::SimpleFirmAverage(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::average_delay_measurement(&o);
            }
        },
        CmdChooser::SimpleFirmAverageOrder(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::average_delay_order_measurement(&o);
            }
        },
        CmdChooser::SimpleFirmAverageOrderMoran(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::recreate_moran(&o);
            }
        },
        CmdChooser::SimpleFirmAverageOrderMoranAvalanch(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::recreate_moran_avalanch(&o);
            }
        }

    }
    
}

