use complexer_firms::SubstitutionVelocitySampleOpts;
use global_opts::{SimpleCommand, SubstitutingCommand};

use crate::complexer_firms::SubstitutionVelocityVideoOpts;

use {
    clap::Parser,
    global_opts::CmdChooser,
    misc::*,
    simple_firm::{
        SimpleFirmDifferentKOpts, 
        SimpleFirmPhase, 
        SimpleFirmBufferHistogram,
        SimpleFirmAverageAfter
    },
    crate::global_opts::RandomState
};
pub mod complexer_firms;
pub mod misc;
pub mod index_sampler;
mod global_opts;
mod simple_firm;
mod any_dist;

fn simple_chooser(opt: SimpleCommand)
{
    match opt{
        SimpleCommand::SimpleFirmDifK(opt) => {
            if opt.print_alternatives{
                SimpleFirmDifferentKOpts::print_alternatives(0);
            } else {
                let o: SimpleFirmDifferentKOpts = parse(opt.json);
                o.exec();
            }
        },
        SimpleCommand::SimpleFirmPhase(opt) => {
            if opt.print_alternatives{
                SimpleFirmPhase::print_alternatives(0);
            } else {
                let o: SimpleFirmPhase = parse(opt.json);
                o.exec();
            }   
        },
        SimpleCommand::SimpleFirmBufHist(opt) => {
            if opt.print_alternatives{
                SimpleFirmBufferHistogram::print_alternatives(0);
            } else {
                let o: SimpleFirmBufferHistogram = parse(opt.json);
                o.exec();
            }  
        },
        SimpleCommand::SimpleOtherFirmDifK(opt) => {
            if opt.print_alternatives{
                SimpleFirmDifferentKOpts::print_alternatives(0);
            } else {
                let o: SimpleFirmDifferentKOpts = parse(opt.json);
                simple_firm::different_k_with_max(&o);
            }
        },
        SimpleCommand::SimpleFirmAverage(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::average_delay_measurement(&o);
            }
        },
        SimpleCommand::SimpleFirmAverageOrder(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::average_delay_order_measurement(&o);
            }
        },
        SimpleCommand::SimpleFirmAverageOrderMoran(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::recreate_moran(&o);
            }
        },
        SimpleCommand::SimpleFirmAverageOrderMoranAvalanch(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse(opt.json);
                simple_firm::recreate_moran_avalanch(&o);
            }
        },
    }
}

fn sub_chooser(opt: SubstitutingCommand)
{
    match opt{
        SubstitutingCommand::SubMean(opt) =>
        {
            let o: SubstitutionVelocitySampleOpts = parse(opt.json);
            let out = opt.out_stub.as_deref().unwrap();
            complexer_firms::sample_velocity(&o, out);
        },
        SubstitutingCommand::CritBVideo(opt) => {
            let o: SubstitutionVelocityVideoOpts = parse(opt.json);
            let out = opt.out_stub.as_deref().unwrap();
            match opt.randomness{
                RandomState::Dynamic => {
                    complexer_firms::sample_velocity_video(
                        &o, 
                        out, 
                        opt.framerate
                    )
                },
                RandomState::Quenched => {
                    complexer_firms::quenched_substituting_firms::sample_velocity_video(
                        &o, 
                        out, 
                        opt.framerate
                    )
                }
            }
        }
    }
}

fn main() {
    
    let option = CmdChooser::parse();

    match option{
        CmdChooser::Single(simple_command) =>
        {
            simple_chooser(simple_command)
        },
        CmdChooser::Sub(opt) => {
            sub_chooser(opt)
        }

    }
    
}

