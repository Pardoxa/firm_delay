use std::process::exit;

use complexer_firms::{auto, AutoOpts, SubstitutionVelocitySampleOpts};
use correlations::calc_correlations;
use global_opts::{Helper, MyModelCommand, SimpleCommand, SubstitutingCommand};

use crate::complexer_firms::{SelfLinks, SubstitutionVelocityVideoOpts};

use {
    clap::Parser,
    global_opts::CmdChooser,
    misc::*,
    simple_firm::{
        SimpleFirmDifferentKOpts, 
        SimpleFirmPhase, 
        SimpleFirmBufferHistogram,
        SimpleFirmAverageAfter
    }
};
pub mod complexer_firms;
pub mod misc;
pub mod index_sampler;
pub mod correlations;
mod my_model;
use my_model::{DemandVelocityCritOpt, DemandVelocityOpt};
mod global_opts;
mod simple_firm;
mod any_dist;
mod config_helper;

fn simple_chooser(opt: SimpleCommand)
{
    match opt{
        SimpleCommand::SimpleFirmDifK(opt) => {
            if opt.print_alternatives{
                SimpleFirmDifferentKOpts::print_alternatives(0);
            } else {
                let o: SimpleFirmDifferentKOpts = parse_and_add_to_global(opt.json);
                o.exec();
            }
        },
        SimpleCommand::SimpleFirmPhase(opt) => {
            if opt.print_alternatives{
                SimpleFirmPhase::print_alternatives(0);
            } else {
                let o: SimpleFirmPhase = parse_and_add_to_global(opt.json);
                o.exec();
            }   
        },
        SimpleCommand::SimpleFirmBufHist(opt) => {
            if opt.print_alternatives{
                SimpleFirmBufferHistogram::print_alternatives(0);
            } else {
                let o: SimpleFirmBufferHistogram = parse_and_add_to_global(opt.json);
                o.exec();
            }  
        },
        SimpleCommand::SimpleOtherFirmDifK(opt) => {
            if opt.print_alternatives{
                SimpleFirmDifferentKOpts::print_alternatives(0);
            } else {
                let o: SimpleFirmDifferentKOpts = parse_and_add_to_global(opt.json);
                simple_firm::different_k_with_max(&o);
            }
        },
        SimpleCommand::SimpleFirmAverage(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse_and_add_to_global(opt.json);
                simple_firm::average_delay_measurement(&o);
            }
        },
        SimpleCommand::SimpleFirmAverageOrder(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse_and_add_to_global(opt.json);
                simple_firm::average_delay_order_measurement(&o);
            }
        },
        SimpleCommand::SimpleFirmAverageOrderMoran(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse_and_add_to_global(opt.json);
                simple_firm::recreate_moran(&o);
            }
        },
        SimpleCommand::SimpleFirmAverageOrderMoranAvalanch(opt) => {
            if opt.print_alternatives{
                SimpleFirmAverageAfter::print_alternatives(0);
            } else {
                let o: SimpleFirmAverageAfter = parse_and_add_to_global(opt.json);
                simple_firm::recreate_moran_avalanch(&o);
            }
        },
    }
}

fn sub_chooser(opt: SubstitutingCommand)
{
    match opt{
        SubstitutingCommand::Velocity(opt) =>
        {
            let o: SubstitutionVelocitySampleOpts = parse_and_add_to_global(opt.json);
            let out = opt.out_stub.as_deref().unwrap();
            complexer_firms::sample_velocity(&o, out, opt.gnuplot);
        },
        SubstitutingCommand::CritBVideo(opt) => {
            if opt.print_alternatives{
                SubstitutionVelocityVideoOpts::print_alternatives(0);
                exit(0);
            }
            let o: SubstitutionVelocityVideoOpts = parse_and_add_to_global(opt.json);
            let out = opt.out_stub.as_deref().unwrap();
            complexer_firms::sample_velocity_video(
                &o, 
                out, 
                opt.framerate,
                opt.no_clean,
                opt.convert_video
            )
        },
        SubstitutingCommand::CritBVideoNetworks(opt) => {
            if opt.print_alternatives{
                SubstitutionVelocityVideoOpts::print_alternatives(0);
                exit(0);
            }
            
            let o: SubstitutionVelocityVideoOpts = parse_and_add_to_global(opt.json);
            if !matches!(o.self_links, SelfLinks::AllowSelfLinks)
            {
                assert_eq!(
                    o.self_links,
                    SelfLinks::AlwaysSelfLink
                );
                println!(
                    "Careful: This is only implemented for cycle networks."
                );
            }
            let out = opt.out_stub.as_deref().unwrap();
            complexer_firms::sample_ring_velocity_video(
                &o, 
                out, 
                opt.framerate,
                opt.no_clean,
                opt.convert_video,
                opt.structure
            )
        },
        SubstitutingCommand::Auto(opt) => {
            if opt.print_alternatives{
                AutoOpts::print_alternatives(0);
                exit(0);
            }
            let auto_opt = parse_and_add_to_global(opt.json);
            auto(&auto_opt, opt.output.as_deref().unwrap(), opt.j)
        }
    }
}

fn helper_chooser(opts: Helper)
{
    match opts{
        Helper::SubAutoBufSwap(opt) => config_helper::exec_auto_sub_buf_swapper(opt),
        Helper::SubAutoKanta(kanta) => config_helper::exec_auto_buf_sub_job_creator(kanta)
    }
}

fn main() {
    
    let option = CmdChooser::parse();

    match option{
        CmdChooser::Single(simple_command) =>
        {
            simple_chooser(simple_command)
        },
        CmdChooser::Substituting(opt) => {
            sub_chooser(opt)
        },
        CmdChooser::Auto(auto_opt) => {
            calc_correlations(auto_opt)
        },
        CmdChooser::Helper(helper_opt) => {
            helper_chooser(helper_opt)
        },
        CmdChooser::Test => {
            my_model::test_profile()
        },
        CmdChooser::MyModel(sub) => {
            match sub {
                MyModelCommand::ChainVelocity(vel) => {
                    let opts: DemandVelocityOpt = parse_and_add_to_global(vel.json);
                    my_model::chain_calc_demand_velocity(opts, vel.out.unwrap());
                },
                MyModelCommand::ChainCrit(opt) => {
                    let opts: DemandVelocityCritOpt = parse_and_add_to_global(opt.json);
                    my_model::chain_crit_scan(opts, opt.out.unwrap().as_str());
                }
            }
        }

    }
    
}

