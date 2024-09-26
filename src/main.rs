#![allow(non_snake_case)]

use std::process::exit;

use complexer_firms::{auto, AutoOpts, SubstitutionVelocitySampleOpts};
use correlations::calc_correlations;
use global_opts::{Helper, MyModelCommand, SimpleCommand, SubstitutingCommand};
use numeric_integration::ModelInput;

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
use my_model::*;
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

        },
        CmdChooser::MyModel(sub) => {
            match sub {
                MyModelCommand::ChainVelocity(vel) => {
                    let opts: DemandVelocityOpt = parse_and_add_to_global(vel.json);
                    my_model::chain_calc_demand_velocity(opts, vel.out.unwrap());
                },
                MyModelCommand::ChainCrit(opt) => {
                    let opts: DemandVelocityCritOpt = parse_and_add_to_global(opt.json);
                    my_model::chain_crit_scan(
                        opts, 
                        opt.out.unwrap().as_str(),
                        opt.no_video
                    );
                },
                MyModelCommand::QuenchedChainCrit(opt) => {
                    let opts: DemandVelocityCritOpt = parse_and_add_to_global(opt.json);
                    my_model::quenched_chain_crit_scan(opts, opt.out.unwrap().as_str());
                },
                MyModelCommand::AlternativeQuenchedChainCrit(opt) => {
                    let opts: DemandVelocityCritOpt = parse_and_add_to_global(opt.json);
                    my_model::alternative_quenched_chain_crit_scan(opts, opt.out.unwrap().as_str());
                },
                MyModelCommand::ChainProfile(opt) => {
                    let my_opt: ChainProfileOpts = parse_and_add_to_global(opt.json);
                    my_model::test_profile(my_opt, opt.out.unwrap())
                },
                MyModelCommand::ChainProfileHist(opt) => {
                    let my_opt: ChainProfileHistOpts = parse_and_add_to_global(opt.json);
                    my_model::profile_hist(my_opt, opt.out.unwrap(), opt.list);
                },
                MyModelCommand::TreeCrit(opt) => {
                    let opts: TreeDemandVelocityCritOpt = parse_and_add_to_global(opt.json);
                    my_model::tree_crit_scan(opts, opt.out.unwrap());
                },
                MyModelCommand::RandTreeCrit(opt) => {
                    let opts: RandTreeDemandVelocityCritOpt = match opt.json {
                        None => print_jsons_rand_tree_crit_scan(),
                        Some(_) => parse_and_add_to_global(opt.json)
                    };
                     
                    my_model::rand_tree_crit_scan(opts, opt.out.unwrap());
                },
                MyModelCommand::DotTree(opt) => {
                    my_model::model::write_tree_dot(
                        opt.chilren_per_node, 
                        opt.depth, 
                        &opt.dot_out,
                        opt.reverse_direction
                    );
                },
                MyModelCommand::DotRandTree(opt) => {
                    let distr = opt.which.get_distr();
                    my_model::model::write_rand_tree_dot(
                        opt.max_depth,
                        &opt.dot_out,
                        opt.reverse_direction,
                        opt.seed.unwrap_or_default(),
                        distr
                    );
                },
                MyModelCommand::DotClosedMultiChain(opt) => {
                    my_model::model::write_closed_multi_chain(
                        opt.other_chain_len,
                        opt.num_chains,
                        opt.appendix_nodes,
                        &opt.dot_out
                    )
                },
                MyModelCommand::ClosedMultiChainCrit(opt) => {
                    let my_opt: ClosedMultiChainCritOpts = parse_and_add_to_global(opt.json);
                    my_model::closed_multi_chain_crit_scan(
                        my_opt,
                        opt.out.unwrap()
                    )
                },
                MyModelCommand::ClosedMultiChainCrit2(opt) => {
                    let my_opt: ClosedMultiChainCritOpts2 = parse_and_add_to_global(opt.json);
                    my_model::closed_multi_chain_crit_scan2(
                        my_opt,
                        opt.out.unwrap()
                    )
                },
                MyModelCommand::Num(test) => {
                    let input: ModelInput = parse_and_add_to_global(test.json);
                    if input.triangle{
                        my_model::numeric_integration_pre::compute_line(input)
                    } else {
                        my_model::numeric_integration::compute_line(input);
                    }
                },
                MyModelCommand::TreeVsRandTree(path) => {
                    let options: RandTreeCompareOpts = parse_and_add_to_global(path.json);
                    my_model::model::regular_vs_random_tree(
                        options,
                        path.out.unwrap()
                    );
                },
                MyModelCommand::LineVsTree(path) => {
                    let input: LineVsTreeOpts = parse_and_add_to_global(path.json);
                    my_model::model::line_vs_tree(input);
                }
            }
        }

    }
    
}

