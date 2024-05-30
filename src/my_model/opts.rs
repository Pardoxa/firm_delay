use derivative::Derivative;
use serde::{Serialize, Deserialize};
use std::{
    num::*,
    ops::RangeInclusive,
    process::exit,
    io::{
        stdin,
        stdout
    }
};

use crate::{create_buf, global_opts::*};


#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct TreeDemandVelocityCritOpt{
    pub opts: TreeDemandVelocityOpt,

    pub tree_depth_start: usize,
    pub tree_depth_end: usize,

    #[derivative(Default(value="Some(0.0..=1.0)"))]
    pub y_range: Option<RangeInclusive<f64>>
}


#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct TreeDemandVelocityOpt{
    #[derivative(Default(value="0.0"))]
    pub root_demand_rate_min: f64,
    #[derivative(Default(value="1.0"))]
    pub root_demand_rate_max: f64,
    #[derivative(Default(value="NonZeroI64::new(100).unwrap()"))]
    pub root_demand_samples: NonZeroI64,
    #[derivative(Default(value="NonZeroU64::new(10000).unwrap()"))]
    pub time: NonZeroU64,
    #[derivative(Default(value="NonZeroUsize::new(100).unwrap()"))]
    pub samples: NonZeroUsize,

    pub tree_depth: usize,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub num_children: NonZeroUsize,
    pub seed: u64,
    pub threads: Option<NonZeroUsize>,
    #[derivative(Default(value="1.0"))]
    pub max_stock: f64
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct RandTreeDemandVelocityCritOpt{
    pub opts: RandTreeDemandVelocityOpt,

    pub tree_depth_start: usize,
    pub tree_depth_end: usize,

    #[derivative(Default(value="Some(0.0..=1.0)"))]
    pub y_range: Option<RangeInclusive<f64>>
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct RandTreeDemandVelocityOpt{
    #[derivative(Default(value="0.0"))]
    pub root_demand_rate_min: f64,
    #[derivative(Default(value="1.0"))]
    pub root_demand_rate_max: f64,
    #[derivative(Default(value="NonZeroI64::new(100).unwrap()"))]
    pub root_demand_samples: NonZeroI64,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub samples_per_tree: NonZeroUsize,

    #[derivative(Default(value="NonZeroU64::new(10000).unwrap()"))]
    pub time: NonZeroU64,
    #[derivative(Default(value="NonZeroUsize::new(100).unwrap()"))]
    pub samples: NonZeroUsize,

    pub seed: u64,
    pub threads: Option<NonZeroUsize>,
    #[derivative(Default(value="1.0"))]
    pub max_stock: f64,

    #[derivative(Default(value="WhichDistr::Uniform(UniformParser{start: 0, end: 1})"))]
    pub distr: WhichDistr
}

pub fn print_jsons_rand_tree_crit_scan() -> !
{
    let options = [
        (0, "Constant value"),
        (1, "Uniform Distribution")
    ];
    let mut buffer = String::new();
    let stdin = stdin();
    let which = loop {
        for (num, str) in options.iter(){
            println!(
                "{num}) {str}"
            );
        }
        println!("Enter number:");
        buffer.clear();
        stdin.read_line(&mut buffer).unwrap();
        match buffer.trim(){
            "0" | "0)" => {
                break WhichDistr::Constant(ConstantParser { value: 1 });
            },
            "1" | "1)" => {
                break WhichDistr::Uniform(UniformParser { start: 1, end: 5 });
            },
            _ => {
                println!(
                    "Input invalid. Try again."
                );
            }
        }
    };

    let opt = RandTreeDemandVelocityOpt { 
        distr: which, 
        ..Default::default() 
    };
    
    serde_json::to_writer_pretty(stdout(), &opt).unwrap();

    println!("\nDo you want to save this to a file? If so, write filename. Otherwise just press enter");
    buffer.clear();
    stdin.read_line(&mut buffer).unwrap();
    let trimmed = buffer.trim();
    if !trimmed.is_empty(){
        let buf = create_buf(trimmed);
        serde_json::to_writer_pretty(buf, &opt).unwrap();
    }
    exit(0)
}


#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct ClosedMultiChainCritOpts{
    pub opts: ClosedMultiChainVelocityOpts,

    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub chain_len_start: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(2).unwrap()"))]
    pub chain_len_end: NonZeroUsize,

    #[derivative(Default(value="Some(0.0..=1.0)"))]
    pub y_range: Option<RangeInclusive<f64>>
}


#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct ClosedMultiChainCritOpts2{
    pub opts: ClosedMultiChainVelocityOpts,

    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub num_chains_start: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(2).unwrap()"))]
    pub num_chains_end: NonZeroUsize,

    #[derivative(Default(value="Some(0.0..=1.0)"))]
    pub y_range: Option<RangeInclusive<f64>>
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct ClosedMultiChainVelocityOpts{
    #[derivative(Default(value="0.0"))]
    pub root_demand_rate_min: f64,
    #[derivative(Default(value="1.0"))]
    pub root_demand_rate_max: f64,
    #[derivative(Default(value="NonZeroI64::new(100).unwrap()"))]
    pub root_demand_samples: NonZeroI64,
    #[derivative(Default(value="NonZeroU64::new(10000).unwrap()"))]
    pub time: NonZeroU64,
    #[derivative(Default(value="NonZeroUsize::new(100).unwrap()"))]
    pub samples: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub other_chain_len: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub num_chains: NonZeroUsize,
    pub seed: u64,
    pub threads: Option<NonZeroUsize>,
    #[derivative(Default(value="1.0"))]
    pub max_stock: f64
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct DemandVelocityCritOpt{
    pub opts: DemandVelocityOpt,

    #[derivative(Default(value="NonZeroUsize::new(2).unwrap()"))]
    pub chain_start: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(100).unwrap()"))]
    pub chain_end: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub chain_step: NonZeroUsize,

    #[derivative(Default(value="Some(0.0..=1.0)"))]
    pub y_range: Option<RangeInclusive<f64>>
}


#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct DemandVelocityOpt{
    #[derivative(Default(value="0.0"))]
    pub root_demand_rate_min: f64,
    #[derivative(Default(value="1.0"))]
    pub root_demand_rate_max: f64,
    #[derivative(Default(value="NonZeroI64::new(100).unwrap()"))]
    pub root_demand_samples: NonZeroI64,
    #[derivative(Default(value="NonZeroU64::new(10000).unwrap()"))]
    pub time: NonZeroU64,
    #[derivative(Default(value="NonZeroUsize::new(100).unwrap()"))]
    pub samples: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(10).unwrap()"))]
    pub chain_length: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub num_chains: NonZeroUsize,
    pub seed: u64,
    pub threads: Option<NonZeroUsize>,
    #[derivative(Default(value="1.0"))]
    pub max_stock: f64
}

#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
pub struct ChainProfileOpts{
    #[derivative(Default(value="NonZeroUsize::new(4).unwrap()"))]
    pub total_len: NonZeroUsize,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub num_chains: NonZeroUsize,
    pub seed: u64,
    #[derivative(Default(value="0.5"))]
    pub root_demand: f64,
    #[derivative(Default(value="1.0"))]
    pub max_stock: f64,
    #[derivative(Default(value="NonZeroU32::new(200).unwrap()"))]
    pub time_steps: NonZeroU32,
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub average_over_samples: NonZeroUsize
}