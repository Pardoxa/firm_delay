use derivative::Derivative;
use serde::{Serialize, Deserialize};
use std::num::*;
use std::ops::RangeInclusive;


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