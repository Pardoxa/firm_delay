use rand::prelude::Distribution;
use serde_json::Value;
use super::measure_phase;

use {
    serde::{Serialize, Deserialize},
    std::{
        num::*,
        io::{Write, BufWriter},
        fs::File
    },
    crate::misc::*
};

#[derive(Serialize, Deserialize, Clone)]
pub struct SimpleFirmDifferentKOpts{
    /// vector containing all k values
    pub k: Vec<NonZeroUsize>,

    /// Buffer of the firm
    pub buffer: f64,

    pub delta: f64,

    /// How many time steps to iterate
    pub iter_limit: NonZeroU64,

    /// Seed for the random number generator
    pub seed: u64
}

impl Default for SimpleFirmDifferentKOpts{
    fn default() -> Self {
        Self { 
            k: vec![NonZeroUsize::new(1).unwrap(), NonZeroUsize::new(10).unwrap(), NonZeroUsize::new(100).unwrap()], 
            buffer: 0.5, 
            delta: 0.49, 
            iter_limit: NonZeroU64::new(1000).unwrap(),
            seed: 294
        }
    }
}

impl SimpleFirmDifferentKOpts{
    pub fn get_name(&self) -> String
    {
        let version = crate::misc::VERSION;

        let ks = if self.k.is_empty(){
            panic!("Invalid! empty k");
        } else if self.k.len() <= 10 {
            let mut s = format!("{}", self.k[0]);
            for k in self.k.iter().skip(1)
            {
                s = format!("{s}_{k}");
            }
            s
        } else {
            let len = self.k.len();
            let start = self.k[0];
            let end = self.k.last().unwrap();
            format!("{start}_l{len}_{end}")
        };

        format!(
            "K_v{version}_b{}_d{}_k{ks}_it{}.dat",
            self.buffer,
            self.delta,
            self.iter_limit
        )
    }

    pub fn get_buf(&self) -> BufWriter<File>
    {
        let name = self.get_name();
        let file = File::create(name)
            .unwrap();
        let mut buf = BufWriter::new(file);
        writeln!(buf, "#Version {VERSION}").unwrap();
        write_commands(&mut buf)
            .expect("write error");
        let val: Value = serde_json::to_value(self.clone())
            .expect("serialization error");
        write_json(&mut buf, &val);
        buf
    }

    pub fn exec(self) 
    {
        super::different_k(&self)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SimpleFirmPhase
{
    pub k_start: NonZeroUsize,
    pub k_end: NonZeroUsize,
    pub k_step_by: NonZeroUsize,
    pub delta: f64,
    pub buffer: Vec<f64>,
    pub iter_limit: NonZeroU64,
    pub seed: u64
}

impl Default for SimpleFirmPhase{
    fn default() -> Self {
        Self{
            k_start: NonZeroUsize::new(1).unwrap(),
            k_end: NonZeroUsize::new(10000).unwrap(),
            k_step_by: NonZeroUsize::new(20).unwrap(),
            delta: 0.5,
            buffer: vec![0.5],
            iter_limit: NonZeroU64::new(2000).unwrap(),
            seed: 2849170
        }
    }
}

impl SimpleFirmPhase{
    pub fn get_name(&self, idx: usize) -> String
    {
        let version = crate::misc::VERSION;

        let ks = format!(
                "{}_s{}_{}",
                self.k_start,
                self.k_step_by,
                self.k_end
            );

        let b = self.buffer.get(idx)
            .expect("buffer index out of bounds");

        format!(
            "P_v{version}_b{b}_d{}_k{ks}_it{}.dat",
            self.delta,
            self.iter_limit
        )
    }

    pub fn get_buf(&self, idx: usize) -> BufWriter<File>
    {
        let name = self.get_name(idx);
        let file = File::create(name)
            .unwrap();
        let mut buf = BufWriter::new(file);
        writeln!(buf, "#Version {VERSION}").unwrap();
        write_commands(&mut buf)
            .expect("write error");
        let val: Value = serde_json::to_value(self.clone())
            .expect("serialization error");
        write_json(&mut buf, &val);
        buf
    }

    pub fn exec(&self) 
    {
        measure_phase(self)
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
struct UniformBuffer{
    mean: f64,
    half_width: f64
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
struct ExpBuffer{
    lambda: f64
}
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BufferDist{
    Uniform(UniformBuffer),
    Exponential(ExpBuffer)
}

impl BufferDist{
    pub fn create_uniform_dist(&self) -> impl Distribution<f64>
    {
        match self {
            BufferDist::Uniform(uni) => {
                let min = uni.mean - uni.half_width;
                let max = uni.mean + uni.half_width;
                if min < 0.0 {
                    panic!("Uniform distribution could have return negative Buffers! Abbort!");
                }
                rand::distributions::Uniform::new_inclusive(min, max)
            },
            BufferDist::Exponential(_) => {
                panic!("Trying to get Uniform dist - But Exponential was requested!")
            }
        }
    }

    pub fn create_exp_dist(&self) -> impl Distribution<f64>
    {
        match self {
            BufferDist::Uniform(_) => {
                panic!("Trying to get Exponential dist - But Uniform was requested!")
            },
            BufferDist::Exponential(exp) => {
                rand_distr::Exp::new(exp.lambda)
                    .expect("Negative lambda not allowed")
            }
        }
    }
}
#[derive(Serialize, Deserialize, Debug, Clone)]
struct SimpleFirmPhaseBufferNoise{
    pub k_start: NonZeroUsize,
    pub k_end: NonZeroUsize,
    pub k_step_by: NonZeroUsize,
    pub delta: f64,
    pub buffer_focus: f64,
    pub iter_limit: NonZeroU64,
    pub seed: u64,
    pub buffer_dist: BufferDist
}