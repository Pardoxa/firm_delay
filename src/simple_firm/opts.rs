use crate::any_dist::*;
use serde_json::Value;
use super::{measure_phase, sample_simple_firm_buffer_hist};

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
    pub k: Vec<usize>,

    /// Buffer of the firm
    pub focus_buffer: f64,

    /// Buffer dist for others
    pub buf_dist: AnyBufDist,

    /// How many time steps to iterate
    pub iter_limit: NonZeroU64,

    /// Seed for the random number generator
    pub seed: u64,

    pub delay_dist: AnyDistCreator,

    pub write_every: Option<u64>
}

impl PrintAlternatives for SimpleFirmDifferentKOpts{
    fn print_alternatives(layer: u8) {
        print_spaces(layer);
        println!("Alternatives for DelayDist:");
        print_spaces(layer);
        AnyDistCreator::print_alternatives(layer + 1);
        println!("Alternatives for BufDist:");
        AnyBufDist::print_alternatives(layer+1);
    }
}

impl Default for SimpleFirmDifferentKOpts{
    fn default() -> Self {
        Self { 
            k: vec![0, 10, 100], 
            focus_buffer: 0.5,
            iter_limit: NonZeroU64::new(1000).unwrap(),
            seed: 294,
            delay_dist: AnyDistCreator::default(),
            buf_dist: AnyBufDist::default(),
            write_every: Some(1)
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
            "K_v{version}_b{}_{}_D{}_k{ks}_it{}.dat",
            self.focus_buffer,
            self.buf_dist.get_name(),
            self.delay_dist.get_name(),
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

    pub fn get_buf2(&self) -> BufWriter<File>
    {
        let name = format!("{}2", self.get_name());
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
    pub delay_dist: AnyDist,
    pub focus_buffer: Vec<f64>,
    pub buffer_dist: AnyBufDist,
    pub iter_limit: NonZeroU64,
    pub seed: u64
}

impl PrintAlternatives for SimpleFirmPhase{
    fn print_alternatives(layer: u8) {
        print_spaces(layer);
        println!("Alternatives for buffer dist:");
        AnyDistCreator::print_alternatives(layer + 1);
        print_spaces(layer);
        println!("Alternatives for delay_dist:");
        AnyDist::print_alternatives(layer + 1);
    }
}

impl Default for SimpleFirmPhase{
    fn default() -> Self {
        Self{
            k_start: NonZeroUsize::new(1).unwrap(),
            k_end: NonZeroUsize::new(10000).unwrap(),
            k_step_by: NonZeroUsize::new(20).unwrap(),
            focus_buffer: vec![0.5],
            iter_limit: NonZeroU64::new(2000).unwrap(),
            seed: 2849170,
            delay_dist: AnyDist::default(),
            buffer_dist: AnyBufDist::default()
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

        let b = self.focus_buffer.get(idx)
            .expect("buffer index out of bounds");

        format!(
            "P_v{version}_b{b}_B{}_d{}_k{ks}_it{}.dat",
            self.buffer_dist.get_name(),
            self.delay_dist.get_name(),
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

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SimpleFirmPhaseBufferNoise{
    pub k_start: NonZeroUsize,
    pub k_end: NonZeroUsize,
    pub k_step_by: NonZeroUsize,
    pub delta: f64,
    pub buffer_focus: f64,
    pub iter_limit: NonZeroU64,
    pub seed: u64,
    pub buffer_dist: AnyDist
}



#[derive(Serialize, Deserialize, Clone)]
pub struct SimpleFirmBufferHistogram{
    /// vector containing all k values
    pub k: usize,

    /// Buffer of the firm
    pub focus_buffer: f64,

    /// Buffer dist for others
    pub buf_dist: UniformDistCreator2,

    /// How many time steps to iterate
    pub iter_limit: NonZeroU64,

    /// Seed for the random number generator
    pub seed: u64,

    pub delay_dist: AnyDistCreator,

    pub samples: usize,

    pub bins: usize,

    pub threads: usize
}

impl PrintAlternatives for SimpleFirmBufferHistogram{
    fn print_alternatives(layer: u8) {
        print_spaces(layer);
        println!("Alternatives for DelayDist:");
        print_spaces(layer);
        AnyDistCreator::print_alternatives(layer + 1);
        println!("Alternatives for BufDist:");
        AnyBufDist::print_alternatives(layer+1);
    }
}

impl Default for SimpleFirmBufferHistogram{
    fn default() -> Self {
        Self { 
            k: 1000, 
            focus_buffer: 0.5,
            iter_limit: NonZeroU64::new(1000).unwrap(),
            seed: 294,
            delay_dist: AnyDistCreator::default(),
            buf_dist: UniformDistCreator2{mean: 0.5, half_width: 0.1},
            samples: 10000,
            bins: 100,
            threads: 8

        }
    }
}

impl SimpleFirmBufferHistogram{
    pub fn get_name(&self) -> String
    {
        let version = crate::misc::VERSION;

        
        let k = self.k;
        format!(
            "H_v{version}_b{}_{}_D{}_k{k}_it{}.dat",
            self.focus_buffer,
            self.buf_dist.get_name(),
            self.delay_dist.get_name(),
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

    pub fn exec(&self) 
    {
        sample_simple_firm_buffer_hist(self)
    }
}