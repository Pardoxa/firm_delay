use rand_distr::{Exp, Distribution};
use rand_pcg::Pcg64;
use rand::{Rng, SeedableRng};
use serde::{Serialize, Deserialize};
use crate::index_sampler::IndexSampler;
use crate::misc::*;
use std::io::Write;

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct SubstitutionVelocitySampleOpts{
    pub buffer: SampleRangeF64,
    pub opts: SubstitutingMeanFieldOpts,
    pub time_steps: usize
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct SubstitutingMeanFieldOpts{
    buffer: f64,
    substitution_prob: f64,
    seed: u64,
    k: usize,
    n: usize,
    lambda: f64
}

impl SubstitutingMeanFieldOpts{
    pub fn get_buffers(&self) -> Vec<f64>
    {
        vec![self.buffer; self.n]
    }

    pub fn get_substitution_prob(&self) -> Vec<f64>
    {
        vec![self.substitution_prob; self.n]
    }
}

pub struct SubstitutingMeanField{
    current_delays: Vec<f64>,
    buffers: Vec<f64>,
    substitution_prob: Vec<f64>,
    next_delays: Vec<f64>,
    k: usize,
    rng: Pcg64,
    index_sampler: IndexSampler,
    dist: Exp<f64>
}

impl SubstitutingMeanField{

    pub fn reset_delays(&mut self)
    {
        self.current_delays.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn change_buffer_to_const(&mut self, const_val: f64)
    {
        self.buffers.iter_mut()
            .for_each(|v| *v = const_val)
    }

    pub fn get_k(&self) -> usize {
        self.k
    }

    pub fn new(opt: &SubstitutingMeanFieldOpts) -> Self
    {
        let current_delays = vec![0.0; opt.n];
        let next_delays = vec![0.0; opt.n];
        let mut rng = Pcg64::seed_from_u64(opt.seed);
        let index_sampler = IndexSampler::measure_which(
            opt.n, 
            opt.k, 
            &mut rng
        );
        let exp = Exp::new(opt.lambda)
            .unwrap();
        Self{
            current_delays,
            next_delays,
            buffers: opt.get_buffers(),
            k: opt.k,
            index_sampler,
            substitution_prob: opt.get_substitution_prob(),
            rng,
            dist: exp
        }
    }

    pub fn step(&mut self)
    {
        self.next_delays.iter_mut()
            .enumerate()
            .for_each(
                |(index, n_delay)|
                {
                    if self.rng.gen::<f64>() < self.substitution_prob[index]{
                        *n_delay = self.dist.sample(&mut self.rng);
                    } else {
                        let mut current = 0.0_f64;
                        todo!("NOTE: Marc is allowiung self liks, so maybe use sample_indices instead of without?");
                        for i in self.index_sampler.sample_indices_without(&mut self.rng, index as u32){
                            let i = *i as usize;
                            current = current.max(self.current_delays[i]);
                        }
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + self.dist.sample(&mut self.rng);
                    }
                    
                }
            );
        std::mem::swap(&mut self.current_delays, &mut self.next_delays);
    }

    pub fn average_delay(&self) -> f64{
        self.current_delays.iter().sum::<f64>() / self.current_delays.len() as f64
    }
}

pub fn sample_velocity(opt: &SubstitutionVelocitySampleOpts){
    let mut writer = create_buf_with_command_and_version("test.dat");
    let header = ["B", "Velocity"];
    write_slice_head(&mut writer, header).unwrap();

    let mut model = SubstitutingMeanField::new(&opt.opts);

    for b in opt.buffer.get_iter(){
        model.change_buffer_to_const(b);
        model.reset_delays();
        for _ in 0..opt.time_steps{
            model.step();
        }
        let velocity = model.average_delay() / opt.time_steps as f64;
        writeln!(writer, "{b} {velocity}").unwrap();
    }
}