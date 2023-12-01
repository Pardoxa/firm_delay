use rand_distr::{Exp, Distribution};
use rand_pcg::Pcg64;
use rand::{Rng, SeedableRng};

use crate::index_sampler::IndexSampler;


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
                        for i in self.index_sampler.sample_indices_without(&mut self.rng, index as u32){
                            let i = *i as usize;
                            current = current.max(self.current_delays[i]);
                        }
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + self.dist.sample(&mut self.rng);
                    }
                    
                }
            )
    }
}