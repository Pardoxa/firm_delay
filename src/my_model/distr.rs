use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64;



pub trait MyDistr{
    fn rand_amount(&self, rng: &mut Pcg64) -> usize;
}

impl MyDistr for Uniform<usize>{
    fn rand_amount(&self, rng: &mut Pcg64) -> usize {
        self.sample(rng)
    }
}

impl MyDistr for usize {
    fn rand_amount(&self, _: &mut Pcg64) -> usize {
        *self
    }
}