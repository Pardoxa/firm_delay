use serde::{Serialize, Deserialize};
use rand::distributions::{Uniform, Distribution};
use std::io::stdout;

use crate::misc::{PrintAlternatives, print_spaces};
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AnyDist {
    Uniform(UniformDistCreator),
    Exponential(ExponentialDist)
}

impl PrintAlternatives for AnyDist{
    fn print_alternatives(layer: u8) {
        let a = AnyDist::Uniform(UniformDistCreator { min: 0.0, max: 1.0 });
        let b = AnyDist::Exponential(ExponentialDist { lambda: 1.0 });
        print_spaces(layer);
        println!("a)");
        let mut stdout = stdout();
        serde_json::to_writer_pretty(&mut stdout, &a)
            .expect("cannot create json a)");
        println!();
        print_spaces(layer);
        println!("b)");
        serde_json::to_writer_pretty(stdout, &b)
            .expect("cannot create json b)");
        println!();
    }
}

impl Default for AnyDist{
    fn default() -> Self {
        AnyDist::Uniform(UniformDistCreator{min: 0.0, max: 0.5})
    }
}

impl AnyDist{
    pub fn get_name(&self) -> String
    {
        match self{
            Self::Uniform(uni) => {
                uni.get_name()
            },
            Self::Exponential(exp) => {
                exp.get_name()
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct UniformDistCreator2 {
    mean: f64,
    half_width: f64
}

impl UniformDistCreator2{
    pub fn get_name(&self) -> String
    {
        format!("UM{}_{}", self.mean, self.half_width)
    }
}


#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct UniformDistCreator{
    min: f64,
    max: f64
}

impl From<UniformDistCreator2> for UniformDistCreator
{
    fn from(value: UniformDistCreator2) -> Self {
        let min = value.mean - value.half_width;
        let max = value.mean + value.half_width;
        UniformDistCreator{
            max,
            min
        }
    }
}

impl UniformDistCreator{
    pub fn create_dist(&self) -> Uniform<f64>
    {
        if self.min < 0.0 {
            panic!("Uniform distribution could have return negative Buffers! Abort!");
        }
        if self.max < self.min{
            panic!("Self max cannot be smaller than self min!");
        }
        Uniform::new_inclusive(self.min, self.max)
    }

    pub fn get_name(&self) -> String
    {
        format!("U{}_{}", self.min, self.max)
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct ExponentialDist {
    lambda: f64
}

impl ExponentialDist{
    pub fn create_dist(&self) -> impl Distribution<f64>
    {
        rand_distr::Exp::new(self.lambda)
            .expect("Negative lambda not allowed")
    }

    pub fn get_name(&self) -> String
    {
        format!("Exp{}", self.lambda)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AnyDistCreator{
    Uniform(UniformDistCreator),
    Uniform2(UniformDistCreator2),
    Exponential(ExponentialDist)
}

impl PrintAlternatives for AnyDistCreator{
    fn print_alternatives(layer: u8) {
        let a = AnyDistCreator::Uniform(UniformDistCreator { min: 0.0, max: 1.0 });
        let b = AnyDistCreator::Uniform2(UniformDistCreator2 { mean: 0.5, half_width: 0.1 });
        let c = AnyDistCreator::Exponential(ExponentialDist { lambda: 1.0 });
        let mut stdout = stdout();
        print_spaces(layer);
        println!("a)");
        serde_json::to_writer_pretty(&mut stdout, &a)
            .expect("unable to create json a");
        println!();
        print_spaces(layer);
        println!("b)");
        serde_json::to_writer_pretty(&mut stdout, &b)
            .expect("unable to create json b");
        println!();
        print_spaces(layer);
        println!("c)");
        serde_json::to_writer_pretty(stdout, &c)
            .expect("unable to create json c");
        println!();
    }
}

impl Default for AnyDistCreator{
    fn default() -> Self {
        AnyDistCreator::Uniform(UniformDistCreator{min: 0.0, max: 0.5})
    }
}

impl AnyDistCreator{
    pub fn get_name(&self) -> String
    {
        match self{
            Self::Uniform(uni) => uni.get_name(),
            Self::Exponential(exp) => exp.get_name(),
            Self::Uniform2(uni2) => uni2.get_name()
        }
    }
}

impl From<AnyDistCreator> for AnyDist{
    fn from(value: AnyDistCreator) -> Self {
        match value {
            AnyDistCreator::Uniform(uni) => {
                AnyDist::Uniform(uni)
            },
            AnyDistCreator::Exponential(exp) => {
                AnyDist::Exponential(exp)
            },
            AnyDistCreator::Uniform2(uni2) => {
                AnyDist::Uniform(uni2.into())
            }
        }
    }
}