use serde::{Serialize, Deserialize};
use rand::distributions::Uniform;

use crate::misc::*;
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AnyDist {
    Uniform(UniformDistCreator),
    Exponential(ExponentialDist)
}

impl PrintAlternatives for AnyDist{
    fn print_alternatives(layer: u8) {
        let a = AnyDist::Uniform(UniformDistCreator { min: 0.0, max: 1.0 });
        let b = AnyDist::Exponential(ExponentialDist { lambda: 1.0 });

        let all = [a, b];
        print_alternatives_helper(&all, layer, "AnyDist");
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
    pub mean: f64,
    pub half_width: f64
}

impl UniformDistCreator2{
    pub fn get_name(&self) -> String
    {
        format!("UM{}_{}", self.mean, self.half_width)
    }

    pub fn is_valid(&self) -> bool
    {
        (self.mean - self.half_width) >= 0.0
    }
}


#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
pub struct UniformDistCreator{
    pub min: f64,
    pub max: f64
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
    pub fn create_dist(&self) -> rand_distr::Exp<f64>
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
pub struct BufferConstMin{
    pub buf_const: f64,
    pub buf_min: f64
}

impl BufferConstMin{
    pub fn get_name(&self) -> String
    {
        format!("C{}M{}", self.buf_const, self.buf_min)
    }
}


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BufferConstMinFrac{
    pub buf_const: f64,
    pub buf_min: f64,
    pub min_frac: f64
}

impl BufferConstMinFrac{
    pub fn get_name(&self) -> String
    {
        format!("C{}M{}f{}", self.buf_const, self.buf_min, self.min_frac)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UniformMax{
    pub buf_max: f64,
    pub uniform: UniformDistCreator2
}

impl UniformMax{
    pub fn get_name(&self) -> String
    {
        format!("{}max{}",self.uniform.get_name(), self.buf_max)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AnyBufDist{
    Any(AnyDistCreator),
    Constant(f64),
    ConstMin(BufferConstMin),
    ConstMinFrac(BufferConstMinFrac),
    UniformMax(UniformMax)
}

impl AnyBufDist{
    pub fn get_name(&self) -> String
    {
        let mut s = "B".to_owned();
        match self{
            Self::Any(any) => {
                let n = any.get_name();
                s.push('A');
                s.push_str(&n);
            },
            Self::Constant(c) => {
                s.push('C');
                s = format!("{s}{c}");
            },
            Self::ConstMin(cm) => {
                s.push_str(&cm.get_name());
            },
            Self::ConstMinFrac(cmf) => {
                s.push_str(&cmf.get_name()); 
            },
            Self::UniformMax(umax) => {
                s.push_str(&umax.get_name())
            }
        }
        s
    }
}

impl Default for AnyBufDist{
    fn default() -> Self {
        Self::Constant(0.5)
    }
}

impl PrintAlternatives for AnyBufDist{
    fn print_alternatives(layer: u8) {
        let a = Self::Any(AnyDistCreator::Uniform(UniformDistCreator { min: 0.0, max: 1.0 }));
        let b = Self::Constant(0.7);
        let c = Self::ConstMin(BufferConstMin{buf_const: 0.5, buf_min: 0.2});
        let d = Self::ConstMinFrac(BufferConstMinFrac { buf_const: 0.5, buf_min: 0.2, min_frac: 0.5 });
        let e = Self::UniformMax(UniformMax { buf_max: 0.5, uniform: UniformDistCreator2 { mean: 0.5, half_width: 0.1 } });

        let all = [a, b, c, d, e];
        print_alternatives_helper(&all, layer, "AnyBufDist");
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

        let all = [a, b, c];
        print_alternatives_helper(&all, layer, "AnyDistCreator");
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
            Self::Uniform2(uni2) => uni2.get_name(),
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