use num_rational::Rational64;
use std::num::*;
use num_traits::cast::ToPrimitive;


pub struct RatioIter{
    start: Rational64,
    end: Rational64,
    // number of samples minus 1
    num_samples_m1: NonZeroI64
}

impl RatioIter{
    pub fn float_iter(&self) -> impl Iterator<Item=f64>
    {
        self.ratio_iter()
            .map(|r| r.to_f64().unwrap())
    }

    pub fn ratio_iter(&self) -> impl Iterator<Item=Rational64>
    {
        let delta = (self.end - self.start) / self.num_samples_m1.get();
        let start = self.start;
        (0..=self.num_samples_m1.get())
            .map(
                move |i| 
                {
                    start + delta * i
                }
            )
    }

    pub fn from_float(min: f64, max: f64, samples: NonZeroI64) -> Self{
        let num_samples_m1 = NonZeroI64::new(samples.get() - 1).unwrap();
        let start = Rational64::approximate_float(min).unwrap();
        let end = Rational64::approximate_float(max).unwrap();
        Self { start, end, num_samples_m1 }
    }
}