use std::ops::*;

use crate::create_buf_with_command_and_version;
use std::io::Write;
use super::numeric_integration::*;
use itertools::*;
use fraction::Ratio;
use fraction::ToPrimitive;
use serde::Deserialize;
use serde::Serialize;


fn h1(z: f64, L: f64, R: f64) -> f64
{
    0.5*(L-R)*(L+R-2.0*(z+1.0))
}

fn h2(z: f64, L: f64, R: f64) -> f64
{
    0.5*(R-L)*(L+R-2.0*z+2.0)
}

fn h3(z: f64, L: f64, R: f64) -> f64
{
    h1(z, z, R) + h2(z, L, z)
}

fn H1(z: f64, L: f64, R: f64) -> f64
{
    let L2 = L*L;
    (2.0*L2*L-3.0*L2*(z+1.0)+R*R*(3.0+3.0*z-2.0*R))/6.0
}

fn H2(z: f64, L: f64, R: f64) -> f64
{
    let L2 = L*L;
    (-2.0*L2*L+3.0*L2*(z-1.0)+R*R*(3.0-3.0*z+2.0*R))/6.0
}

fn H3(z: f64, L: f64, R: f64) -> f64
{
    H1(z, z, R) + H2(z, L, z)
}

fn delta_left_b_update(L: f64, R: f64, b: f64) -> f64
{
    let rml = R-L;
    b*(
        0.5*(L-1.0)*(R-1.0)*rml
        + rml*rml*rml/6.0
    )
}

fn delta_left_a_update(L: f64, R: f64, a: f64) -> f64
{
    let l2 = L*L;
    let lmr = L-R;
    a*(
        (
            (R-1.0)
            *(-4.0*l2*L+3.0*l2*(R+1.0)+(R-3.0)*R*R)
        ) /12.0
        - lmr * lmr * lmr * (3.0 * L + R)
          / 24.0
    )
}

fn delta_right_b_update(L: f64, R: f64, b: f64, s: f64) -> f64
{
    let rml = R-L;
    b*(
        0.5*rml*(L-s+1.0)*(R-s+1.0)
        + rml * rml * rml / 6.0
    )
}

fn delta_right_a_update(L: f64, R: f64, a: f64, s: f64) -> f64
{
    let rml = R-L;
    a * (
        rml
            * (L - s + 1.0)
            * (
                L * L
                + L * (R - 3.0 * s + 3.0)
                + R * (4.0 * R - 3.0 * s + 3.0)
            ) 
            / 12.0
        + (rml * rml * rml) 
            * (L + 3.0 * R)
            / 24.0
    ) 
}

struct LinearInterpolation{
    a: f64,
    b: f64
}

impl LinearInterpolation{
    #[allow(dead_code)]
    pub fn eval(&self, x: f64) -> f64
    {
        x.mul_add(self.a, self.b)
    }
}

#[allow(non_snake_case)]
pub fn compute_line(input: ModelInput)
{

    println!("Triangle");
    let (
        bins,
        k_density
    ) = k_of_leaf_parent(input.s, 1e-8, input.precision.get());

}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bins {
    bins_f64: Vec<f64>,
    bins_ratio: Vec<Ratio<isize>>,
    positive_bin_count: usize,
    bin_size: f64,
    s_approx: f64
}

impl Bins{
    pub fn new(bin_count: usize, s_approx: f64, bin_size: f64) -> Self
    {
        let bin_count_isize = bin_count as isize;
        let bins = (0..=bin_count*2)
            .map(
                |i|
                {
                    Ratio::new(i as isize - bin_count_isize, bin_count_isize)
                }
            ).collect_vec();
        let bins_f64 = bins
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect_vec();
        Self{
            bins_f64,
            bins_ratio: bins,
            positive_bin_count: bin_count,
            bin_size,
            s_approx
        }
    }

    #[allow(dead_code)]
    pub fn get_all_bin_borders_f64(&self) -> &[f64]
    {
        &self.bins_f64
    }

    pub fn get_positive_bin_borders_f64(&self) -> &[f64]
    {
        &self.bins_f64[self.positive_bin_idx_range()]
    }

    // excluding 0
    #[allow(dead_code)]
    pub fn get_negative_bin_borders_f64(&self) -> &[f64]
    {
        &self.bins_f64[self.negative_bin_idx_range()]
    }

    #[inline]
    fn positive_bin_idx_range(&self) -> RangeFrom<usize>
    {
        self.positive_bin_count..
    }

    #[inline]
    #[allow(dead_code)]
    fn negative_bin_idx_range(&self) -> RangeTo<usize>
    {
        ..self.positive_bin_count
    }

    // k has to include bin border for s
    fn interpolate_k<'a>(&'a self, k: &'a [f64]) -> impl Iterator<Item = (LinearInterpolation, (f64, f64))> + 'a
    {
        let bin_slice = self.get_positive_bin_borders_f64();
        bin_slice.windows(2)
            .zip(k.windows(2))
            .map(
                |(x, y)|
                {
                    let x_diff = x[1]-x[0];
                    let a = (y[1]-y[0])/x_diff;
                    let b = (y[0]*x[1]-y[1]*x[0])/x_diff;
                    let inter = LinearInterpolation{
                        a,
                        b
                    };
                    (
                        inter,
                        (x[0], x[1])
                    )
                }
            )

    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DensityK {
    bin_borders: Vec<f64>,
    delta_left: f64,
    delta_right: f64
}

impl DensityK{
    pub fn new(num_borders: usize, s: f64) -> Self
    {
        let delta_left = 0.2;
        let delta_right = 0.2;

        let height_rest = 0.6 / s;
        let bin_borders = vec![height_rest; num_borders];

        Self { bin_borders, delta_left, delta_right }
    }

    pub fn new_zeroed(&self) -> Self
    { 
        Self{
            delta_left: 0.0,
            delta_right: 0.0,
            bin_borders: vec![0.0; self.bin_borders.len()]
        }
    }
    pub fn make_zeroed(&mut self)
    {
        self.delta_left = 0.0;
        self.delta_right = 0.0;
        self.bin_borders.iter_mut()
            .for_each(|v| *v = 0.0);
    }

    pub fn write(&self, bins: &Bins, counter: u16, s: f64)
    {
        let name = format!("{counter}.dat");
        let mut buf = create_buf_with_command_and_version(name);
        let positive_bins = bins.get_positive_bin_borders_f64();

        for (val, bin) in self.bin_borders.iter().zip(positive_bins)
        {
            writeln!(
                buf,
                "{bin} {val}"
            ).unwrap();
        }

        let name = format!("{counter}_delta.dat");
        let mut buf = create_buf_with_command_and_version(name);
        writeln!(
            buf,
            "0 {}\n{s} {}",
            self.delta_left,
            self.delta_right
        ).unwrap();
    }

    pub fn integral(&self, bin_size: f64) -> f64
    {
        let mut sum = 0.0;
        // optimizable, but who cares
        for slice in self.bin_borders.windows(2)
        {
            let av = slice[0] + slice[1];
            sum += av;
        }
        sum /= 2.0;
        sum *= bin_size;
        sum + self.delta_left + self.delta_right
    }

    pub fn normalize(&mut self, bin_size: f64)
    {
        let integral = self.integral(bin_size);
        let factor = integral.recip();
        self.delta_left *= factor;
        self.delta_right *= factor;
        self.bin_borders
            .iter_mut()
            .for_each(
                |v| *v *= factor
            );
    }

    pub fn abs_diff(&self, other: &Self) -> f64
    {
        let mut diff = (self.delta_left - other.delta_left).abs()
            + (self.delta_right - other.delta_right).abs();
        self.bin_borders.iter()
            .zip(other.bin_borders.iter())
            .for_each(
                |(a,b)| 
                diff += (a-b).abs()
            );
        diff
    }
}

fn k_of_leaf_parent(
    s: f64, 
    threshold: f64,
    bin_count: usize
)-> (Bins, DensityK)
{
    let bin_size = Ratio::new(1, bin_count);
    let s_ratio = Ratio::approximate_float_unsigned(s).unwrap();
    let s_index = s_ratio / bin_size;
    let s_idx_ratio = s_index.ceil();
    let s_idx = s_idx_ratio.to_integer();
    let s_usable = s_idx_ratio * bin_size;
    let s_approx: f64 = s_usable.to_f64().unwrap();
    let bin_size_approx = bin_size.to_f64().unwrap();

    let mut k_guess = DensityK::new(s_idx + 1, s_approx);
    let mut k_result = k_guess.new_zeroed();
    
    let bins = Bins::new(
        bin_count,
        s_approx,
        bin_size_approx
    );
    k_guess.write(&bins, 100, s_approx);

    let bins_positive = bins.get_positive_bin_borders_f64();
    let mut counter = 0;

    loop{
        
        let k_interpolation_iter = bins.interpolate_k(&k_guess.bin_borders);

        // L is left border of k bin, R is right border of k bin
        for (interpolation, (L, R)) in k_interpolation_iter
        {
            // use bin_borders of guess to update bin_borders of result
            for (k_val_result, &z) in k_result.bin_borders.iter_mut().zip(bins_positive){
                *k_val_result += if z -L <= 0.0{
                    interpolation.a * H1(z, L, R) + interpolation.b * h1(z, L, R)
                } else if z-R <= 0.0 {
                    interpolation.a * H3(z, L, R) + interpolation.b * h3(z, L, R)
                } else {
                    interpolation.a * H2(z, L, R) + interpolation.b * h2(z, L, R)
                };
            }

            // use bin_borders to update delta left of result
            k_result.delta_left += delta_left_b_update(L, R, interpolation.b)
                + delta_left_a_update(L, R, interpolation.a); // the a part of the deltas is not symmetric yet!
            
            // use bin_borders to update delta right of result
            k_result.delta_right += delta_right_b_update(L, R, interpolation.b, s_approx)
                + delta_right_a_update(L, R, interpolation.a, s_approx);
        }

        
        // delta left effect on bin_borders
        for (k_val_result, &z) in k_result.bin_borders.iter_mut().zip(bins_positive){
            *k_val_result += k_guess.delta_left * (1.0 - z); 
        }
        // delta left effect on delta left
        k_result.delta_left += k_guess.delta_left * 0.5;
        // delta left effect on delta right
        let sm1 = 1.0 - s_approx;
        k_result.delta_right += k_guess.delta_left * sm1 * sm1 * 0.5;

        // delta right effect on delta right
        k_result.delta_right += k_guess.delta_right * 0.5;

        // delta right effect on bin borders
        for (k_val_result, &z) in k_result.bin_borders.iter_mut().zip(bins_positive){
            *k_val_result += k_guess.delta_right * (1.0 + z - s_approx);
        }

        // delta right effect on delta left
        k_result.delta_left += k_guess.delta_right * 0.5 * sm1 * sm1;
        
        counter += 1;
        k_result.normalize(bin_size_approx);
        k_result.write(&bins, counter, s);
        if counter >= 20 {
            let diff = k_guess.abs_diff(&k_result);
            if diff <= threshold{
                return (
                    bins,
                    k_result
                )
            }
        }

        std::mem::swap(&mut k_guess, &mut k_result);
        k_result.make_zeroed();
    }
    
}

#[allow(dead_code)]
fn dreieck_integrations_helfer(slice: &[f64]) -> Vec<f64>
{
    let mut result = Vec::with_capacity(slice.len());
    result.extend(
        slice.windows(2)
            .map(
                |w|
                {
                    0.5 * (w[0] + w[1])
                }
            )
        );
    let last = &slice[slice.len()-2..];
    debug_assert_eq!(last.len(), 2);
    // extrapolate last value
    //let extrapolated_value = 2.0*(last[1]-last[0])+last[0];
    //let triangle_val = (extrapolated_value + last[1])*0.5;
    // val is mathematically equal to triangle_val
    let val = (3.0 * last[1]-last[0])*0.5;
    result.push(
        val
    );
    result
}