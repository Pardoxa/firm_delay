use std::ops::*;

use crate::create_buf_with_command_and_version;
use crate::create_buf_with_command_and_version_and_header;
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

    let density_I = DensityI::calculate_above_leaf(&bins, &k_density);

    density_I.write(&bins, "I_density_test.dat");
    let integral = density_I.integral(&bins);
    println!("I_integral = {integral}");
    density_I.calc_crit(&bins);
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bins {
    bins_f64: Vec<f64>,
    bins_ratio: Vec<Ratio<isize>>,
    positive_bin_count: usize,
    bin_size: f64,
    s_approx: f64,
    s_idx_inclusive: usize
}

impl Bins{
    pub fn new(
        bin_count: usize,
        s: f64
    ) -> Self
    {
        let bin_size = Ratio::new(1, bin_count);
        let s_ratio = Ratio::approximate_float_unsigned(s).unwrap();
        let s_index = s_ratio / bin_size;
        let s_idx_ratio = s_index.ceil();
        let s_idx = s_idx_ratio.to_integer();
        let s_usable = s_idx_ratio * bin_size;
        let s_approx: f64 = s_usable.to_f64().unwrap();
        let bin_size_approx = bin_size.to_f64().unwrap();
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
            bin_size: bin_size_approx,
            s_approx,
            s_idx_inclusive: s_idx
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

    #[inline]
    pub fn get_left_I_slice(&self) -> &[f64]
    {
        let positive = self.get_positive_bin_borders_f64();
        &positive[..=self.s_idx_inclusive]
        
    }

    #[inline]
    pub fn get_right_I_slice(&self) -> &[f64]
    {
        let positive = self.get_positive_bin_borders_f64();
        &positive[self.s_idx_inclusive..]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DensityK {
    bin_borders: Vec<f64>,
    delta_left: f64,
    delta_right: f64
}

impl DensityK{
    pub fn new(s_idx_inclusive: usize, s: f64) -> Self
    {
        // since index counting starts at 0, the number of indices can be calculated by adding 1
        let num_bins = s_idx_inclusive + 1;
        const DELTA_HEIGHT: f64 = 0.2;
        const REST: f64 = 0.6;

        let height_rest = REST / s;
        let bin_borders = vec![height_rest; num_bins];

        Self { bin_borders, delta_left: DELTA_HEIGHT, delta_right: DELTA_HEIGHT }
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

    pub fn write(&self, bins: &Bins, counter: u32, s: f64)
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
    let bins = Bins::new(
        bin_count,
        s
    );

    let mut k_guess = DensityK::new(bins.s_idx_inclusive, bins.s_approx);
    let mut k_result = k_guess.new_zeroed();

    let bins_positive = bins.get_positive_bin_borders_f64();
    let mut counter: u32 = 0;

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
            k_result.delta_right += delta_right_b_update(L, R, interpolation.b, bins.s_approx)
                + delta_right_a_update(L, R, interpolation.a, bins.s_approx);
        }

        
        // delta left effect on bin_borders
        for (k_val_result, &z) in k_result.bin_borders.iter_mut().zip(bins_positive){
            *k_val_result += k_guess.delta_left * (1.0 - z); 
        }
        // delta left effect on delta left
        k_result.delta_left += k_guess.delta_left * 0.5;
        // delta left effect on delta right
        let sm1 = 1.0 - bins.s_approx;
        k_result.delta_right += k_guess.delta_left * sm1 * sm1 * 0.5;

        // delta right effect on delta right
        k_result.delta_right += k_guess.delta_right * 0.5;

        // delta right effect on bin borders
        for (k_val_result, &z) in k_result.bin_borders.iter_mut().zip(bins_positive){
            *k_val_result += k_guess.delta_right * (1.0 + z - bins.s_approx);
        }

        // delta right effect on delta left
        k_result.delta_left += k_guess.delta_right * 0.5 * sm1 * sm1;
        
        counter += 1;
        k_result.normalize(bins.bin_size);
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

#[derive(Clone, Serialize, Deserialize)]
pub struct DensityI{
    left_borders: Vec<f64>,
    right_borders: Vec<f64>
}

impl DensityI{
    pub fn new(bins: &Bins) -> Self
    {
        let slice_left = bins.get_left_I_slice();
        let slice_right = bins.get_right_I_slice();
        Self{
            left_borders: vec![0.0; slice_left.len()],
            right_borders: vec![0.0; slice_right.len()]
        }
    }

    pub fn write(&self, bins: &Bins, name: &str)
    {
        let header = [
            "I",
            "P(I)"
        ];
        let mut buf = create_buf_with_command_and_version_and_header(
            name,
            header
        );
        // left borders
        let iter_left = bins.get_positive_bin_borders_f64()
            .iter()
            .zip(self.left_borders.iter());
        // right borders
        let iter_right = bins.get_right_I_slice()
            .iter()
            .zip(self.right_borders.iter());
        // write both to file
        for (x, val) in iter_left.chain(iter_right) {
            writeln!(
                buf,
                "{x} {val}"
            ).unwrap()
        }
        
    }

    /// Calculations can be found in https://nxt.yfeld.de/s/StAyW6HkBcHqKJJ
    pub fn calculate_above_leaf(bins: &Bins, this_k: &DensityK) -> Self
    {
        // for later usage
        let mut this = Self::new(bins);
        let positive_bins = bins.get_positive_bin_borders_f64();
        // to simplify calculations while ignoring delta_right
        let mut I_bin_borders = vec![0.0; positive_bins.len()];

        let k_interpolation_iter = bins.interpolate_k(&this_k.bin_borders);

        for (interpolation, (L, R)) in k_interpolation_iter
        {
            let L_minus_R = L - R;
            let R_minus_L = R - L;
            let L_minus_R_divided_by_6 = L_minus_R / 6.0;
            let L_minus_R_squared_divided_by_6 = L_minus_R * L_minus_R_divided_by_6; // (L-R)^2 / 6
            let L_times_2 = L * 2.0;
            let L_times_2_plus_R = R + L_times_2; // 2 L + R
            let R_squared = R * R;
            let L_squared = L * L;
            let a = interpolation.a;
            let b = interpolation.b;
            let b_times_3 = 3.0 * b;
            let b_times_2 = 2.0 * b;

            let diff1A = L_minus_R_squared_divided_by_6 * (a * L_times_2_plus_R + b_times_3); // (L-R)^2*(a*(2L+R)+3b) / 6
            let part_of_B = (a * (R + L) + b_times_2) * R_minus_L / 2.0;
            let diff1B = (L_minus_R + 1.0) * part_of_B;

            let partC = L_minus_R * L_minus_R_divided_by_6 
                * (a*(L +2.0*R) + b_times_3);
            for (&x, I_of_x) in positive_bins.iter().zip(I_bin_borders.iter_mut())
            {
                if x <= L {
                    *I_of_x += diff1A;
                } else if x <= R {
                    // Note: In the current state this can be optimized quite a bit.
                    // If nothing changes we will only reach this if x == R
                    // and in that case ALMOST all of the below becomes 0,
                    // so we don't really need it.
                    // 
                    // I chose to keep it anyways, as the code is quite fast now 
                    // and this part will never become the bottleneck
                    // so I keep it
                    let one_minus_x = 1.0 - x;
                    let R_minus_x = R - x;
                    let part_that_appears_twice = b_times_3 * (R + x - L_times_2) 
                    + a * (-3.0 * L_squared + R_squared + R * x  + x * x);
                   
                    let offset = (
                        - one_minus_x * R_minus_x * (b_times_3 + a * (R + 2.0 * x))
                        + one_minus_x * part_that_appears_twice
                        + R_minus_x * part_that_appears_twice
                    ) / 6.0;
                    *I_of_x += offset;
                }
                // Yes, this is no else here! This needs to be executed regardless of the if statement above
                if x <= R {
                    *I_of_x += diff1B;
                } else if x <= L + 1.0 {
                    *I_of_x += (L - 2.0 * x + 2.0) * part_of_B;
                }

                // We don't need the if below (commented out, since the else is not reachable!)
                *I_of_x += partC;
                /* 
                    if x <= L + 1.0 {
                        *I_of_x += partC;
                    } else {
                        unreachable!("I think this part is unreachable. Otherwise I do have the equation that goes here");
                    }
                */
            }
        }

        // Now the part of the left Delta distribution
        let factor = 2.0 * this_k.delta_left;
        for (&x, I_of_x) in positive_bins.iter().zip(I_bin_borders.iter_mut())
        {
            if x <= 1.0 { // optimizable
                *I_of_x += factor * (1.0 - x);
            }
        }

        // now copy everything into result type:
        let left = &I_bin_borders[..this.left_borders.len()];
        this.left_borders.copy_from_slice(left);
        let right = &I_bin_borders[left.len()-1..];
        this.right_borders.copy_from_slice(right);

        // lastly I still need to add the part of the right delta

        this.left_borders.iter_mut()
            .for_each(
                |v| 
                *v += this_k.delta_right
            );
        let bins_for_right_delta = bins.get_right_I_slice();
        let offset = bins.s_approx + 2.0;
        for (&x, I_of_x) in bins_for_right_delta.iter().zip(this.right_borders.iter_mut())
        {
            // Note: x has to be >= s here, since we have thrown away the 
            // smaller bins
            *I_of_x += (offset - 2.0 * x) * this_k.delta_right;
            
        }
        this.special_normalize(bins);
        this
    }

    pub fn integral(&self, bins: &Bins) -> f64
    {
        let bin_size = bins.bin_size;
        self.integral_left(bin_size)
            + self.integral_right(bin_size)
    }

    pub fn integral_left(&self, bin_size: f64) -> f64
    {
        integrate_triangle_const_binsize(&self.left_borders, bin_size)
    }

    pub fn integral_right(&self, bin_size: f64) -> f64
    {
        integrate_triangle_const_binsize(&self.right_borders, bin_size)
    }

    /// Normalize by only adjusting the left bin_borders.
    /// This is used for the node above the leaf, since we know that the error
    /// for the right is way smaller than the error for the left
    /// And we need the integral to be 1 in total
    pub fn special_normalize(&mut self, bins: &Bins)
    {
        let integral_left = self.integral_left(bins.bin_size);
        let integral_right = self.integral_right(bins.bin_size);

        let target = 1.0 - integral_right;
        let factor = target / integral_left;
        self.left_borders.iter_mut()
            .for_each(
                |v| *v *= factor
            )
    }

    #[allow(dead_code)]
    pub fn normalize(&mut self, bins: &Bins)
    {
        let integral = self.integral(bins);
        let factor = integral.recip();
        let multiply = |slice: &mut [f64]| 
        {
            slice.iter_mut()
                .for_each(
                    |v| *v *= factor
                );
        };
        multiply(&mut self.left_borders);
        multiply(&mut self.right_borders);
    }

    pub fn calc_crit(&self, bins: &Bins)
    {
        let left_times_x = self.left_borders
            .iter()
            .zip(bins.get_positive_bin_borders_f64())
            .map(|(a,b)| a*b)
            .collect_vec();
        let left = integrate_triangle_const_binsize(&left_times_x, bins.bin_size);
        let right_times_x = self.right_borders
            .iter()
            .zip(bins.get_right_I_slice())
            .map(|(a,b)| a*b)
            .collect_vec();
        let right = integrate_triangle_const_binsize(&right_times_x, bins.bin_size);
        let sum = left + right;
        println!(
            "{left} {right} {sum}"
        );
    }
}

fn integrate_triangle_const_binsize(slice: &[f64], bin_size: f64) -> f64
{
    let sum: f64 = slice
            .windows(2)
            .map(|slice| slice[0] + slice[1])
            .sum();
    sum * 0.5 * bin_size
}