use crate::create_buf_with_command_and_version;
use crate::create_buf_with_command_and_version_and_header;
use std::io::Write;
use super::numeric_integration::*;
use itertools::*;
use fraction::Ratio;
use fraction::ToPrimitive;
use serde::Deserialize;
use serde::Serialize;
use super::array_windows::*;

#[allow(non_snake_case)]
pub fn compute_line(input: ModelInput)
{

    println!("Triangle");
    let (
        bins,
        k_density
    ) = k_of_leaf_parent(input.s, 1e-9, input.precision.get());
    let density_I = DensityI::calculate_above_leaf(&bins, &k_density);

    density_I.write(&bins, "I_density_test.dat");
    let integral = density_I.integral(&bins);
    println!("I_integral = {integral}");
    density_I.calc_crit(&bins);
    let density_lambda = DensityLambda::calculate_above_leaf(&bins, &k_density);
    density_lambda.write(&bins, "lambda.dat");
    let lambda_integral = density_lambda.integral(&bins);
    println!("Lambda_integral: {lambda_integral}");
    let mut deltas = Delta_kij_of_Ii_intervals::new(&density_lambda, &bins);
    //deltas.delta_error_correction(k_density.delta_left, k_density.delta_right);
    deltas.write_deltas(&bins, "test_d.dat");

    let test_left_iter =
        deltas.delta_left.iter()
        .zip(deltas.delta_right.iter());
    let mut sum_left = 0.0;
    let mut sum_right = 0.0;
    for (delta_left, delta_right) in test_left_iter{
        sum_left += delta_left;
        sum_right += delta_right;
    }
    dbg!(sum_left - k_density.delta_left);
    dbg!(sum_right - k_density.delta_right);
    dbg!(
        sum_left - sum_right
    );
    println!("{}", k_density.delta_left);
    println!("{sum_left}");
    println!("{sum_right}");

    let test = TestKij::new(&density_lambda, &bins);
    test.check(&bins, &deltas);
    test.write_sum(&bins, "k_test_sum.dat");

    let mut Ii_given_pre_Ii_interval = Ii_given_pre_Ii_interval::calc_above_leaf(
        &deltas, 
        &density_lambda,
        &bins,
        &test
    );

    Ii_given_pre_Ii_interval.check(&density_I, &bins);
    Ii_given_pre_Ii_interval.write_sum_all(&bins, "current_test.dat");

}



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

#[derive(Debug)]
pub struct LinearInterpolation{
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bins {
    bins_f64: Vec<f64>,
    bins_ratio: Vec<Ratio<isize>>,
    idx_of_0: usize,
    bin_size: f64,
    s_approx: f64,
    s_idx_inclusive_of_positive_slice: usize,
    s_bin_idx_total: usize,
    idx_of_one: usize
}

impl Bins{
    pub fn new(
        bin_count_until_one: usize,
        s: f64
    ) -> Self
    {
        let bin_size = Ratio::new(1, bin_count_until_one);
        let s_ratio = Ratio::approximate_float_unsigned(s).unwrap();
        let s_index = s_ratio / bin_size;
        let s_idx_ratio = s_index.ceil();
        let s_idx = s_idx_ratio.to_integer();
        let s_usable = s_idx_ratio * bin_size;
        let s_approx: f64 = s_usable.to_f64().unwrap();
        let bin_size_approx = bin_size.to_f64().unwrap();
        let bin_count_isize = bin_count_until_one as isize;
        let bins = (0..=bin_count_until_one*2+s_idx)
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
            idx_of_0: bin_count_until_one,
            bin_size: bin_size_approx,
            s_approx,
            s_idx_inclusive_of_positive_slice: s_idx,
            idx_of_one: bin_count_until_one*2,
            s_bin_idx_total: bin_count_until_one+s_idx
        }
    }

    #[allow(dead_code)]
    pub fn get_all_bin_borders_f64(&self) -> &[f64]
    {
        &self.bins_f64
    }

    // including 0
    pub fn get_positive_bin_borders_f64(&self) -> &[f64]
    {
        &self.bins_f64[self.idx_of_0..]
    }

    pub fn slice_starting_at_s(&self) -> &[f64]
    {
        &self.bins_f64[self.s_bin_idx_total..]
    }

    pub fn slice_starting_at_1(&self) -> &[f64]
    {
        &self.bins_f64[self.idx_of_one..]
    }

    // excluding 0
    #[allow(dead_code)]
    pub fn get_negative_bin_borders_f64(&self) -> &[f64]
    {
        &self.bins_f64[..self.idx_of_0]
    }

    // k has to include bin border for s
    fn interpolate_k<'a>(&'a self, k: &'a [f64]) -> impl Iterator<Item = (LinearInterpolation, (f64, f64))> + 'a
    {
        let bin_slice = self.get_positive_bin_borders_f64();
        linear_interpolation_iter(bin_slice, k)
    }

    #[inline]
    pub fn get_left_I_slice(&self) -> &[f64]
    {
        &self.bins_f64[self.idx_of_0..=self.s_bin_idx_total]
        
    }

    #[inline]
    pub fn get_right_I_slice(&self) -> &[f64]
    {
        &self.bins_f64[self.s_bin_idx_total..=self.idx_of_one]
    }

    // both 0 and 1 are included
    fn bins_in_range_0_to_1(&self) -> &[f64]
    {
        &self.bins_f64[self.idx_of_0..=self.idx_of_one]
    }

    fn get_lambda_bin_borders(&self) -> &[f64]
    {
        // For now it is the same as get positive bin_borders.
        // This might change, as I don't yet know if I need to add more positive
        // bin borders later on. For this reason I added this method.
        // So I only need to adjust it here once
        self.get_positive_bin_borders_f64()
    }

    fn offset_by_one(&self) -> usize
    {
        self.idx_of_0
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
        self.bin_borders.fill(0.0);
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
        let integral = integrate_triangle_const_binsize(&self.bin_borders, bin_size);
        integral + self.delta_left + self.delta_right
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

    let mut k_guess = DensityK::new(bins.s_idx_inclusive_of_positive_slice, bins.s_approx);
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
        let bins_range_0_to_1 = bins.bins_in_range_0_to_1();
        // to simplify calculations while ignoring delta_right
        let mut I_bin_borders = vec![0.0; bins_range_0_to_1.len()];

        let k_interpolation_iter = bins.interpolate_k(&this_k.bin_borders);

        for (LinearInterpolation { a, b }, (L, R)) in k_interpolation_iter
        {
            let L_minus_R = L - R;
            let R_minus_L = R - L;
            let L_minus_R_divided_by_6 = L_minus_R / 6.0;
            let L_minus_R_squared_divided_by_6 = L_minus_R * L_minus_R_divided_by_6; // (L-R)^2 / 6
            let L_times_2 = L * 2.0;
            let L_times_2_plus_R = R + L_times_2; // 2 L + R
            let R_squared = R * R;
            let L_squared = L * L;
            let b_times_3 = 3.0 * b;
            let b_times_2 = 2.0 * b;

            let diff1A = L_minus_R_squared_divided_by_6 * (a * L_times_2_plus_R + b_times_3); // (L-R)^2*(a*(2L+R)+3b) / 6
            let part_of_B = (a * (R + L) + b_times_2) * R_minus_L / 2.0;
            let diff1B = (L_minus_R + 1.0) * part_of_B;

            let partC = L_minus_R * L_minus_R_divided_by_6 
                * (a*(L +2.0*R) + b_times_3);
            for (&x, I_of_x) in bins_range_0_to_1.iter().zip(I_bin_borders.iter_mut())
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
        for (&x, I_of_x) in bins_range_0_to_1.iter().zip(I_bin_borders.iter_mut())
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
        let integration = |val_slice: &[f64], bin_slice: &[f64]|
        {
            let multiplied_slice = val_slice
                .iter()
                .zip(bin_slice)
                .map(|(a,b)| a*b)
                .collect_vec();
            integrate_triangle_const_binsize(&multiplied_slice, bins.bin_size)
        };
        let left = integration(
            &self.left_borders,
            bins.get_positive_bin_borders_f64()
        );
        let right = integration(
            &self.right_borders,
            bins.get_right_I_slice()
        );
        let sum = left + right;
        println!(
            "{left} {right} {sum}"
        );
    }
}

fn integrate_triangle_const_binsize(slice: &[f64], bin_size: f64) -> f64
{
    // optimizable, but who cares
    let sum: f64 = slice
            .windows(2)
            .map(|slice| slice[0] + slice[1])
            .sum();
    sum * 0.5 * bin_size
}

pub struct DensityLambda{
    left_border_vals: Vec<f64>,
    mid_border_vals: Vec<f64>,
    right_border_vals: Vec<f64>
}

impl DensityLambda{
    pub fn calculate_above_leaf(bins: &Bins, k_density: &DensityK) -> Self
    {
        let offset_by_one = bins.offset_by_one();
        let bin_borders = bins.get_lambda_bin_borders();
        let mut lambda_border_vals = vec![0.0; bin_borders.len()];

        let iter = bins.interpolate_k(&k_density.bin_borders);

        // result of k_function
        for (iteration, (LinearInterpolation { a, b }, (L, R))) in iter.enumerate() {
            

            let R_minus_L = R-L;
            let R_plus_L = R+L;

            // x in [L,R] 
            // ignore L, as the result is always 0.
            // In between L and R: I have no bins, so I can ignore it.
            // For x=R it is the same as in the range [R, L+1]
            // In that range the result does not depend on x. 
            // So I only calculate that
            let res = 0.5 * R_minus_L 
                * (a * R_plus_L + b * 2.0);
            let idx_of_R = iteration + 1;
            let idx_of_L_plus_one = iteration + offset_by_one;
            let lambda_range_R_to_L_plus_one = &mut lambda_border_vals[idx_of_R..=idx_of_L_plus_one];
            
            lambda_range_R_to_L_plus_one
                .iter_mut()
                .for_each(
                    |v| *v += res
                );
            // [L+1,R+1]
            // L+1 is already inside the above
            // R+1 results in 0
            // in between I have no bins
        }

        let s_idx = bins.s_idx_inclusive_of_positive_slice;
        let old_val_at_offset_by_s = lambda_border_vals[s_idx];

        // Now the right lambda
        let lambda_range_from_s_to_s_plus_1 = &mut lambda_border_vals[bins.s_idx_inclusive_of_positive_slice..];

        lambda_range_from_s_to_s_plus_1
            .iter_mut()
            .for_each(
                |v| *v += k_density.delta_right
            );

        let right = lambda_border_vals[offset_by_one..].to_vec();

        // the left delta distribution
        let lambda_range_from_0_to_1 = &mut lambda_border_vals[..=offset_by_one];
        lambda_range_from_0_to_1
            .iter_mut()
            .for_each(
                |v| *v += k_density.delta_left
            );
        
        let mid = lambda_border_vals[s_idx..=offset_by_one].to_vec();

        lambda_border_vals.truncate(s_idx+1);
        // Strictly speaking, the value exactly at s has the delta jump, but 
        // the values before have not
        //
        // For this reason I need to recalculate the last value, such that the integral from the 
        // left later gives the correct results.
        // This is also the reason why the bin border appears more than once
        *lambda_border_vals
            .last_mut()
            .unwrap() = old_val_at_offset_by_s + k_density.delta_left;
        let left = lambda_border_vals;


        
        let mut result = Self{ 
            left_border_vals: left,
            mid_border_vals: mid,
            right_border_vals: right,
        };
        result.normalize(bins);
        result
    }

    pub fn write(&self, bins: &Bins, name: &str)
    {
        let header = [
            "lambda",
            "P(lambda)"
        ];

        let mut buf = create_buf_with_command_and_version_and_header(
            name, 
            header
        );

        let mut write_helper = |bin_border_slice: &[f64], val_slice: &[f64]|
        {
            for (lambda, p_of_lambda) in bin_border_slice.iter().zip(val_slice)
            {
                writeln!(
                    buf,
                    "{lambda} {p_of_lambda}"
                ).unwrap()
            }
        };
        let bin_borders = bins.get_lambda_bin_borders();
        write_helper(bin_borders, &self.left_border_vals);
        let mid = &bin_borders[bins.s_idx_inclusive_of_positive_slice..];
        write_helper(mid, &self.mid_border_vals);
        let right = &bin_borders[bins.offset_by_one()..];
        write_helper(right, &self.right_border_vals);

    }

    pub fn integral(&self, bins: &Bins) -> f64
    {
        let bin_size = bins.bin_size;
        let left = integrate_triangle_const_binsize(&self.left_border_vals, bin_size);
        let mid = integrate_triangle_const_binsize(&self.mid_border_vals, bin_size);
        let right = integrate_triangle_const_binsize(&self.right_border_vals, bin_size);
        left + mid + right
    }

    pub fn normalize(&mut self, bins: &Bins)
    {
        let factor = self.integral(bins).recip();
        let multiply = |slice: &mut [f64]|
        {
            slice
                .iter_mut()
                .for_each(
                    |v| *v *= factor
                )
        };
        multiply(&mut self.left_border_vals);
        multiply(&mut self.mid_border_vals);
        multiply(&mut self.right_border_vals);
        
    }

    pub fn left_interpolation_iter<'a>(
        &'a self,
        bins: &'a Bins
    ) -> impl Iterator<Item = (LinearInterpolation, (f64, f64))> + 'a
    {
        let bin_slice = bins.get_positive_bin_borders_f64();
        linear_interpolation_iter(
            bin_slice,
            &self.left_border_vals
        )
    }

    pub fn mid_interpolation_iter<'a>(
        &'a self,
        bins: &'a Bins
    ) -> impl Iterator<Item = (LinearInterpolation, (f64, f64))> + 'a
    {
        let bin_slice = bins.slice_starting_at_s();
        linear_interpolation_iter(
            bin_slice, 
            &self.mid_border_vals
        )
    }

    pub fn right_interpolation_iter<'a>(
        &'a self,
        bins: &'a Bins
    ) -> impl Iterator<Item = (LinearInterpolation, (f64, f64))> + 'a
    {
        let bin_slice = bins.slice_starting_at_1();
        linear_interpolation_iter(
            bin_slice, 
            &self.right_border_vals
        )
    }

    pub fn get_all_interpolations(
        &self,
        bins: &Bins
    ) -> Vec<LinearInterpolation>
    {
        let mut interpolation = self
            .left_interpolation_iter(bins)
            .map(|(interpolation, _)| interpolation)
            .collect_vec();
        interpolation.extend(
            self
                .mid_interpolation_iter(bins)  
                .map(|(interpolation, _)| interpolation)
        );
        interpolation.extend(
            self
              .right_interpolation_iter(bins)
              .map(|(interpolation, _)| interpolation)
        );
        interpolation
    }
}

pub struct TestKij{
    pub func: Vec<Vec<f64>>
}

impl TestKij{

    pub fn check(&self, bins: &Bins, deltas: &Delta_kij_of_Ii_intervals)
    {
        let header = [
            "all",
            "only_integral"
        ];

        let mut buf = create_buf_with_command_and_version_and_header(
            "k_check.dat",
            header
        );
        for (slice, deltas) in self.func.iter().zip(deltas.delta_left.iter().zip(deltas.delta_right.iter()))
        {
            let integral: f64 = ArrayWindows::<_,2>::new(slice)
                .map(
                    |[a,b]| 0.5*(a+b) * bins.bin_size
                ).sum();
            let val = integral + deltas.0 + deltas.1;
            writeln!(
                buf,
                "{val} {integral}"
            ).unwrap();
        }
    }

    pub fn new(lambda_dist: &DensityLambda, bins: &Bins) -> Self {
        // y <= x

        let windows = ArrayWindows::<_,2>::new(bins.bins_in_range_0_to_1());

        let interpolation = lambda_dist.get_all_interpolations(bins);
        let relevant_bins = bins.get_left_I_slice();
        let func = windows
            .enumerate()
            .map(
                |(counter, [Ly, Ry])|
                {
                    let Ry_minus_Ly_div_2 = (Ry - Ly) * 0.5;
                    let Ly_plus_Ry = Ly + Ry;

                    let z = relevant_bins;
                    /// I think the error is here! I think the edge case is not correctly calculated!
                    let x_interpolation = &interpolation[counter..];

                    let iter = z.iter()
                        .zip(x_interpolation);

                    iter.map(
                        |(z, LinearInterpolation { a, b })|
                        {
                            Ry_minus_Ly_div_2 * 
                            (
                                b.mul_add(
                                    2.0, 
                                    a * z.mul_add(
                                        2.0, 
                                        Ly_plus_Ry
                                    )
                                )
                            )
                        }
                    ).collect_vec()

                }
            ).collect_vec();
        
        Self { func }
    }

    pub fn write_sum(&self, bins: &Bins, name: &str)
    {
        let mut sum = self.func[0].clone();
        for slice in self.func[1..].iter()
        {
            sum.iter_mut()
                .zip(slice)
                .for_each(
                    |(a, b)|
                    {
                        *a += b
                    }
                );
        }

        let mut writer = create_buf_with_command_and_version(name);
        
        let positive = bins.get_positive_bin_borders_f64();
        
        positive.iter()
            .zip(sum)
            .for_each(
                |(z, val)|
                writeln!(
                    writer,
                    "{z} {val}"
                ).unwrap()
            );
    }
}

// this is temporary, until I know how to store this better
#[allow(non_camel_case_types)]
pub struct Delta_kij_of_Ii_intervals
{
    delta_left: Vec<f64>,
    delta_right: Vec<f64>
}

impl Delta_kij_of_Ii_intervals{
    pub fn new(lambda_dist: &DensityLambda, bins: &Bins) -> Self
    {
        let I_range = bins.bins_in_range_0_to_1();
        // number of bins is number of bin_borders minus 1:
        let max_len_of_deltas = I_range.len() - 1;

        let delta_left_calc = |(LinearInterpolation { a, b }, (L, R))|
        {
            let R_times_2_minus_3 = 2.0 * R - 3.0;
            (L - R)
            *   (a * 
                    (
                        L * (2.0 * L + R_times_2_minus_3) + R * R_times_2_minus_3
                    )
                + 3.0 * b * (L + R - 2.0)
                ) / 6.0
        };
        let mut delta_left = lambda_dist
            .left_interpolation_iter(bins)
            .map(
                delta_left_calc
            ).take(max_len_of_deltas)
            .collect_vec();
        if delta_left.len() < max_len_of_deltas{
            let mut remaining = max_len_of_deltas - delta_left.len();
            delta_left.extend(
                lambda_dist.mid_interpolation_iter(bins)
                    .map(delta_left_calc)
                    .take(remaining)
            );
            if delta_left.len() < max_len_of_deltas{
                remaining = max_len_of_deltas - delta_left.len();
                delta_left.extend(
                    lambda_dist.right_interpolation_iter(bins)
                        .map(delta_left_calc)
                        .take(remaining)   
                );
            }
        }
        assert_eq!(
            delta_left.len(),
            max_len_of_deltas
        );

        
        // s+y<=Lx 
        // y is at least 0
        // but the interval [s=Lx, Rx] should be treaded with the other method.
        // so, I should calculate from s+(one interval) to the end
        //
        // the lambda left interval goes from 0 to s
        // so we need to start at the mid interval while skipping the first
        let mid_iter = lambda_dist
            .mid_interpolation_iter(bins)
            .skip(1);
        // Ry-Ly is bin_size and the same as Rx-Lx
        let bin_size_squared_div_2 = bins.bin_size * bins.bin_size * 0.5;
        let mut help_vec = Vec::with_capacity(
            lambda_dist.mid_border_vals.len()
            + lambda_dist.right_border_vals.len()
        );

        let help_calc = |(LinearInterpolation { a, b }, (Lx, Rx))|
        {
            (a * (Lx + Rx)
                + 2.0 * b
            ) * bin_size_squared_div_2
        };
        help_vec.push(0.0);
        help_vec.extend(
            mid_iter
            .map(
                help_calc
            )  
        );
        help_vec.extend(
            lambda_dist.right_interpolation_iter(bins)
                .map(help_calc)
        );
        // Insert picture
        let mut running_sum = 0.0;
        help_vec.iter_mut()
            .rev()
            .for_each(
                |v|
                {
                    let tmp = running_sum;
                    running_sum += *v;
                    *v = tmp;
                }
            );
        let lambda_bin_slice = bins.slice_starting_at_s();
        let s = bins.s_approx;

        let lambda_val_iter = ArrayWindows::<_, 2>::new(&lambda_dist.mid_border_vals)
            .chain(
                ArrayWindows::<_, 2>::new(&lambda_dist.right_border_vals)
            );

        let mut delta_right = Vec::with_capacity(delta_left.len());
        delta_right.extend(
            ArrayWindows::<_, 2>::new(lambda_bin_slice)
                .zip(help_vec)
                .zip(ArrayWindows::<_,2>::new(I_range))
                .zip(lambda_val_iter)
                .map(
                    |((([Lx, Rx], sum), [Ly, Ry]), [lambda_left, lambda_right])|
                    {
                        let LinearInterpolation{a, b} = calculate_interpolation(
                            *Lx,
                            *Rx,
                            *lambda_left,
                            *lambda_right
                        );
                        let Ly_plus_Ry = Ly + Ry;
                        (Ly - Ry)
                        * (
                            a.mul_add(
                                3.0 * (s * (Ry + s + Ly) - Rx * Rx)
                                    + Ly * Ly_plus_Ry
                                    + Ry * Ry,
                                3.0 * b * (Ly_plus_Ry + 2.0 * (s - Rx))
                            ) 
                        )
                        / 6.0
                        + sum
                    }
                )   
        );
        dbg!(
            delta_left.len()
        );
        dbg!(
            delta_right.len()
        );

        Self { delta_left, delta_right }
    }

    fn write_deltas(&self, bins: &Bins, name: &str)
    {
        let header = [
            "middle_of_I_bin",
            "delta_left",
            "delta_right"
        ];
        let mut writer = create_buf_with_command_and_version_and_header(
            name, 
            header
        );

        let iter = bins.get_positive_bin_borders_f64()
            .windows(2)
            .zip(self.delta_left.iter())
            .zip(self.delta_right.iter());
        for ((bin, left), right) in iter {
            let mid = (bin[0] + bin[1]) * 0.5;
            writeln!(
                writer,
                "{mid} {left} {right}"
            ).unwrap();
        }
    }

    pub fn delta_error_correction(&mut self, delta_left: f64, delta_right: f64)
    {
        let left_sum: f64 = self.delta_left.iter().sum();
        let factor = delta_left / left_sum;
        self.delta_left
            .iter_mut()
            .for_each(
                |v| *v *= factor
            );
        let right_sum: f64 = self.delta_right.iter().sum();
        let factor = delta_right / right_sum;
        self.delta_right
            .iter_mut()
            .for_each(
                |v| *v *= factor
            );
    }
}


fn linear_interpolation_iter<'a>(
    bin_slice: &'a [f64], 
    bin_border_vals: &'a [f64]
) -> impl Iterator<Item = (LinearInterpolation, (f64, f64))> + 'a
{
    ArrayWindows::<_,2>::new(bin_slice)
        .zip(ArrayWindows::<_,2>::new(bin_border_vals))
        .map(
            |(x, y)|
            {
                let inter = calculate_interpolation(
                    x[0], 
                    x[1], 
                    y[0],
                    y[1]
                );
                (
                    inter,
                    (x[0], x[1])
                )
            }
        )
}

#[inline]
pub fn calculate_interpolation(
    bin_left: f64,
    bin_right: f64,
    val_left: f64,
    val_right: f64
) -> LinearInterpolation
{
    let bin_diff = bin_right - bin_left;
    let a = (val_right - val_left) / bin_diff;
    let b = (val_left * bin_right - val_right * bin_left) / bin_diff;
    LinearInterpolation{
        a,
        b
    }
}

#[allow(non_camel_case_types)]
pub struct Ii_given_pre_Ii_interval
{
    matrix: Vec<DensityI>
}

impl Ii_given_pre_Ii_interval{

    pub fn check(&mut self, other: &DensityI, bins: &Bins)
    {
        let mut probs = ArrayWindows::<_,2>::new(&other.left_borders)
            .map(
                |[L, R]|
                {
                    0.5*(L+R) * bins.bin_size
                }
            ).collect_vec();

        probs.extend(
            ArrayWindows::<_,2>::new(&other.right_borders)
                .map(
                    |[L, R]|
                    {
                        0.5*(L+R) * bins.bin_size
                    }
                )
        );

        let header = [
            "val-target",
            "val",
            "target"
        ];

        let mut buf = create_buf_with_command_and_version_and_header(
            "check.dat",
            header
        );



        for (i, (line, target)) in self.matrix.iter_mut().zip(probs).enumerate()
        {
            let val = line.integral(bins);
           
            
            writeln!(buf, "{} {val} {target}", val - target).unwrap();
            let name = format!("db_{i}.dat");
            line.write(bins, &name);
        }
    }

    /// The relevant calculations can be found at: https://nxt.yfeld.de/s/7PAgnx6Lo9g5i4d
    /// The side calculation can be found at: https://nxt.yfeld.de/s/FtpWGRxnJoc8bdM
    pub fn calc_above_leaf(
        deltas_of_kij: &Delta_kij_of_Ii_intervals, 
        density_lambda: &DensityLambda,
        bins: &Bins,
        k: &TestKij
    ) -> Self
    {
        let s = bins.s_approx;
        let bins_range_0_to_1 = bins.bins_in_range_0_to_1();
        // both intervals include s here!
        let until_s = &bins_range_0_to_1[..=bins.s_idx_inclusive_of_positive_slice];
        let from_s = &bins_range_0_to_1[bins.s_idx_inclusive_of_positive_slice..];

        let iter = deltas_of_kij
            .delta_left
            .iter()
            .zip(deltas_of_kij.delta_right.iter());


        
        let mut matrix = iter
            .map(
                |(&delta_left, &delta_right)|
                {
                    let mut left_result = Vec::with_capacity(until_s.len());
                    left_result.extend(
                        until_s.iter()
                            .map(
                                |x| 
                                x.mul_add(-2.0, 2.0)
                                    .mul_add(delta_left, delta_right)
                                // the above should be equivalent to:
                                // delta_right + delta_left * (2.0 - 2.0 * x)
                            )
                    );
                    let delta_right_times_s = s * delta_right;
                    let delta_sum = delta_left + delta_right;
                    let mut right_result = Vec::with_capacity(from_s.len()); 
                    right_result.extend(
                        from_s.iter()
                            .map(
                                |x|
                                {
                                    x.mul_add(-2.0, 2.0)
                                        .mul_add(delta_sum, delta_right_times_s)
                                    // the above should be equivalent to:
                                    // delta_right_times_s + delta_sum * (2.0 - 2.0 * x)
                                }
                            )
                    );
                    DensityI{
                        left_borders: left_result,
                        right_borders: right_result
                    }
                }
            ).collect_vec();
         
        let y_windows = ArrayWindows::<_,2>::new(bins_range_0_to_1);

        let lambda_interpolations = density_lambda.get_all_interpolations(bins);
        let k_len = until_s.len();
        matrix.iter_mut()
            .zip(y_windows)
            .enumerate()
            .for_each(
                |(offset, (matrix_slice, [Ly, Ry]))|
                {
                    //matrix_slice.left_borders.fill(0.0);
                    //matrix_slice.right_borders.fill(0.0);
                    let Ly_minus_Ry = Ly - Ry;
                    let Ly_plus_Ry = Ry + Ly;
                    // check if this is the correct range!
                    let this_lambda_interpolations = &lambda_interpolations[offset..offset + k_len - 1];
                    let lambda_range = &bins.get_positive_bin_borders_f64()[offset..];
                    dbg!(lambda_range);
                    dbg!(this_lambda_interpolations);

                    let windows = ArrayWindows::<_,2>::new(until_s);
                    let iter = this_lambda_interpolations
                        .iter()
                        .zip(windows)
                        .enumerate();

                    println!("y: {Ly} {Ry}");

                    for (counter, (LinearInterpolation { a, b }, [F1, F2])) in iter {
                        // Range in which J1 is valid: [..=counter]
                        dbg!(counter);
                        dbg!(a);
                        dbg!(b);
                        let b_times_2 = b * 2.0;
                        let F1_minus_F2 = F1 - F2;
                        let F1_plus_F2 = F1 + F2;
                        // first calculate J1
                        let J1 = 0.5*F1_minus_F2*Ly_minus_Ry*(
                            b_times_2 + a*(F1_plus_F2+Ly_plus_Ry)
                        );

                        let counter_plus_1 = counter + 1;
                        
                        let len_left = matrix_slice.left_borders.len();
                        let split_idx = len_left.min(counter_plus_1);
                        let (left_until_A, left_rest) = matrix_slice.left_borders
                            .split_at_mut(split_idx);
                        let z_right_slice_dbg = &until_s[..left_until_A.len()];
                        let z_left_slice = &until_s[left_until_A.len()..];
                        dbg!(z_right_slice_dbg);
                        dbg!(z_left_slice);
                        dbg!(J1);

                        // I use it for testing now
                        let calc_test = |x: f64|
                        {
                            -1.0/12.0 * (Ly-Ry)*(a*(-4.0*F1*F1*F1+6.0*F2*(F2+Ly+Ry)-3.0*F1*F1*(
                                4.0+Ly+Ry-4.0*x)
                            +12.0*F1*(Ly+Ry)*(x-1.0)
                            +6.0 * (Ly+Ry)*x-3.0*(3.0*(Ly+Ry)-2.0)*x*x-8.0*x*x*x)
                            -6.0*b*(F1*(4.0+F1)-4.0*F1*x+3.0*x*x-2.0*(F2+x)))
                        };
                        let z_first = until_s[left_until_A.len().saturating_sub(1)];
                        let t = calc_test(z_first);
                        dbg!(t);
                        let t2 = calc_test(*z_left_slice.get(0).unwrap_or(&0.0));
                        dbg!(t2);

                        left_until_A.iter_mut()
                            .for_each(
                                |v| *v += J1
                            );

                       
                        // I think I can get away with skipping J2:
                        // It is only used to calculate two values, on the left this value is equal to J1
                        // On the right it is equal to J3.

                        


                        let twelve_recip = 12.0_f64.recip();
                        let b_times_6 = 6.0*b;
                        let outer = twelve_recip*F1_minus_F2*Ly_minus_Ry;
                        let inner_summand = -( 
                            F1_minus_F2*(
                                b_times_6+a*(2.0*F1+4.0*F2+3.0*F1_plus_F2)
                            )
                        );
                        let inner_factor = 6.0 * 
                        (b_times_2 + a *
                            (
                                F1_plus_F2+Ly_plus_Ry
                            )
                        );
                        let other_factor = 0.5*F1_minus_F2*Ly_minus_Ry*(
                            b_times_2+a*(
                                F1_plus_F2+Ly_plus_Ry
                            )
                        );
                        let calc = |z_slice: &[f64], val_slice: &mut [f64]|
                        {
                            let mut first = None;
                            z_slice.iter()
                                .zip(val_slice.iter_mut())
                                .for_each(
                                    |(z, val)|
                                    {
                                        let one_minus_z = 1.0 - z;
                                        let res = 
                                        //other_factor.mul_add(
                                        //    one_minus_z, 
                                        //    outer*(
                                        //        inner_factor.mul_add(
                                        //            one_minus_z+F1, 
                                        //            inner_summand
                                        //        )
                                        //    )
                                        //);
                                        twelve_recip*(F1-F2)*(Ly-Ry)*(-((F1-F2)*(6.0*b+a*(2.0*F1+4.0*F2+3.0*(Ly+Ry))))
                                        + 6.0*(2.0*b+a*(F1+F2+Ly+Ry))
                                        *(1.0+F1-z))
                                        +0.5*(F1-F2)*(Ly-Ry)*(2.0*b+a*(F1+F2+Ly+Ry))*(1.0-z);
                                        if first.is_none(){
                                            first = Some((res, z));
                                        }
                                        *val += res;
                                    }
                                );
                            dbg!(first);
                        };
                        
                        debug_assert_eq!(z_left_slice.len(), left_rest.len());
                        println!("here");
                        dbg!(z_left_slice);
                        dbg!(from_s);
                        calc(z_left_slice, left_rest);
                        calc(from_s, &mut matrix_slice.right_borders);

                    }
                    //panic!();
                }
            );
        
        Self { matrix }
    }

    pub fn write_sum_all(&self, bins: &Bins, name: &str)
    {
        let (first, rest) = self.matrix.split_first().unwrap();
        let mut sum_left = first.left_borders.clone();
        let mut sum_right = first.right_borders.clone();
        rest.iter()
            .for_each(
                |slice|
                {
                    let sum = |into: &mut [f64], from: &[f64]|
                    {
                        into
                            .iter_mut()
                            .zip(from)
                            .for_each(
                                |(a,b)|
                                *a += b
                            );
                    };
                    sum(&mut sum_left, &slice.left_borders);
                    sum(&mut sum_right, &slice.right_borders);
                }
            );

        let mut sum_density = DensityI{
            left_borders: sum_left,
            right_borders: sum_right
        };
        sum_density.normalize(bins);

        sum_density.write(bins, name);
        let integral = sum_density.integral(bins);
        println!(
            "res_integral: {integral}"
        );
    }
}


fn add(matr_slice: &mut [f64], value_slice: &[f64])
{
    matr_slice
        .iter_mut()
        .zip(value_slice)
        .for_each(
            |(matr_value, helper_value)|
            {
                *matr_value += helper_value;
            }
        );
}