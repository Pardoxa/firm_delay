use std::num::NonZeroUsize;
use itertools::*;
use serde::{Serialize, Deserialize};
use derivative::Derivative;
use std::io::Write;
use crate::create_buf_with_command_and_version_and_header;


#[derive(Debug, Clone, Derivative, Serialize, Deserialize)]
#[derivative(Default)]
    
pub struct ModelInput{
    pub s: f64,
    // Branch- or Child-count
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub z: NonZeroUsize,
    /// should be at least 1000
    #[derivative(Default(value="NonZeroUsize::new(10000).unwrap()"))]
    pub precision: NonZeroUsize
}

pub fn line_test(input: &ModelInput)
{
    let uniform = vec![1.0; input.precision.get()];
    // uniform is I_-1
    // now one up
    let pk = calc_k(&uniform, input.s, 1e-12);

    pk.write_files("N-2");

}

pub struct Pk{
    delta_left: f64,
    delta_right: f64,
    function: Vec<f64>,
    bin_size: f64,
    s: f64
}

impl Pk{
    pub fn write_files(self, stub: &str)
    {
        let header = [
            "k",
            "P(k)"
        ];
        let name = format!("s{}{stub}.dat", self.s);
        let mut buf_fun = create_buf_with_command_and_version_and_header(name, header);
        let header = [
            "k",
            "delta P(k)"
        ];
        let name = format!("s{}{stub}_delta.dat", self.s);
        let mut buf_delta: std::io::BufWriter<fs_err::File> = create_buf_with_command_and_version_and_header(name, header);
    
        for (i, val) in self.function.iter().enumerate(){
            let k = i as f64 * self.bin_size + self.bin_size / 2.0;
            writeln!(
                buf_fun,
                "{k} {val}"
            ).unwrap();
        }
    
        writeln!(
            buf_delta,
            "0 {}\n{} {}",
            self.delta_left,
            self.s,
            self.delta_right
        ).unwrap();
    }
}

// for now I think this only works for 0 < s < 1, 
// but I should be able to adjust this so it works for all s.
fn calc_k(a: &[f64], s: f64, threshold: f64) -> Pk
{
    let len = a.len();
    let index_s = (s * (len - 1) as f64).round() as usize;
    let bin_size = (len as f64).recip();

    let i_len = len as isize;

    let a_mul = a.iter()
        .map(
            |val| val * bin_size
        ).collect_vec();

    // a has same binsize as uniform
    let p_am: Vec<f64> = ((-i_len)..i_len)
        .map(
            |index|
            {
                let start = 0.max(index) as usize;
                let end = (index+i_len).min(i_len-1) as usize;

                a_mul[start..end]
                    .iter()
                    .sum()
            }
        ).collect_vec();

    assert!(index_s >= 99, "please increase precision");


    let mut delta_left = 0.3;
    let mut delta_right = 0.3;
    let guess_height = (1.0-delta_left-delta_right)/s;
    let mut k_guess = vec![guess_height; index_s + 1];

    let mut k_result = k_guess.clone();

    loop{
        
        k_result.iter_mut()
            .enumerate()
            .for_each(
                |(x, r)|
                {
                    // first the function f(x)
                    *r = 0.0; 
                    k_guess.iter().enumerate()
                        .for_each(
                            |(x_prime, k_of_x_prime)|
                            {
                                *r += bin_size * k_of_x_prime * p_am[len+x-x_prime];
                            }
                        );
                    // then the offset of the delta functions
                    *r += p_am[x+len] * delta_left;
                    *r +=p_am[len+x-index_s] * delta_right;
                }
            );
        
        let mut tmp_delta_left = 0.0;
        
        for x in 0..len {
            // f(x):
            let mut f_x = 0.0;
            for (x_prime, k_val) in k_guess.iter().enumerate() {
                if x >= x_prime{
                    f_x += k_val * p_am[x-x_prime] * bin_size;
                }
            }
            f_x += p_am[x] * delta_left;
            if x >= index_s{
                f_x += delta_right * p_am[x-index_s];
            }
            tmp_delta_left += f_x * bin_size;
        }

        let delta_left_diff = (delta_left - tmp_delta_left).abs();
        delta_left = tmp_delta_left;

        let mut total = 0.0;
        let mut diff = 0.0;
        k_result.iter()
            .zip(k_guess.iter())
            .for_each(
            |(new_val, old_val)|
            {
                total +=new_val * bin_size;
                diff += (old_val - new_val).abs();
            }
        );
        total += delta_left;
        let tmp_delta_right = 1.0 - total;

        let delta_right_diff = (delta_right - tmp_delta_right).abs();
        delta_right = tmp_delta_right;


        let sum_of_differences = diff + delta_left_diff + delta_right_diff;

        if sum_of_differences <= threshold{

            return Pk{
                delta_left,
                delta_right,
                function: k_result,
                bin_size,
                s
            };
        }

        // break before this!
        std::mem::swap(
            &mut k_guess, 
            &mut k_result
        );
    }

}