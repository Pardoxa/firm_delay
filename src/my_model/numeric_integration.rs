use std::num::NonZeroUsize;
use itertools::*;
use serde::{Serialize, Deserialize};
use derivative::Derivative;
use std::io::Write;

use crate::create_buf;


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
    calc_k(&uniform, input.s);

}

// for now I think this only works for 0 < s < 1, 
// but I should be able to adjust this so it works for all s.
fn calc_k(a: &[f64], s: f64)
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

    let mut counter = 0;
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

        let mut buf = create_buf(format!("test_{counter}.dat"));
    
        k_result.iter().enumerate()
            .for_each(
                |(index, val)|
                {
                    writeln!(
                        buf,
                        "{} {} {}",
                        index as f64 * bin_size,
                        (index + 1) as f64 * bin_size,
                        val
                    ).unwrap();
                }
            );

        let mut buf = create_buf(format!("test_{counter}_impuls.dat"));

        writeln!(
            buf,
            "0 {}\n{} {}",
            delta_left,
            s,
            delta_right
        ).unwrap();

        let sum_of_differences = diff + delta_left_diff + delta_right_diff;
        println!("sum_of_differences: {sum_of_differences}");
        if counter == 100{

            break
        }
        counter +=1;

        // break before this!
        std::mem::swap(
            &mut k_guess, 
            &mut k_result
        );
    }
    // result of the delta functions


    

    let mut buf = create_buf("pam.dat");
    p_am.iter().enumerate()
        .for_each(
            |(index, val)|
            {
                writeln!(
                    buf,
                    "{} {} {}",
                    index as f64 * bin_size - 1.0,
                    (index + 1) as f64 * bin_size - 1.0,
                    val
                ).unwrap();
            }
        );
    //offset should be
    //let result = delta_left * pam[x] + delta_right * pam[x-s];

}