use std::num::NonZeroUsize;
use itertools::*;
use serde::{Serialize, Deserialize};
use derivative::Derivative;
use std::io::Write;
use crate::{create_buf, create_buf_with_command_and_version_and_header};


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
    let mut a_ij = vec![1.0; input.precision.get()];
    // uniform is I_-1
    // now one up


    for counter in 0..3{
        let debug_delta = None;
        //(counter == 1).then_some(
        //    DebugDelta{
        //        left: 0.42150594999999996,
        //        right: 0.186465725
        //    }
        //);

        let pk = calc_k(
            &a_ij, 
            input.s, 
            1e-12, 
            counter,
            debug_delta
        );
        let stub = format!("_PK{counter}");
        pk.write_files(&stub);
        a_ij = calc_I(&pk, &a_ij, counter);
    }


}

pub struct Pk{
    delta_left: f64,
    delta_right: f64,
    function: Vec<f64>,
    bin_size: f64,
    s: f64,
    len_of_1: usize,
    index_s: usize
}

impl Pk{
    pub fn write_files(&self, stub: &str)
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

pub struct DebugDelta{
    left: f64,
    right: f64
}

// for now I think this only works for 0 < s < 1, 
// but I should be able to adjust this so it works for all s.
fn calc_k(
    a: &[f64], 
    s: f64, 
    threshold: f64, 
    counter: usize,
    delta: Option<DebugDelta>
) -> Pk
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

    let name = format!("s{s}p_am{counter}.dat");
    let mut buf = create_buf(name);
    for (index, val) in p_am.iter().enumerate()
    {
        let x = index as f64 * bin_size;
        writeln!(
            buf,
            "{} {}",
            x,
            val
        ).unwrap();
    }

    assert!(index_s >= 99, "please increase precision");


    let mut delta_left = 0.01;
    let mut delta_right = 0.01;

    if let Some(delta) = delta.as_ref()
    {
        delta_left = delta.left;
        delta_right = delta.right;
    }

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
                    *r += p_am[len+x] * delta_left;
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

        let mut delta_left_diff = 0.0;
        let mut delta_right_diff = 0.0;
        if let Some(delta) = delta.as_ref()
        {
            delta_left = delta.left;
            delta_right = delta.right;

            diff = 0.0;
            let factor = (1.0-delta_left -delta_right)/total;
            total = 0.0;
            k_result.iter_mut()
                .zip(k_guess.iter())
                .for_each(
                |(new_val, old_val)|
                {
                    *new_val *= factor;
                    total += *new_val * bin_size;
                    diff += (old_val - *new_val).abs();
                }
            );
        }else {
            delta_left_diff = (delta_left - tmp_delta_left).abs();
            delta_left = tmp_delta_left;
    
            
            total += delta_left;
            let tmp_delta_right = 1.0 - total;
    
            delta_right_diff = (delta_right - tmp_delta_right).abs();
            delta_right = tmp_delta_right;
        }

       

        dbg!(diff);
        dbg!(delta_left_diff);
        dbg!(delta_right_diff);
        let sum_of_differences = diff + delta_left_diff + delta_right_diff;
        println!("differences: {sum_of_differences}");
        println!("counter: {counter} total: {total}");
        if sum_of_differences <= threshold{

            return Pk{
                delta_left,
                delta_right,
                function: k_result,
                bin_size,
                s,
                len_of_1: len,
                index_s
            };
        }

        // break before this!
        std::mem::swap(
            &mut k_guess, 
            &mut k_result
        );
    }

}

#[allow(non_snake_case)]
fn calc_I(
    pk: &Pk, 
    a_ij: &[f64], 
    counter: usize
) -> Vec<f64>
{

    let p_ka = (0..(pk.function.len() + pk.len_of_1))
        .map(
            |x|
            {
                let mut integral = 0.0;
                let start = if x < pk.len_of_1-1{
                    0
                } else {
                    x - (pk.len_of_1 - 1)
                };
                let end = if x >= pk.function.len(){
                    pk.function.len() - 1
                } else {
                    x
                };
                for j in start..=end{
                    if x - j == 10000 {
                        dbg!(x);
                        dbg!(j);
                        dbg!(pk.index_s);
                        dbg!(pk.len_of_1);
                        dbg!(end);
                        dbg!(start);
                    }
                    integral += pk.function[j] * a_ij[x-j];
                }
                integral *= pk.bin_size;

                if start == 0{
                    integral += pk.delta_left * a_ij[x]; 
                }
                if x >= pk.index_s && x-pk.index_s < a_ij.len() {
                    integral += pk.delta_right * a_ij[x-pk.index_s];
                }

                integral
            }
        ).collect_vec();

    let p_ka_total: f64 = p_ka.iter()
        .map(
            |val|
            {
                val * pk.bin_size
            }
        ).sum();
    println!("pka total: {p_ka_total}");

    let name = format!("s{}p_ka_{counter}.dat", pk.s);
    let mut buf = create_buf(name);
    for (index, val) in p_ka.iter().enumerate()
    {
        let x = index as f64 * pk.bin_size;
        writeln!(
            buf,
            "{} {}",
            x,
            val
        ).unwrap();
    }

    let mut prob = (0..p_ka.len())
        .map(
            |i|
            {
                let sum: f64 = p_ka[i..]
                    .iter()
                    .sum();
                sum * pk.bin_size
            }
        ).collect_vec();

    let name = format!("s{}prob_{counter}.dat", pk.s);
    let mut buf = create_buf(name);

    let error = prob[0];
    let error_correction_factor = error.recip();
    prob.iter_mut()
        .for_each(|val| *val *= error_correction_factor);

    for (index, val) in prob.iter().enumerate()
    {
        let x = index as f64 * pk.bin_size;
        writeln!(
            buf,
            "{} {}",
            x,
            val
        ).unwrap();
    }

    // now convert prob into cumulative prob

    prob.iter_mut()
        .enumerate()
        .for_each(
            |(idx, val)|
            {
                let x = idx as f64 * pk.bin_size;
                *val = 1.0 - (1.0 - x) * *val;
            }
        );

    let name = format!("s{}cum_prob_{counter}.dat", pk.s);

    let mut buf = create_buf(name);
    for (index, val) in prob.iter().enumerate()
    {
        let x = index as f64 * pk.bin_size;
        writeln!(
            buf,
            "{} {}",
            x,
            val
        ).unwrap();
    }

    let mut derivative = sampling::glue::derivative::derivative(&prob[..pk.len_of_1]);

    let len = pk.len_of_1 as f64;
    derivative.iter_mut()
        .for_each(
            |val|
            {
                *val *= len;
            }
        );

    let name = format!("s{}derivative_{counter}.dat", pk.s);
    let mut buf = create_buf(name);
    for (index, val) in derivative.iter().enumerate()
    {
        let x = index as f64 * pk.bin_size;
        writeln!(
            buf,
            "{} {}",
            x,
            val
        ).unwrap();
    }

    assert_eq!(
        derivative.len(),
        a_ij.len()
    );

    derivative
}

