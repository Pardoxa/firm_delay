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
        let debug_delta = match counter{
            1 => {
                DebugDelta{
                    left: None, //Some(0.42150594999999996),
                    right: None //Some(0.186465725)
                }
            },
            _ => {
                DebugDelta{
                    left: None,
                    right: None //0.186465725
                }
            }
        };

        let pk = calc_k(
            &a_ij, 
            input.s, 
            1e-4, 
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
    left: Option<f64>,
    right: Option<f64>
}

// for now I think this only works for 0 < s < 1, 
// but I should be able to adjust this so it works for all s.
fn calc_k(
    a: &[f64], 
    s: f64, 
    threshold: f64, 
    counter: usize,
    delta: DebugDelta
) -> Pk
{
    let len = a.len();
    let index_s = (s * (len - 1) as f64).floor() as usize;
    let bin_size = ((len+1) as f64).recip();

    let i_len = len as isize;

    let a_mul = a.iter()
        .map(
            |val| val * bin_size
        ).collect_vec();

    // Calculating for jump probability
    let mut p_am: Vec<f64> = ((-i_len)..i_len)
        .map(
            |index|
            {
                let start = 0.max(index) as usize;
                let end = (index+i_len).min(i_len) as usize;

                a_mul[start..end]
                    .iter()
                    .sum()
            }
        ).collect_vec();

    let total_jump_prob: f64 = p_am.iter()
        .map(|val| *val * bin_size)
        .sum();
    println!("JUMP: {total_jump_prob}");
    p_am.iter_mut()
        .for_each(
            |val| *val /= total_jump_prob
        );
    let total_jump_prob: f64 = p_am.iter()
        .map(|val| *val * bin_size)
        .sum();
    println!("JUMP Corrected: {total_jump_prob}");

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

    if let Some(delta) = delta.left.as_ref()
    {
        delta_left = *delta;
    }
    if let Some(delta) = delta.right.as_ref(){
        delta_right = *delta;
    }

    let guess_height = (1.0-delta_left-delta_right)/s;
    let mut k_guess = vec![guess_height*2.0; index_s+1]; // maybe I somewhere have indexmissmatch for index s?

    let mut guess_total = 0.0;
    k_guess.iter()
        .for_each(
            |val|
            {
                guess_total += *val;
            }
        );
    guess_total = guess_total * bin_size + delta_left + delta_right;
    println!("GUESS TOTAL: {guess_total}");
    let length = k_guess.len() as f64 * bin_size;
    println!("LENNN {length}");

    let mut k_result = k_guess.clone();
    let mut bla = 0;
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
                                *r += k_of_x_prime * p_am[len+x-x_prime];
                            }
                        );
                    *r *= bin_size;
                    // then the offset of the delta functions
                    *r += p_am[len+x] * delta_left;
                    *r += p_am[len+x-index_s] * delta_right;
                }
            );

        /*if delta.left.is_some() && delta.right.is_some(){
            let mut total: f64 = k_result.iter().sum();
            total *= bin_size;
            let factor = (1.0 - delta_left - delta_right)/total;
            k_result.iter_mut()
                .for_each(
                    |val| 
                    {
                        *val *= factor;
                    }
                )
        }*/
        
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

        let dd = delta_left - tmp_delta_left;
        println!("DELTA_DIFF: {dd}");

        let mut total: f64 = 0.0;
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
        let mut tmp_delta_right;
        {
            // calc delta right
            // integral integral pk(x)p_am(z)dz
            // borders: x+z=s -> s-x
            // uper border infinity
            let mut tmp = 0.0;
            k_guess.iter().enumerate()
                .for_each(
                    |(index, k_val)|
                    {
                        let mut integral = 0.0;
                        let start = index_s + len - index;
                        for jump in &p_am[start..]{
                            integral += jump;
                        }
                        tmp += integral * k_val;
                    }
                );
            tmp *= bin_size;
            let reachable_by_left: f64 = p_am[len+index_s..].iter().sum();
            let reachable_by_right: f64 = p_am[len..].iter().sum();
            tmp += reachable_by_left * delta_left + reachable_by_right * delta_right;
            tmp *= bin_size;
            println!("DELTA RIGHT: {tmp}");
            println!("OTHER DELTA RIGHT {delta_right}");
            tmp_delta_right = tmp;
        }

        let mut delta_left_diff = 0.0;
        let mut delta_right_diff = 0.0;
        if let Some(left) = delta.left.as_ref()
        {
            delta_left = *left;

            
        }else {
            tmp_delta_left = 1.0 - total - delta_right;
            delta_left_diff = (delta_left - tmp_delta_left).abs();
            delta_left = tmp_delta_left;
        }
        
        if let Some(right) = delta.right.as_ref(){
            delta_right = *right;
        } else {
            //tmp_delta_right = 1.0 - total - delta_left;
    
            delta_right_diff = (delta_right - tmp_delta_right).abs();
            delta_right = tmp_delta_right;
        }

        /*total += delta_left + delta_right;

        k_result.iter_mut()
            .for_each(
                |val| *val /= total
            );
        delta_left /= total;
        delta_right /= total;

        let mut checking_total = 0.0;
        for k in k_result.iter(){
            checking_total += k;
        }
        checking_total *= bin_size;
        checking_total += delta_left + delta_right;
        println!("CHECKING: {checking_total}");*/

       

        dbg!(diff);
        dbg!(delta_left_diff);
        dbg!(delta_right_diff);
        let sum_of_differences = diff + delta_left_diff + delta_right_diff;
        println!("differences: {sum_of_differences}");
        println!("counter: {counter} total: {total}");
        if bla > 100 || sum_of_differences <= threshold{

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
        bla += 1;

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

    //let mut derivative = sampling::glue::derivative::derivative(&prob[..pk.len_of_1]);
    let derivative_left = sampling::glue::derivative::derivative(&prob[..=pk.index_s]);
    let derivative_right = sampling::glue::derivative::derivative(&prob[pk.index_s+1..pk.len_of_1]);
    let mut derivative = derivative_left;
    derivative.extend_from_slice(&derivative_right);

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

