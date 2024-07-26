use std::num::NonZeroUsize;
use indicatif::ProgressIterator;
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

#[allow(non_snake_case)]
pub fn line_test(input: &ModelInput)
{
    let uniform = vec![1.0; input.precision.get()];
    // uniform is I_-1
    // now one up

    let counter = 0;
    let pk = master_ansatz_k(
        &uniform, 
        input.s, 
        1e-4, 
        counter,
        DebugDelta{left: None, right: None}
    );
    let stub = format!("_PK{counter}");
    pk.write_files(&stub);
    
    let after_i = calc_I(&pk, &uniform, counter); 

    let P_I_given_prior_I = master_ansatz_i_test(&pk, &uniform, &after_i);

    calk_k_master_test(
        &pk,
        &P_I_given_prior_I,
        &after_i
    );

    // OLD:
    /*for counter in 0..3{
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

        let pk = master_ansatz_k(
            &a_ij, 
            input.s, 
            1e-4, 
            counter,
            debug_delta
        );
        let stub = format!("_PK{counter}");
        pk.write_files(&stub);
        a_ij = calc_I(&pk, &a_ij, counter);
    }*/


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

pub struct ProbabilityDensity{
    pub func: Vec<f64>,
    pub delta: (f64, f64)
}

impl ProbabilityDensity{
    pub fn new(len: usize, bin_size: f64) -> Self 
    {
        let delta = (0.1, 0.1);
        let height = 0.8 / (bin_size * len as f64);
        let func = vec![height; len];
        Self { func, delta }
    }

    pub fn new_zeroed(len: usize) -> Self 
    {
        let delta = (0.0, 0.0);
        let func = vec![0.0; len];
        Self { func, delta }
    }

    pub fn make_zero(&mut self)
    {
        self.delta.0 = 0.0;
        self.delta.1 = 0.0;
        self.func.iter_mut()
            .for_each(|val| *val = 0.0);
    }
}

#[allow(non_snake_case)]
fn calk_k_master_test(
    prior_pk: &Pk,
    input_P_I_given_prior_I: &[Vec<f64>],
    prior_I_for_normalization: &[f64]
){
    let mut current_estimate_given_prior_I = (0..input_P_I_given_prior_I.len())
        .map(|_| ProbabilityDensity::new(prior_pk.function.len(), prior_pk.bin_size))
        .collect_vec();

    let mut next_estimate_given_prior_I = (0..input_P_I_given_prior_I.len())
        .map(|_| ProbabilityDensity::new_zeroed(prior_pk.function.len()))
        .collect_vec();

    let idx_s = prior_pk.index_s;
    let len_of_1 = prior_pk.len_of_1;
    let bin_size = prior_pk.bin_size;


    let mut resulting_density = ProbabilityDensity::new_zeroed(prior_pk.function.len());
    let m_factor = (len_of_1 as f64).recip();
    for counter in 0..100{
        for (prior_I_idx, prior_I_prob) in prior_I_for_normalization.iter().enumerate().progress(){
            let current_k = &current_estimate_given_prior_I[prior_I_idx];
            let m_factor_times_prior_I_prob = prior_I_prob * m_factor;
            for (k, k_prob) in current_k.func.iter().enumerate(){
                let probability_increment = k_prob * m_factor_times_prior_I_prob;
                let kI = prior_I_idx + k;
                for m in 0..len_of_1{
                    let this_I = m.min(kI);
                    let update_k_vec = &mut next_estimate_given_prior_I[this_I];
    
                    if m > kI {
                        update_k_vec.delta.0 += probability_increment;
                    } else if kI -m > idx_s {
                        update_k_vec.delta.1 += probability_increment;
                    } else {
                        update_k_vec.func[kI-m] += probability_increment;
                    }
    
                }
            }
    
            // left
            let left_increment = m_factor_times_prior_I_prob * current_k.delta.0 / bin_size; // TODO probably /bin_size or something like that missing!
            for m in 0..len_of_1{
                let this_I = m.min(prior_I_idx);
                let update_k_vec = &mut next_estimate_given_prior_I[this_I];
                if m > prior_I_idx {
                    update_k_vec.delta.0 += left_increment;
                } else if prior_I_idx -m > idx_s {
                    update_k_vec.delta.1 += left_increment;
                } else {
                    update_k_vec.func[prior_I_idx-m] += left_increment;
                }
            }
    
            // right
            let right_increment = m_factor_times_prior_I_prob * current_k.delta.1 / bin_size;
            let kI = prior_I_idx + idx_s;
            for m in 0..len_of_1{
                let this_I = m.min(kI);
    
                let update_k_vec = &mut next_estimate_given_prior_I[this_I];
    
                if m > kI {
                    update_k_vec.delta.0 += right_increment;
                } else if kI -m > idx_s {
                    update_k_vec.delta.1 += right_increment;
                } else {
                    update_k_vec.func[kI-m] += right_increment;
                }
            }
        }
    
        next_estimate_given_prior_I.iter_mut()
            .for_each(
                |estimate| 
                {
                    estimate.delta.0 *= bin_size;
                    estimate.delta.1 *= bin_size
                }
            );
    
        // normalize (and print)
        for estimate in next_estimate_given_prior_I.iter_mut(){
            let sum: f64 = estimate.func.iter().sum();
            let integral = sum * bin_size + estimate.delta.0 + estimate.delta.1;
            println!(
                "I: {integral}"
            );
            let factor = integral.recip();
            estimate.delta.0 *= factor;
            estimate.delta.1 *= factor;
            estimate.func.iter_mut()
                .for_each(
                    |val|
                    {
                        *val *= factor;
                    }
                );
        }

        for (estimate, norm) in next_estimate_given_prior_I.iter().zip(prior_I_for_normalization){
            resulting_density.func
                .iter_mut()
                .zip(estimate.func.iter())
                .for_each(
                    |(res, est)|
                    {
                        *res += norm * est * bin_size;
                    }
                );
            resulting_density.delta.0 += norm * estimate.delta.0 * bin_size;
            resulting_density.delta.1 += norm * estimate.delta.1 * bin_size;
        }

        let name = format!("E_RES{counter}.dat");
        let mut buf = create_buf(name);
        for (i, val) in resulting_density.func.iter().enumerate(){
            let x = i as f64 * bin_size;
            writeln!(
                buf,
                "{x} {val}"
            ).unwrap();
        }
        
        let name = format!("E_RES_delta{counter}.dat");
        let mut buf = create_buf(name);
        writeln!(
            buf,
            "0 {}\n{} {}",
            resulting_density.delta.0,
            prior_pk.s,
            resulting_density.delta.1
        ).unwrap();

        resulting_density.make_zero();
    
        for i in (0..prior_I_for_normalization.len()).step_by(100)
        {
            let name = format!("P_k_given_I_c{counter}_{i}.dat");
            let mut buf = create_buf(name);
            let density = &next_estimate_given_prior_I[i];
    
            for (i, val) in density.func.iter().enumerate(){
                let x = i as f64 * bin_size;
                writeln!(
                    buf,
                    "{x} {val}"
                ).unwrap();
            }
    
            let name = format!("P_k_given_I_c{counter}_delta_{i}.dat");
            let mut buf = create_buf(name);
    
            writeln!(
                buf,
                "0 {}\n{} {}",
                density.delta.0,
                prior_pk.s,
                density.delta.1
            ).unwrap();
        }

        std::mem::swap(&mut next_estimate_given_prior_I, &mut current_estimate_given_prior_I);
        next_estimate_given_prior_I
            .iter_mut()
            .for_each(ProbabilityDensity::make_zero);
    }
    

}

/// For now only for N-2
/// this assumes that J (jump prob) is not dependent on k
#[allow(non_snake_case)]
fn master_ansatz_i_test(
    pk: &Pk,
    prob_prior_I: &[f64],
    prob_I_after: &[f64]
) -> Vec<Vec<f64>>
{
    // Given I(t) I want to know P_I(t+1)
    // For this I first calculate:
    //      given I(t) what is P_k(t)

    let mut Ik_matr = vec![vec![0.0; pk.function.len()]; prob_prior_I.len()];
    let mut delta_matr = vec![(0.0,0.0); prob_prior_I.len()];

    let factor = 1.0 / (pk.len_of_1 * pk.len_of_1) as f64;
    for (k_idx, k_val) in pk.function.iter().enumerate()
    {
        let probability_of_k_branch = k_val * pk.bin_size;
        let probability_of_both_m = probability_of_k_branch * factor;
        for m1 in 0..pk.len_of_1{
            for m2 in 0..pk.len_of_1{
                let resulting_i_idx = m1.min(m2+k_idx);
                // There is certainly room for optimization here XD

                let new_k_idx = m2 + k_idx - resulting_i_idx;
                let ik_vec: &mut Vec<f64> = Ik_matr.get_mut(resulting_i_idx).unwrap();

                if new_k_idx > pk.index_s {
                    delta_matr[resulting_i_idx].1 += probability_of_both_m;
                } else if m1 > m2+k_idx {
                    delta_matr[resulting_i_idx].0 += probability_of_both_m;
                } else{
                    ik_vec[new_k_idx] += probability_of_both_m;
                }
            }
        }
    }
    

    let probability_of_k_branch = pk.delta_left;
    let probability_of_both_m = probability_of_k_branch * factor;
    for m1 in 0..pk.len_of_1{
        for m2 in 0..pk.len_of_1{
            let resulting_i_idx = m1.min(m2);
            // There is certainly room for optimization here XD
            
            let new_k_idx = m2 - resulting_i_idx;
            let ik_vec: &mut Vec<f64> = Ik_matr.get_mut(resulting_i_idx).unwrap();

            if new_k_idx > pk.index_s {
                delta_matr[resulting_i_idx].1 += probability_of_both_m;
            } else if m1 > m2 {
                delta_matr[resulting_i_idx].0 += probability_of_both_m;
            } else{
                ik_vec[new_k_idx] += probability_of_both_m;
            }
        }
    }
    
    let probability_of_k_branch = pk.delta_right;
    let probability_of_both_m = probability_of_k_branch * factor;
    for m1 in 0..pk.len_of_1{
        for m2 in 0..pk.len_of_1{
            let resulting_i_idx = m1.min(m2+pk.index_s);
            let new_k_idx = m2 + pk.index_s - resulting_i_idx;
            let ik_vec: &mut Vec<f64> = Ik_matr.get_mut(resulting_i_idx).unwrap();

            if new_k_idx > pk.index_s {
                delta_matr[resulting_i_idx].1 += probability_of_both_m;
            } else if m1 > m2+pk.index_s {
                delta_matr[resulting_i_idx].0 += probability_of_both_m;
            } else{
                ik_vec[new_k_idx] += probability_of_both_m;
            }
        }
    }
        
    // normalization
    // afterwards ik_vec[i][j] entries correspond to the probability that the next k value is j given the next I value i
    for (ik_vec, delta) in Ik_matr.iter_mut().zip(delta_matr.iter_mut())
    {
        let mut sum: f64= ik_vec.iter().sum();
        sum += delta.0 + delta.1;
        for ik_val in ik_vec.iter_mut()
        {
            *ik_val /= sum;
        }
        let factor = pk.bin_size / sum;
        delta.0 *= factor;
        delta.1 *= factor;
    }
    

    let mut resulting_prob = vec![0.0; pk.function.len()];
    let mut resulting_delta = (0.0, 0.0);
    for ((ik_vec, i_prob), delta) in Ik_matr.iter().zip(prob_I_after).zip(delta_matr.iter()){
        for (k_val, res) in ik_vec.iter().zip(resulting_prob.iter_mut())
        {
            *res += i_prob * k_val;
        }
        resulting_delta.0 += i_prob * delta.0;
        resulting_delta.1 += i_prob * delta.1;
    }

    // I think that is it. Now testing




    let mut buf = create_buf("Res.dat");
    for (idx, res) in resulting_prob.iter().enumerate(){
        let x = idx as f64 * pk.bin_size;
        writeln!(
            buf,
            "{x} {res}"
        ).unwrap();
    }
    let mut buf = create_buf("Res_delta.dat");
    writeln!(
        buf,
        "0 {}\n{} {}",
        resulting_delta.0,
        pk.s,
        resulting_delta.1
    ).unwrap();

    for ik_vec in Ik_matr.iter()
    {
        let sum: f64 = ik_vec.iter().sum();
        let val = sum * pk.bin_size;
        println!("{val}");
    }

    let mut P_I_given_old_I = vec![vec![0.0; prob_prior_I.len()]; Ik_matr.len()];
    let factor = 1.0 / (Ik_matr.len() * Ik_matr.len()) as f64;
    for (old_i_index, (k_dist, delta)) in Ik_matr.iter().zip(delta_matr.iter()).enumerate().progress()
    {
        for (k_index, k_prob_dens) in k_dist.iter().enumerate(){
            let probability_density_increment = k_prob_dens * factor;
            for m1 in 0..Ik_matr.len(){
                for m2 in 0..Ik_matr.len(){
    
                    let new_I = m1.min(m2 + k_index);
                    P_I_given_old_I[old_i_index][new_I] += probability_density_increment;
                }
            }
        }


        for m1 in 0..Ik_matr.len(){
            for m2 in 0..Ik_matr.len(){
                // left 
                let new_I = m1.min(m2);
                P_I_given_old_I[old_i_index][new_I] += delta.0 * factor / pk.bin_size; // TODO: Correct factor was not checked yet, might be something else!

                // right
                let new_I = m1.min(m2 + pk.index_s);
                P_I_given_old_I[old_i_index][new_I] += delta.1 * factor / pk.bin_size; // TODO: Correct factor was not checked yet, might be something else!
            }
        }

    }

    for (idx, vector) in P_I_given_old_I.iter().enumerate(){
        let mut i_buf = create_buf(format!("test_I_{idx}.dat"));
        let I = idx as f64 * pk.bin_size;
        writeln!(
            i_buf,
            "#{I}"
        ).unwrap();

        for (index, val) in vector.iter().enumerate()
        {
            let x = index as f64 * pk.bin_size;
            writeln!(
                i_buf,
                "{x} {val}"
            ).unwrap();
        }
    }

    // TODO: The resulting vector contains an off by one error - the discontinuity is off by one!
    let mut I_check = vec![0.0; prob_prior_I.len()];
    for (vec, prob) in P_I_given_old_I.iter().zip(prob_I_after){
        for (res, part) in I_check.iter_mut().zip(vec.iter())
        {
            *res += part * prob;
        }
    }

    let mut buf = create_buf("I_check.dat");
    for (i, val) in I_check.iter().enumerate(){
        let x = i as f64 * pk.bin_size;
        writeln!(
            buf,
            "{x} {val}"
        ).unwrap();
    }

    P_I_given_old_I
}

fn master_ansatz_k(
    a: &[f64], 
    s: f64, 
    threshold: f64, 
    counter: usize,
    delta: DebugDelta
)-> Pk
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


    let mut delta_left = 0.2;
    let mut delta_right = 0.2;

    if let Some(delta) = delta.left.as_ref()
    {
        delta_left = *delta;
    }
    if let Some(delta) = delta.right.as_ref(){
        delta_right = *delta;
    }

    let guess_height = (1.0-delta_left-delta_right)/s;
    let mut k_guess = vec![guess_height; index_s+1]; // maybe I somewhere have indexmissmatch for index s?

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


    let mut testing = 0;

    loop{
        let mut delta_left_input = 0.0;
        let mut delta_right_input = 0.0;
        k_result.iter_mut().for_each(|val| *val = 0.0);
        k_guess.iter()
            .enumerate()
            .for_each(
                |(index, val)|
                {
                    for (jump_index, prob) in p_am.iter().enumerate()
                    {
                        let amount = prob * val * bin_size;
                        let resulting_index = index as isize + jump_index as isize - i_len;
                        if resulting_index < 0 {
                            delta_left_input += amount;
                        } else if let Some(val) = k_result.get_mut(resulting_index as usize){
                            *val += amount;
                        } else {
                            delta_right_input += amount;
                        }
                    }
                }
            );


        // left delta
        for (jump_index, prob) in p_am.iter().enumerate()
        {
            let resulting_index = jump_index as isize - i_len;
            let amount = delta_left * prob;
            if resulting_index < 0 {
                delta_left_input += amount;
            } else if let Some(val) = k_result.get_mut(resulting_index as usize){
                *val += amount;
            } else {
                delta_right_input += amount;
            }
        }

        // right delta
        for (jump_index, prob) in p_am.iter().enumerate()
        {
            let resulting_index = index_s as isize + jump_index as isize - i_len;
            let amount = delta_right * prob;
            if resulting_index < 0 {
                delta_left_input += amount;
            } else if let Some(val) = k_result.get_mut(resulting_index as usize){
                *val += amount;
            } else {
                delta_right_input += amount;
            }
        }

        delta_left_input *= bin_size;
        delta_right_input *= bin_size;

       
        
        delta_left = delta_left_input;
        delta_right = delta_right_input;

        if testing == 100 {
            dbg!(&k_result);
            dbg!(delta_left_input);
            dbg!(delta_right_input);
    
            let mut test_buf = create_buf("test.dat");
            let mut delta = create_buf("test_delta.dat");
            for (index, val) in k_result.iter().enumerate(){
                let x = index as f64 * bin_size;
                writeln!(
                    test_buf,
                    "{x} {val}"
                ).unwrap();
            }
    
            writeln!(
                delta,
                "0 {}\n{} {}",
                delta_left_input,
                s,
                delta_right_input
            ).unwrap();
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
        testing += 1;

        std::mem::swap(&mut k_guess, &mut k_result);
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

