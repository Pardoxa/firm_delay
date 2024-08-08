use std::{io::BufWriter, num::*};
use fs_err::File;
use indicatif::ProgressIterator;
use itertools::*;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use derivative::Derivative;
use std::io::Write;
use crate::misc::*;
/*
    IMPORTANT:

    To understand the relations between the quantities I recommend looking at:
    https://nxt.yfeld.de/apps/files_sharing/publicpreview/rTjakkQiiDcsCpH?file=/&fileId=53661&x=2560&y=1440&a=true&etag=22fca54f9c79ad4e9453e05457e29686
*/


#[derive(Debug, Clone, Derivative, Serialize, Deserialize, PartialEq)]
#[derivative(Default)]
    
pub struct ModelInput{
    pub s: f64,
    /// should be at least 1000
    #[derivative(Default(value="NonZeroUsize::new(10000).unwrap()"))]
    pub precision: NonZeroUsize,
    /// Save name
    pub input_save: Option<String>,
    /// For creating the output saves
    pub output_save_stub: String,
    /// where to stop
    #[derivative(Default(value="NonZeroUsize::new(1).unwrap()"))]
    pub n_max: NonZeroUsize,
    /// If none exist, no k_densities will be written
    pub write_densities_stub: Option<String>
}


pub struct Crit{
    left: f64,
    left_prob: f64,
    right: f64,
    right_prob: f64,
    total: f64
}

impl Crit{
    pub fn write<W>(&self, mut buf: W, n: usize)
    where W: std::io::Write
    {
        writeln!(
            buf,
            "{n} {} {} {} {} {}",
            self.left,
            self.left_prob,
            self.right,
            self.right_prob,
            self.total
        ).unwrap()
    }
}

#[allow(non_snake_case)]
pub fn calc_crit(I: &[f64], param: &Parameter) -> Crit
{
    let bin_size2 = param.bin_size * param.bin_size;

    let (left_slice, right_slice) = I.split_at(param.index_s);

    let left: f64 = left_slice.iter()
        .enumerate()
        .map(
            |(idx, I_density)|
            {
                I_density * bin_size2 * idx as f64
            }
        ).sum();
    let mut left_prob = left_slice.iter().sum();
    left_prob *= param.bin_size;
    let right = right_slice.iter()
        .zip(param.index_s..)
        .map(
            |(I_density, idx)|
            {
                I_density * bin_size2 * idx as f64
            }
        ).sum();
    let mut right_prob = right_slice.iter()
        .sum();
    right_prob *= param.bin_size;
    Crit{
        left,
        right,
        total: left + right,
        left_prob,
        right_prob
    }
}

#[allow(non_snake_case)]
pub fn compute_line(input: ModelInput)
{
    /// TODO: p "s0.5_pr401__s.dat" u 1:3 w lp, "" u 1:5 w lp, "s0.5_pr501__s.dat" u 1:3 w lp, "" u 1:5 w lp
    /// The probability to be left should be the same as the probability to be right!
    let f_stub = format!("s{}_pr{}", input.s, input.precision);
    
    let s_name = format!("{f_stub}_s.dat");
    let mut s_buf = create_buf_with_command_and_version(s_name);

    // here I count: N=0 is leaf, N=1 is first node after etc
    let I0 = vec![1.0; input.precision.get()];

    let (parameter, k_i1j0) = master_ansatz_k(
        &I0, 
        input.s, 
        1e-8
    );

    let crit = calc_crit(&I0, &parameter);
    crit.write(&mut s_buf, 0);

    let I1 = calc_I(
        &parameter,
        &k_i1j0, 
        &I0
    ); 
    let crit = calc_crit(&I1, &parameter);
    crit.write(&mut s_buf, 1);

    if let Some(stub) = input.write_densities_stub.as_deref(){
        let stub = format!("{stub}_{f_stub}_1");
        let name = format!("{stub}_I.dat");
        write_I(&I1, parameter.bin_size, &name);
        let name = format!("{f_stub}_0_I.dat");
        write_I(&I0, parameter.bin_size, &name);
        k_i1j0.write(&stub, &parameter);
    }

    let I1_given_prev_I1 = master_ansatz_i_Ij_independent(
        &parameter,
        &I1,
        &k_i1j0
    );

    let Ij = I1;
    let Ij_given_prev_Ij = I1_given_prev_I1;

    let save = SaveState{
        f_stub,
        Ij,
        Ij_given_prev_Ij,
        j_idx: 1,
        parameter
    };

    continue_calculation(
        save, 
        input.n_max, 
        s_buf, 
        input.write_densities_stub
    );


}

#[allow(non_snake_case)]
pub fn continue_calculation(
    save: SaveState, 
    max_n: NonZeroUsize, 
    mut s_buf: BufWriter<File>,
    write_densities_stub: Option<String>
)
{
    let mut Ij_given_prev_Ij = save.Ij_given_prev_Ij;
    let mut Ij = save.Ij;
    let parameter = save.parameter;
    let f_stub = save.f_stub;

    for i in 2..=max_n.get(){
        println!("i = {i};");
        let (k_ij_given_Ij, k_ij) = calk_k_master_test(
            &parameter,
            &Ij_given_prev_Ij,
            &Ij,
            1e-7
        );
    
    
        let (Ii_given_prev_Ii, Ii) = master_ansatz_i_Ij_dependent(
            &parameter,
            &k_ij, 
            &Ij, 
            &Ij_given_prev_Ij,
            &k_ij_given_Ij
        );
        let crit = calc_crit(&Ii, &parameter);
        crit.write(&mut s_buf, i);

        if let Some(stub) = write_densities_stub.as_deref(){
            let stub = format!("{stub}_{f_stub}_1");
            let name = format!("{stub}_I.dat");
            write_I(&Ii, parameter.bin_size, &name);
            k_ij.write(&stub, &parameter);
        }

        Ij = Ii;
        Ij_given_prev_Ij = Ii_given_prev_Ii;
    }

    let save_name_json = format!("{f_stub}_{}.json", max_n);
    let save_name_bincode = format!("{f_stub}_{}.bincode", max_n);
    println!("saving: {save_name_json}");
    let save_buf = create_buf(save_name_json);
    let save = SaveState{
        Ij,
        Ij_given_prev_Ij,
        j_idx: max_n.get(),
        parameter,
        f_stub
    };
    serde_json::to_writer_pretty(save_buf, &save)
        .unwrap();
    println!("saving: {save_name_bincode}");
    let save_buf = create_buf(save_name_bincode);
    bincode::serialize_into(save_buf, &save)
        .unwrap();
}


#[allow(non_snake_case)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaveState{
    Ij: Vec<f64>,
    Ij_given_prev_Ij: Vec<Vec<f64>>,
    j_idx: usize,
    parameter: Parameter,
    f_stub: String
}

#[allow(non_snake_case)]
pub fn write_I(I: &[f64], bin_size: f64, name: &str)
{
    let mut buf = create_buf_with_command_and_version(name);
    for (i, val) in I.iter().enumerate(){
        let x = i as f64 * bin_size;
        writeln!(
            buf,
            "{x} {val}"
        ).unwrap();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter{
    bin_size: f64,
    s: f64,
    len_of_1: usize,
    index_s: usize,
    len_of_k_func: usize
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ProbabilityDensity{
    pub func: Vec<f64>,
    pub delta: (f64, f64)
}

impl ProbabilityDensity{

    pub fn normalize(&mut self, bin_size: f64)
    {
        let factor = self.integral(bin_size).recip();
        self.delta.0 *= factor;
        self.delta.1 *= factor;
        self.func.iter_mut()
            .for_each(
                |val| *val *= factor
            );
    }

    pub fn integral(&self, bin_size: f64) -> f64
    {
        let sum: f64 = self.func.iter().sum();
        sum * bin_size + self.delta.0 + self.delta.1
    }

    pub fn new(len: usize, bin_size: f64) -> Self 
    {
        let delta = (0.4, 0.4);
        let height = 0.2 / (bin_size * len as f64);
        let func = vec![height; len];
        Self { func, delta }
    }

    pub fn new_zeroed(len: usize) -> Self 
    {
        let delta = (0.0, 0.0);
        let func = vec![0.0; len];
        Self { func, delta }
    }

    pub fn create_zeroed(&self) -> Self{
        let len = self.func.len();
        Self::new_zeroed(len)
    }

    pub fn make_zero(&mut self)
    {
        self.delta.0 = 0.0;
        self.delta.1 = 0.0;
        self.func.iter_mut()
            .for_each(|val| *val = 0.0);
    }

    pub fn write(&self, stub: &str, parameter: &Parameter)
    {
        let name_func = format!("{stub}_func.dat");
        let mut buf = create_buf_with_command_and_version(name_func);

        for (i, val) in self.func.iter().enumerate() {
            let x = i as f64 * parameter.bin_size;
            writeln!(
                buf,
                "{x} {val}"
            ).unwrap();
        }

        let name = format!("{stub}_delta.dat");
        let mut buf = create_buf_with_command_and_version(name);
        writeln!(
            buf,
            "0 {}\n{} {}",
            self.delta.0,
            parameter.s,
            self.delta.1
        ).unwrap();
    }

    pub fn add_scaled(&mut self, other: &Self, scaling_factor: f64)
    {
        self.delta.0 += other.delta.0 * scaling_factor;
        self.delta.1 += other.delta.1 * scaling_factor;
        self.func
            .iter_mut()
            .zip(other.func.iter())
            .for_each(
                |(a, b)| *a += b * scaling_factor
            );
    }
}

// matrix needs to be square matrix
#[allow(dead_code)]
pub fn reverse_prob_matrix(
    a_given_b: &[Vec<f64>], //matrix
    probability_b: &[f64],
    bin_size: f64
) -> Vec<Vec<f64>>
{
    let mut b_give_a = a_given_b.iter()
        .map(|line| vec![0.0; line.len()])
        .collect_vec();

    for (b_idx, (a_line, b_prob)) in a_given_b.iter().zip(probability_b).enumerate()
    {
        for (a_prob, b_line) in a_line.iter().zip(b_give_a.iter_mut()){
            b_line[b_idx] += a_prob * b_prob;
        }
    }

    // normalization
    normalize_prob_matrix(&mut b_give_a, bin_size);
    b_give_a
}

fn normalize_prob_matrix(matr: &mut [Vec<f64>], bin_size: f64)
{
    matr.iter_mut()
        .for_each(
            |line|
            {
                normalize_vec(line, bin_size);
            }
        );
}

fn normalize_vec(vec: &mut [f64], bin_size: f64)
{
    let mut sum: f64 = vec.iter().sum();
    sum *= bin_size;
    let factor = sum.recip();
    vec.iter_mut()
        .for_each(|val| *val *= factor)
}

#[allow(non_snake_case)]
fn calk_k_master_test(
    parameter: &Parameter,
    input_P_I_given_prior_I: &[Vec<f64>],
    prior_I_for_normalization: &[f64],
    threshold: f64
) -> (Vec<ProbabilityDensity>, ProbabilityDensity)
{
    let mut current_kij_t0_estimate_given_Ij_t0 = (0..input_P_I_given_prior_I.len())
        .map(|_| ProbabilityDensity::new(parameter.len_of_k_func, parameter.bin_size))
        .collect_vec();

    let mut next_kij_t0_estimate_given_Ij_t0 = (0..input_P_I_given_prior_I.len())
        .map(|_| ProbabilityDensity::new_zeroed(parameter.len_of_k_func))
        .collect_vec();

    let idx_s = parameter.index_s;
    let len_of_1 = parameter.len_of_1;
    let bin_size = parameter.bin_size;


    let mut resulting_density = ProbabilityDensity::new_zeroed(parameter.len_of_k_func);
    let m_factor = (len_of_1 as f64).recip();

    let for_helper = |kI: usize, update_k_vec: &mut ProbabilityDensity, probability_increment: f64|
    {
        // first border exlusive
        let m_range_delta_left = kI.min(len_of_1)..len_of_1;
        let weight = if kI == 0{
            m_range_delta_left.len() - 1
        } else {
            m_range_delta_left.len()
        };
        update_k_vec.delta.0 += probability_increment * weight as f64;
        let m_range_delta_right = if kI >= idx_s{
            let end = if kI >= idx_s{
                kI - idx_s
            } else {
                0
            };
            0..end
        } else {
            0..0
        };
        update_k_vec.delta.1 += probability_increment * m_range_delta_right.len() as f64;
        let m_range_mid = (m_range_delta_right.end)..m_range_delta_left.start;
        let k_range = (kI-m_range_mid.end)..(kI-m_range_mid.start).max(1);
        /*
        dbg!(kI);
        dbg!(&m_range_delta_left);
        dbg!(&m_range_delta_left.len());
        dbg!(&m_range_delta_right);
        dbg!(&m_range_delta_right.len());
        dbg!(&m_range_mid);
        dbg!(&m_range_mid.len());
        dbg!(kI-m_range_mid.start);
        dbg!(kI - m_range_mid.end);
        dbg!(&k_range);
        */
        debug_assert_eq!(
            weight + m_range_delta_right.len() + k_range.len(),
            len_of_1
        );
        update_k_vec.func[k_range]
            .iter_mut()
            .for_each(
                |val| *val += probability_increment
            );
    };
    loop {
        for (prior_Ij_idx, current_Ij_distribution) in input_P_I_given_prior_I.iter().enumerate().progress(){
            let current_k = &current_kij_t0_estimate_given_Ij_t0[prior_Ij_idx];
            let prior_I_prob = prior_I_for_normalization[prior_Ij_idx];
            let m_factor_times_prior_I_prob = prior_I_prob * m_factor;
            for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                let update_k_vec = &mut next_kij_t0_estimate_given_Ij_t0[Ij_idx];
                let level_2_prob = m_factor_times_prior_I_prob * Ij_prob;
                for (k, k_prob) in current_k.func.iter().enumerate(){
                    let probability_increment = k_prob * level_2_prob;
                    let kI = Ij_idx + k;

                    
                    for_helper(kI, update_k_vec, probability_increment);
                }

            }
    
            // left
            let left_increment = m_factor_times_prior_I_prob * current_k.delta.0 / bin_size; 
            for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                let kI = Ij_idx;
                let update_k_vec = &mut next_kij_t0_estimate_given_Ij_t0[Ij_idx];
                let probability_increment = left_increment * Ij_prob;
                
                for_helper(kI, update_k_vec, probability_increment);
            }
    
            // right
            let right_increment = m_factor_times_prior_I_prob * current_k.delta.1 / bin_size;
            for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                let kI: usize = Ij_idx + idx_s;
                let probability_increment = right_increment * Ij_prob;
                let update_k_vec = &mut next_kij_t0_estimate_given_Ij_t0[Ij_idx];
                
                for_helper(kI, update_k_vec, probability_increment);
            }

        }
    
        next_kij_t0_estimate_given_Ij_t0.iter_mut()
            .for_each(
                |estimate| 
                {
                    // This is an optimization
                    // It is faster to multiply by bin_size here
                    // Than to do it over and over again in the loop
                    estimate.delta.0 *= bin_size;
                    estimate.delta.1 *= bin_size;

                    // Normalize the estimate
                    estimate.normalize(bin_size);
                }
            );

        for (estimate, norm) in next_kij_t0_estimate_given_Ij_t0.iter().zip(prior_I_for_normalization){
            resulting_density.func
                .iter_mut()
                .zip(estimate.func.iter())
                .for_each(
                    |(res, est)|
                    {
                        *res += norm * est * bin_size;
                    }
                );
            let delta_norm = norm * bin_size;
            resulting_density.delta.0 += estimate.delta.0 * delta_norm;
            resulting_density.delta.1 += estimate.delta.1 * delta_norm;
        }

        let estimate_diff: f64 = current_kij_t0_estimate_given_Ij_t0.iter()
            .zip(next_kij_t0_estimate_given_Ij_t0.iter())
            .map(
                |(current_estimate, next_estimate)|
                {
                    let diff_sum: f64 = current_estimate.func.iter()
                        .zip(next_estimate.func.iter())
                        .map(|(a,b)| (a-b).abs())
                        .sum();
                    let delta_diff = (current_estimate.delta.0 - next_estimate.delta.0).abs()
                        + (current_estimate.delta.1 - next_estimate.delta.1).abs();
                    diff_sum * bin_size + delta_diff
                }
            ).sum();
        println!("Estimate_diff: {estimate_diff}");
        std::mem::swap(&mut next_kij_t0_estimate_given_Ij_t0, &mut current_kij_t0_estimate_given_Ij_t0);
        if estimate_diff <= threshold {
            break;
        }
        next_kij_t0_estimate_given_Ij_t0
            .iter_mut()
            .for_each(ProbabilityDensity::make_zero);
        resulting_density.make_zero();
    }
    
    (current_kij_t0_estimate_given_Ij_t0, resulting_density)

}

#[allow(non_camel_case_types)]
#[derive(Clone)]
struct Ii_given_k {
    delta_left: Vec<f64>,
    delta_right: Vec<f64>,
    func: Vec<Vec<f64>>
}

impl Ii_given_k{
    pub fn normalize(&mut self, bin_size: f64){
        normalize_vec(self.delta_left.as_mut(), bin_size);
        normalize_vec(self.delta_right.as_mut(), bin_size);
        normalize_prob_matrix(&mut self.func, bin_size);
    }
}

/// For now only for N-2
/// this assumes that J (jump prob) is not dependent on k
#[allow(non_snake_case)]
fn master_ansatz_i_Ij_independent(
    parameter: &Parameter,
    prob_Ii: &[f64],
    kij_density: &ProbabilityDensity
) -> Vec<Vec<f64>>
{
    let len = prob_Ii.len();
    let idx_s = parameter.index_s;
    let bin_size = parameter.bin_size;

    let mut Ii_given_k = Ii_given_k{
        delta_left: vec![0.0; len],
        delta_right: vec![0.0; len],
        func: vec![vec![0.0; len]; parameter.len_of_k_func]
    };

    let len_recip = (len as f64).recip();
    let len_recip2 = len_recip * len_recip;

    let Ii_given_k_for_helper = |Ii: &mut [f64], kIj: usize| 
    {

        let right = kIj.min(len);
        // Ij is independent of k here, thank god
        // (its uniform)
        Ii[..right]
            .iter_mut()
            .for_each(
                |entry|
                {
                    *entry += len_recip2;
                }
            );
        let remaining = len - right;
        if remaining > 0{
            Ii[kIj] += len_recip2 * remaining as f64;
        }
    };

    for (k, Ii) in Ii_given_k.func.iter_mut().enumerate()
    {
        for Ij in 0..len {
            let kIj = k + Ij;
            Ii_given_k_for_helper(Ii, kIj);
        }
    }

    // delta left 
    let Ii_delta_left = Ii_given_k.delta_left.as_mut_slice();
    for Ij in 0..len {
        let kIj = Ij;
        Ii_given_k_for_helper(Ii_delta_left, kIj);
    }

    // delta right 
    let Ii_delta_right = Ii_given_k.delta_right.as_mut_slice();
    for Ij in 0..len {
        let kIj = Ij + idx_s;
        Ii_given_k_for_helper(Ii_delta_right, kIj);
    }

    Ii_given_k.normalize(bin_size);

    // SANITY CHECK
    {
        let mut  sanity_check = vec![0.0; len];

        for (prob, vec) in kij_density.func.iter().zip(Ii_given_k.func.iter())
        {
            let prob = prob * bin_size;
            sanity_check.iter_mut()
                .zip(vec)
                .for_each(
                    |(a,b)|
                    {
                        *a += b * prob;
                    }
                );
        }
    
        let prob = kij_density.delta.0;
        sanity_check.iter_mut()
            .zip(Ii_given_k.delta_left.iter())
            .for_each(
                |(a,b)|
                {
                    *a += b * prob;
                }
            );
        let prob = kij_density.delta.1;
        sanity_check.iter_mut()
            .zip(Ii_given_k.delta_right.iter())
            .for_each(
                |(a,b)|
                {
                    *a += b * prob;
                }
            );
    
        normalize_vec(&mut sanity_check, bin_size);
    
        write_I(&sanity_check, bin_size, "sanity_gone.dat");
    }

    let mut Ii_given_this_k_and_this_Ij = vec![vec![vec![0.0; len]; len]; kij_density.func.len()];
    let mut Ii_given_this_k_delta_left_and_this_Ij = vec![vec![0.0; len]; len];
    let mut Ii_given_this_k_delta_right_and_this_Ij = vec![vec![0.0; len]; len];

    let Ii_given_this_k_and_this_Ij_for_helper = |kIj: usize, Ii_given_k_and_Ij: &mut [f64]|
    {
        let right = kIj.min(len);
        Ii_given_k_and_Ij[..right]
            .iter_mut()
            .for_each(
                |val| *val += len_recip
            );
        let remaining = len - right;
        if remaining > 0 {
            Ii_given_k_and_Ij[kIj] += len_recip * remaining as f64;
        }
    };

    for (k, Ii_given_k) in Ii_given_this_k_and_this_Ij.iter_mut().enumerate() {
        for (Ij, Ii_given_k_and_Ij) in Ii_given_k.iter_mut().enumerate(){
            let kIj = k + Ij;
            Ii_given_this_k_and_this_Ij_for_helper(kIj, Ii_given_k_and_Ij);
        }
    }
    // delta left 
    for (Ij, Ii_given_k_and_Ij) in Ii_given_this_k_delta_left_and_this_Ij.iter_mut().enumerate()
    {
        let kIj = Ij; // k = 0
        Ii_given_this_k_and_this_Ij_for_helper(kIj, Ii_given_k_and_Ij);
    }
    // delta right 
    for (Ij, Ii_given_k_and_Ij) in Ii_given_this_k_delta_right_and_this_Ij.iter_mut().enumerate()
    {
        let kIj = Ij + idx_s; 
        Ii_given_this_k_and_this_Ij_for_helper(kIj, Ii_given_k_and_Ij);
    }

    Ii_given_this_k_and_this_Ij
        .par_iter_mut()
        .for_each(
            |line|
            {
                normalize_prob_matrix(line, bin_size)
            }
        );
    normalize_prob_matrix(&mut Ii_given_this_k_delta_left_and_this_Ij, bin_size);
    normalize_prob_matrix(&mut Ii_given_this_k_delta_right_and_this_Ij, bin_size);

    // sanity
    {
        let mut another_sanity = vec![0.0; len];
        for (matr, k_prob) in Ii_given_this_k_and_this_Ij.iter().zip(kij_density.func.iter()){
            let prob = k_prob * bin_size * len_recip;
            for line in matr{
                another_sanity
                    .iter_mut()
                    .zip(line)
                    .for_each(
                        |(a,b)|
                        {
                            *a += b * prob;
                        }
                    )
            }
        }
        // left 
        let prob = kij_density.delta.0  * len_recip;
        for line in Ii_given_this_k_delta_left_and_this_Ij.iter(){
            another_sanity
                .iter_mut()
                .zip(line)
                .for_each(
                    |(a,b)|
                    {
                        *a += b * prob;
                    }
                )
        }
        // right
        let prob = kij_density.delta.1  * len_recip;
        for line in Ii_given_this_k_delta_right_and_this_Ij.iter(){
            another_sanity
                .iter_mut()
                .zip(line)
                .for_each(
                    |(a,b)|
                    {
                        *a += b * prob;
                    }
                )
        }
        normalize_vec(&mut another_sanity, bin_size);
        write_I(&another_sanity, bin_size, "another_sanity.dat");
    }
    

    // This aggregation only works for Ij uniform. Otherwise I need to also multiply with the prob of Ij
    let aggregate = |matr: &Vec<Vec<f64>>|
    {
        let mut sum = matr[0].clone();
        for line in &matr[1..]
        {
            sum.iter_mut()
                .zip(line)
                .for_each(|(a,b)| *a += b);
        }
        sum
    };


    // This aggregation only works for Ij uniform. Otherwise I need to also multiply with the prob of Ij
    let aggregated_Ii_given_this_k_and_this_Ij = Ii_given_this_k_and_this_Ij
        .iter()
        .map(aggregate)
        .collect_vec();
    let aggregated_Ii_given_this_k_delta_right_and_this_Ij = aggregate(&Ii_given_this_k_delta_right_and_this_Ij);

    let agg_counting = |res_Ii_vec: &mut [f64], aggregate: &[f64], prob: f64|
    {
        res_Ii_vec
            .iter_mut()
            .zip(aggregate)
            .for_each(
                |(res, Ii_prob)|
                {
                    *res += Ii_prob * prob
                }
            );
    };
    let mut Ii_given_prev_Ii = vec![vec![0.0; len]; len];
    let mut Ii_given_prev_Ii_for_helper = |prev_k: usize, prob: f64|
    {
        for prev_Ij in 0..len{
            let prev_kIj = prev_k + prev_Ij;

            let end = prev_kIj.min(len);
            let m_smaller_range = 0..end;
            for m in m_smaller_range{
                let prev_Ii = m;
                let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();
                let other_k = (prev_kIj - prev_Ii).min(idx_s);
                if other_k < idx_s {
                    let ag = aggregated_Ii_given_this_k_and_this_Ij[other_k].as_slice();
                    agg_counting(res_Ii_vec, ag, prob);
                } else {
                    let Ii_given_k_slice_agg = &aggregated_Ii_given_this_k_delta_right_and_this_Ij;
                    agg_counting(res_Ii_vec, Ii_given_k_slice_agg, prob);
                }
                
            }
            if len > end {
                let kIj_range = end..len;
                let remaining = kIj_range.len();
                let prob = prob * remaining as f64;
                let prev_Ii = prev_kIj;
                let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();
                let other_k = (prev_kIj - prev_Ii).min(idx_s);
                let agg = aggregated_Ii_given_this_k_and_this_Ij[other_k].as_slice();

                agg_counting(res_Ii_vec, agg, prob);
            }
        }
    };

    let iter = kij_density
        .func.iter()
        .enumerate();

    for (prev_k, k_density) in iter {
        let prob: f64 = k_density * bin_size * len_recip2;
        Ii_given_prev_Ii_for_helper(prev_k, prob);
    }

    // delta left
    let prev_k_prob = kij_density.delta.0;
    let prob = prev_k_prob * len_recip2;
    Ii_given_prev_Ii_for_helper(0, prob);
    
    // delta right
    let prev_k_prob = kij_density.delta.1;
    let prob = prev_k_prob * len_recip2;
    Ii_given_prev_Ii_for_helper(idx_s, prob);

    normalize_prob_matrix(&mut Ii_given_prev_Ii, bin_size);

    let mut sanity_check_final = vec![0.0; len];

    for (slice, prob_density) in Ii_given_prev_Ii.iter().zip(prob_Ii)
    {
        let prob = prob_density * bin_size;
        sanity_check_final
            .iter_mut()
            .zip(slice)
            .for_each(
                |(a,b)|
                {
                    *a += b * prob;
                }
            );
    }
    write_I(&sanity_check_final, bin_size, "sanity_check_final_non_normalized.dat");
    normalize_vec(&mut sanity_check_final, bin_size);
    write_I(&sanity_check_final, bin_size, "sanity_check_final.dat");
    Ii_given_prev_Ii 
}

/// For now only for N-2
/// this assumes that J (jump prob) is not dependent on k
#[allow(non_snake_case)]
fn master_ansatz_i_Ij_dependent(
    parameter: &Parameter,
    kij_density: &ProbabilityDensity,
    prob_Ij: &[f64],
    Ij_given_prev_Ij: &[Vec<f64>],
    kij_t0_given_Ij_t0: &[ProbabilityDensity]
) -> (Vec<Vec<f64>>, Vec<f64>)
{
    let len = prob_Ij.len();
    let idx_s = parameter.index_s;
    let bin_size = parameter.bin_size;

    let Ii_given_k_zeroed = Ii_given_k{
        delta_left: vec![0.0; len],
        delta_right: vec![0.0; len],
        func: vec![vec![0.0; len]; kij_density.func.len()]
    };

    let mut Ii_given_Ij_and_kij = vec![Ii_given_k_zeroed; len];

    let len_recip = (len as f64).recip();
    let len_recip2 = len_recip * len_recip;

    let Ii_given_k_for_helper = |Ii: &mut [f64], kIj: usize, prob: f64| 
    {

        let right = kIj.min(len);
        // Ij is independent of k here, thank god
        // (its uniform)
        let internal_prob = prob * len_recip2;
        Ii[..right]
            .iter_mut()
            .for_each(
                |entry|
                {
                    *entry += internal_prob;
                }
            );
        let remaining = len - right;
        if remaining > 0{
            Ii[kIj] += internal_prob * remaining as f64;
        }
    };

    let prob = len_recip2;
    for (Ij, Ii_given_Ij_given_k) in Ii_given_Ij_and_kij.iter_mut().enumerate(){
        for (k, Ii) in Ii_given_Ij_given_k.func.iter_mut().enumerate()
        {
            let kIj = k + Ij;
            Ii_given_k_for_helper(Ii, kIj, prob);
        
        }
    
        // delta left 
        let Ii_delta_left = Ii_given_Ij_given_k.delta_left.as_mut_slice();
        
        let kIj = Ij;
        Ii_given_k_for_helper(Ii_delta_left, kIj, prob);
        
    
        // delta right 
        let Ii_delta_right = Ii_given_Ij_given_k.delta_right.as_mut_slice();
        
        let kIj = Ij + idx_s;
        Ii_given_k_for_helper(Ii_delta_right, kIj, prob);
        
    }
    
    Ii_given_Ij_and_kij.iter_mut()
        .for_each(|Ii_given_k| Ii_given_k.normalize(bin_size));

    // SANITY CHECK
    let mut Ii_probability_density = vec![0.0; len];
    {
        println!("Current sanity check");
        for (Ij_t0_idx, Ij_t0_density) in prob_Ij.iter().enumerate().progress(){
            let k_density_t0 = &kij_t0_given_Ij_t0[Ij_t0_idx];
            let Ij_t1_density = Ij_given_prev_Ij[Ij_t0_idx].as_slice();

            let level_0_prob = Ij_t0_density * bin_size;
            for (Ij_t1_dens, Ii_given_Ij_t1) in Ij_t1_density.iter().zip(Ii_given_Ij_and_kij.iter())
            {
                // Iterating through Ij_t1
                let level_1_prob = level_0_prob * Ij_t1_dens * bin_size;
                for (k_density, Ii_given_Ij_t1_kij_t0_density) in k_density_t0.func.iter().zip(Ii_given_Ij_t1.func.iter()){

                    let level_2_prob = level_1_prob * k_density * bin_size;

                    Ii_probability_density.iter_mut()
                        .zip(Ii_given_Ij_t1_kij_t0_density)
                        .for_each(
                            |(a,b)| *a += b * level_2_prob
                        );

                }

                // delta_left 
                let level_2_prob = level_1_prob * k_density_t0.delta.0;
                Ii_probability_density.iter_mut()
                    .zip(Ii_given_Ij_t1.delta_left.iter())
                    .for_each(
                        |(a,b)| *a += b * level_2_prob
                    );
                // delta_right 
                let level_2_prob = level_1_prob * k_density_t0.delta.1;
                Ii_probability_density.iter_mut()
                    .zip(Ii_given_Ij_t1.delta_right.iter())
                    .for_each(
                        |(a,b)| *a += b * level_2_prob
                    );
            }

        }
        normalize_vec(&mut Ii_probability_density, bin_size);
        write_I(&Ii_probability_density, bin_size, "next_sanity_gone.dat");
    }

    let mut k_tm1_given_Ij_t = vec![kij_density.create_zeroed(); len];
    
    let bin_size2 = bin_size * bin_size;
    for (Ij_tm1, Ij_tm1_density) in prob_Ij.iter().enumerate() {
        let kij_tm1_density = &kij_t0_given_Ij_t0[Ij_tm1];
        for (Ij_t0, Ij_t0_density) in Ij_given_prev_Ij[Ij_tm1].iter().enumerate()
        {
            k_tm1_given_Ij_t[Ij_t0].add_scaled(kij_tm1_density, Ij_tm1_density * Ij_t0_density * bin_size2);
        }
    }
    k_tm1_given_Ij_t.iter_mut()
        .for_each(|density| density.normalize(bin_size));

    let mut k_sanity = kij_density.create_zeroed();
    for (Ij_prob, k_density) in prob_Ij.iter().zip(k_tm1_given_Ij_t.iter())
    {
        k_sanity.add_scaled(k_density, Ij_prob * bin_size);
    }
    k_sanity.write("k_sanity", parameter);

    let mut Ii_given_prev_Ii = vec![vec![0.0; len]; len];

    let aggregated_func: Vec<_> = (0..len)
        .into_par_iter()
        .map(
            |Ij_t0|
            {
                (0..idx_s)
                    .map(
                        |other_k: usize|
                        {
                            let mut agg = vec![0.0; len];
                            for (Ij_t1, Ij_t0_prob) in Ij_given_prev_Ij[Ij_t0].iter().enumerate(){
                                let density = Ii_given_Ij_and_kij[Ij_t1].func[other_k].as_slice();
                                let outer_prob = Ij_t0_prob * bin_size2;
                                agg.iter_mut()
                                    .zip(density)
                                    .for_each(
                                        |(a,b)|
                                        {
                                            *a += b * outer_prob
                                        }
                                    );
                            }
                            agg
                        }
                    ).collect_vec()
            }
        ).collect();
    let aggregated_delta_right = (0..len)
        .map(
            |Ij_t0|
            {

                let mut agg = vec![0.0; len];
                for (Ij_t1, Ij_t0_prob) in Ij_given_prev_Ij[Ij_t0].iter().enumerate(){
                    let density = Ii_given_Ij_and_kij[Ij_t1].delta_right.as_slice();
                    let outer_prob = Ij_t0_prob * bin_size2;
                    agg.iter_mut()
                        .zip(density)
                        .for_each(
                            |(a,b)|
                            {
                                *a += b * outer_prob
                            }
                        );
                }
                agg

            }
        ).collect_vec();

    let use_aggregation = |res_Ii_vec: &mut [f64], aggregate: &[f64], prob: f64|
    {
        res_Ii_vec
            .iter_mut()
            .zip(aggregate)
            .for_each(
                |(a, b)|
                {
                    *a += b * prob
                }
            );
    };
    

    let mut Ii_given_prev_Ii_for_helper = |kij_tm1: usize, Ij_t0: usize, prob: f64|
    {
        let kIj_t0 = Ij_t0 + kij_tm1;
        let end = kIj_t0.min(len);
        let m_smaller_range = 0..end;
        for m in m_smaller_range{
            let prev_Ii = m;
            let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();
            let other_k = (kIj_t0 - prev_Ii).min(idx_s);
            if other_k < idx_s {
                let aggregate = aggregated_func[Ij_t0][other_k].as_slice();
                use_aggregation(
                    res_Ii_vec,
                    aggregate,
                    prob
                );
            } else {
                // delta right
                let aggregate = aggregated_delta_right[Ij_t0].as_slice();
                use_aggregation(
                    res_Ii_vec,
                    aggregate,
                    prob
                );
            }
            
        }
        if len > end {
            let kIj_range = end..len;
            let remaining = kIj_range.len();
            let prob = prob * remaining as f64;
            let prev_Ii = kIj_t0;
            let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();
            let other_k = (kIj_t0 - prev_Ii).min(idx_s);

            let aggregate = aggregated_func[Ij_t0][other_k].as_slice();
            use_aggregation(
                res_Ii_vec,
                aggregate,
                prob
            );
        }
        
    };

    for (Ij_t0, Ij_t0_dens) in prob_Ij.iter().enumerate().progress()
    {
        let Ij_t0_prob = Ij_t0_dens * bin_size;
        let k_tm1_density = &k_tm1_given_Ij_t[Ij_t0];
        for (kij_tm1, k_tm1_dens) in k_tm1_density.func.iter().enumerate()
        {
            let prob = k_tm1_dens * bin_size * Ij_t0_prob;
            Ii_given_prev_Ii_for_helper(kij_tm1, Ij_t0, prob);
        }

        let delta_left_prob = Ij_t0_prob * k_tm1_density.delta.0;
        Ii_given_prev_Ii_for_helper(0, Ij_t0, delta_left_prob);

        let delta_right_prob = Ij_t0_prob * k_tm1_density.delta.1;
        Ii_given_prev_Ii_for_helper(idx_s, Ij_t0, delta_right_prob);  
    }
    
    normalize_prob_matrix(&mut Ii_given_prev_Ii, bin_size);

    let mut sanity_check_final = vec![0.0; len];

    for (slice, prob_density) in Ii_given_prev_Ii.iter().zip(Ii_probability_density.iter())
    {
        let prob = prob_density * bin_size;
        sanity_check_final
            .iter_mut()
            .zip(slice)
            .for_each(
                |(a,b)|
                {
                    *a += b * prob;
                }
            );
    }
    write_I(&sanity_check_final, bin_size, "Tsanity_check_final_non_normalized.dat");
    normalize_vec(&mut sanity_check_final, bin_size);
    write_I(&sanity_check_final, bin_size, "Tsanity_check_final.dat");
    (Ii_given_prev_Ii, Ii_probability_density)
     
}

fn master_ansatz_k(
    a: &[f64], 
    s: f64, 
    threshold: f64
)-> (Parameter, ProbabilityDensity)
{
    let len = a.len();
    let bin_size = ((len) as f64).recip();
    let index_s = (s / bin_size).ceil() as usize;

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
    let total_jump_recip = total_jump_prob.recip();
    p_am.iter_mut()
        .for_each(
            |val| *val *= total_jump_recip
        );

    let mut delta_left = 0.2;
    let mut delta_right = 0.2;

    let guess_height = (1.0-delta_left-delta_right)/s;
    let mut k_guess = vec![guess_height; index_s];

    let mut k_result = vec![0.0; k_guess.len()];

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
                        let index_plus_jump = index + jump_index;
                        if index_plus_jump < len {
                            delta_left_input += amount;
                            continue;
                        }
                        let resulting_index = index_plus_jump - len;
                        if let Some(val) = k_result.get_mut(resulting_index){
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
            let amount = delta_left * prob;
            
            if jump_index < len {
                delta_left_input += amount;
                continue;
            }
            let resulting_index = jump_index - len;
            if let Some(val) = k_result.get_mut(resulting_index){
                *val += amount;
            } else {
                delta_right_input += amount;
            }
        }

        // right delta
        for (jump_index, prob) in p_am.iter().enumerate()
        {
            let amount = delta_right * prob;
            let k_plus_jump = index_s + jump_index;
            if k_plus_jump < len {
                delta_left_input += amount;
                continue;
            }
            let resulting_index = k_plus_jump - len;
            if let Some(val) = k_result.get_mut(resulting_index){
                *val += amount;
            } else {
                delta_right_input += amount;
            }
        }

        delta_left_input *= bin_size;
        delta_right_input *= bin_size;

        let mut difference: f64 = k_guess.iter()
            .zip(k_result.iter())
            .map(|(a,b)| (a-b).abs())
            .sum();
        difference *= bin_size;
        difference += (delta_left - delta_left_input).abs()
            + (delta_right - delta_right_input).abs();
        
        delta_left = delta_left_input;
        delta_right = delta_right_input;

        if difference <= threshold { 
            let mut density = ProbabilityDensity{
                func: k_result,
                delta: (delta_left, delta_right)
            };
            density.normalize(bin_size);
            let parameter = Parameter{
                bin_size,
                s,
                len_of_1: len,
                index_s,
                len_of_k_func: density.func.len()
            };
            return (parameter, density)
        }
        std::mem::swap(&mut k_guess, &mut k_result);
    }
    
}


#[allow(non_snake_case)]
fn calc_I(
    parameter: &Parameter, 
    kij_density: &ProbabilityDensity,
    a_ij: &[f64]
) -> Vec<f64>
{

    let p_ka = (0..(kij_density.func.len() + parameter.len_of_1))
        .map(
            |x|
            {
                let mut integral = 0.0;
                let start = if x < parameter.len_of_1 {
                    0
                } else {
                    x - (parameter.len_of_1 - 1)
                };
                let end = if x >= kij_density.func.len(){
                    kij_density.func.len() - 1
                } else {
                    x
                };
                for j in start..=end{
                    integral += kij_density.func[j] * a_ij[x-j];
                }
                integral *= parameter.bin_size;

                if start == 0{
                    integral += kij_density.delta.0 * a_ij[x]; 
                }
                if x >= parameter.index_s && x-parameter.index_s < a_ij.len() {
                    integral += kij_density.delta.1 * a_ij[x-parameter.index_s];
                }

                integral
            }
        ).collect_vec();
    /// I probably want to normalize p_ka here
    let p_ka_total: f64 = p_ka.iter()
        .map(
            |val|
            {
                val * parameter.bin_size
            }
        ).sum();
    println!("pka total: {p_ka_total}");

    let name = format!("s{}p_ka_1.dat", parameter.s);
    let mut buf = create_buf(name);
    for (index, val) in p_ka.iter().enumerate()
    {
        let x = index as f64 * parameter.bin_size;
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
                sum * parameter.bin_size
            }
        ).collect_vec();

    let name = format!("s{}prob_1.dat", parameter.s);
    let mut buf = create_buf(name);

    let error = prob[0];
    let error_correction_factor = error.recip();
    prob.iter_mut()
        .for_each(|val| *val *= error_correction_factor);

    for (index, val) in prob.iter().enumerate()
    {
        let x = index as f64 * parameter.bin_size;
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
                let x = idx as f64 * parameter.bin_size;
                *val = 1.0 - (1.0 - x) * *val;
            }
        );

    let name = format!("s{}cum_prob_1.dat", parameter.s);

    let mut buf = create_buf(name);
    for (index, val) in prob.iter().enumerate()
    {
        let x = index as f64 * parameter.bin_size;
        writeln!(
            buf,
            "{} {}",
            x,
            val
        ).unwrap();
    }

    //let mut derivative = sampling::glue::derivative::derivative(&prob[..pk.len_of_1]);
    let derivative_left = sampling::glue::derivative::derivative(&prob[..parameter.index_s]);
    let derivative_right = sampling::glue::derivative::derivative(&prob[parameter.index_s..parameter.len_of_1]);
    let mut derivative = derivative_left;
    derivative.extend_from_slice(&derivative_right);

    let len = parameter.len_of_1 as f64;
    derivative.iter_mut()
        .for_each(
            |val|
            {
                *val *= len;
            }
        );

    let name = format!("s{}derivative_1.dat", parameter.s);
    let mut buf = create_buf(name);
    for (index, val) in derivative.iter().enumerate()
    {
        let x = index as f64 * parameter.bin_size;
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

