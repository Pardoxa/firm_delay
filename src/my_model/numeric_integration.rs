use std::{io::BufReader, num::NonZeroUsize, sync::Mutex, time::Duration};
use indicatif::{ParallelProgressIterator, ProgressIterator};
use itertools::*;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use derivative::Derivative;
use std::io::Write;
use crate::misc::*;


#[derive(Debug, Clone, Derivative, Serialize, Deserialize, PartialEq)]
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
#[derive(Serialize, Deserialize)]
pub struct SaveState{
    input: ModelInput,
    pkij_given_pre_Ij: Vec<ProbabilityDensity>,
    pk_N2: ProbabilityDensity,
    Ij: Vec<f64>,
    Ij_given_pre_Ij: Vec<Vec<f64>>,
    len_of_1: usize,
    idx_s: usize,
    bin_size: f64
}

impl SaveState{
    pub fn try_read(name: &str) -> Option<Self>
    {
        let reader = fs_err::File::open(name).ok()?;
        let buf_reader = BufReader::new(reader);

        bincode::deserialize_from(buf_reader).ok()
    }
}

#[allow(non_snake_case)]
pub fn line_test(input: ModelInput)
{
    let save_name = "test.save";

    let mut save_state_opt = SaveState::try_read(save_name);

    if let Some(save_state) = save_state_opt.as_ref(){
        if !save_state.input.eq(&input){
            save_state_opt = None;
            println!("SAVE STATE INPUT IS MISMATCHED!");
            std::thread::sleep(Duration::from_secs(5));
        }
    }

    let mut save_state = match save_state_opt{
        None => {
            // here I count: N=0 is leaf, N=1 is first node after etc
            let production_N0 = vec![1.0; input.precision.get()];

            let counter = 0;
            let pk_N1 = master_ansatz_k(
                &production_N0, 
                input.s, 
                1e-8
            );
            let stub = format!("_PK{counter}");
            pk_N1.write_files(&stub);

            let production_N1 = calc_I(&pk_N1, &production_N0, counter); 
            write_I(&production_N1, pk_N1.bin_size, "I_2_bla1.dat");




            let P_I_N1_given_prior_I_N1 = master_ansatz_i_test(&pk_N1, &production_N1);

            let (pk_N2_given_I_N1, pk_N2) = calk_k_master_test(
                &pk_N1,
                &P_I_N1_given_prior_I_N1,
                &production_N1,
                1e-6
            );



            pk_N2.write("pk_N2_res", pk_N1.bin_size, input.s);

            let save_state = SaveState{
                input,
                pkij_given_pre_Ij: pk_N2_given_I_N1,
                Ij: production_N1,
                Ij_given_pre_Ij: P_I_N1_given_prior_I_N1,
                len_of_1: pk_N1.len_of_1,
                bin_size: pk_N1.bin_size,
                idx_s: pk_N1.index_s,
                pk_N2
            };
            let buf = create_buf(save_name);
            bincode::serialize_into(buf, &save_state)
                .expect("Serialization Issue");
            println!("SAVED");
            save_state
        },
        Some(save_state) => {
            save_state
        }
    };


    

    for i in 3..5{

        let calc_result = calc_next_test(
            &save_state.pkij_given_pre_Ij, 
            &save_state.Ij,
            &save_state.Ij_given_pre_Ij,
            save_state.len_of_1,
            save_state.idx_s,
            save_state.bin_size,
            save_state.input.s
        );

        let name_I = format!("I_{i}_bla1.dat");
        write_I(&calc_result.I2_density, save_state.bin_size, &name_I);
        panic!("TEST PANIC");
        let pk = Pk{
            bin_size: save_state.bin_size,
            k_density: save_state.pk_N2,
            s: save_state.input.s,
            len_of_1: save_state.len_of_1,
            index_s: save_state.idx_s
        };
    
        let (pk_N3_given_I_N2, pk_N3) = calk_k_master_test(
            &pk,
            &calc_result.I2_given_prev_I2,
            &calc_result.I2_density,
            1e-6
        );

        let stub = format!("pk_N{i}_test_res");
        pk_N3.write(&stub, save_state.bin_size, save_state.input.s);

        save_state = SaveState{
            input: save_state.input,
            pkij_given_pre_Ij: pk_N3_given_I_N2,
            Ij: calc_result.I2_density,
            Ij_given_pre_Ij: calc_result.I2_given_prev_I2,
            len_of_1: save_state.len_of_1,
            bin_size: save_state.bin_size,
            idx_s: save_state.idx_s,
            pk_N2: pk_N3
        };

        let save_name = format!("SAVE{i}.save");
        let buf = create_buf(save_name);
        bincode::serialize_into(buf, &save_state)
            .expect("Serialization Issue");
        println!("SAVED");
    }

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

pub struct Pk{
    k_density: ProbabilityDensity,
    bin_size: f64,
    s: f64,
    len_of_1: usize,
    index_s: usize
}

impl Pk{
    pub fn write_files(&self, stub: &str)
    {
        self.k_density.write(stub, self.bin_size, self.s)
    }

    pub fn len(&self) -> usize 
    {
        self.k_density.func.len()
    }
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

    pub fn write(&self, stub: &str, bin_size: f64, s: f64)
    {
        let name_func = format!("{stub}_func.dat");
        let mut buf = create_buf_with_command_and_version(name_func);

        for (i, val) in self.func.iter().enumerate() {
            let x = i as f64 * bin_size;
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
            s,
            self.delta.1
        ).unwrap();
    }
}

// matrix needs to be square matrix
fn reverse_prob_matrix(
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
pub struct CalcResult{
    I2_density: Vec<f64>,
    I2_given_prev_I2: Vec<Vec<f64>>
}

#[allow(non_snake_case)]
fn calc_next_test(
    pk_N2_given_pre_I_N1: &[ProbabilityDensity],
    I_N1: &[f64],
    I1_given_pre_I1: &[Vec<f64>],
    len_of_1: usize,
    idx_s: usize,
    bin_size: f64,
    s: f64
) -> CalcResult
{

    let pre_I1_given_I1 = reverse_prob_matrix(
        I1_given_pre_I1, 
        I_N1, 
        bin_size
    );

    // checking if it is correct
    let mut pre_I1_sanity_check = vec![0.0; I_N1.len()];
    for (line, prob_density) in I1_given_pre_I1.iter().zip(I_N1.iter())
    {
        let prob = prob_density * bin_size;
        pre_I1_sanity_check.iter_mut()
            .zip(line)
            .for_each(
                |(pre, line_entry)| *pre += line_entry * prob
            );
    }

    let mut buf = create_buf_with_command_and_version("Sanity_check.dat");

    for (i, sanity_val) in pre_I1_sanity_check.iter().enumerate(){
        let x = i as f64 * bin_size;
        writeln!(
            buf, 
            "{x} {sanity_val}"
        ).unwrap();
    }

    // Irgendwo ist noch ein off by 1 error der zu einem fehler bei index_s führt, suche ich sobald der Rest läuft
    let mut probability_I2 = vec![0.0; I_N1.len()];
    let mut I1_given_I2  = vec![vec![0.0; I_N1.len()]; I_N1.len()]; // this is not previous I2 but this I2! I think this is my current error
    let mut pk_given_preI2 = (0..I_N1.len())
        .map(
            |_| pk_N2_given_pre_I_N1[0].create_zeroed()
        ).collect_vec();

    let recip_len1 = (len_of_1 as f64).recip();

    for (idx_pre_I1, I1_given_preI1_prob_vec) in I1_given_pre_I1.iter().enumerate().progress(){
        // previous production of node below was `idx_pre_I1`
        let prob_level1 = I_N1[idx_pre_I1] * bin_size;
        for (this_I1, prob_this_I1) in I1_given_preI1_prob_vec.iter().enumerate(){
            let prob_level2 = prob_level1 * prob_this_I1;

            let k_density = &pk_N2_given_pre_I_N1[idx_pre_I1];
            for (k_idx, k_prob) in k_density.func.iter().enumerate(){
                let level_3_prob = prob_level2 * k_prob;
                let level_4_prob = level_3_prob * recip_len1; // this is the relevant increment, maybe I need to multiply with binsize or so
                let Ik = this_I1 + k_idx;
                for m in 0..len_of_1{
                    let this_I2 = Ik.min(m); // Optimization possible
                    probability_I2[this_I2] += level_4_prob;
                    I1_given_I2[this_I2][this_I1] += level_4_prob;

                    let inc_density = &mut pk_given_preI2[this_I2];
                    if m > Ik {
                        // delta left 
                        inc_density.delta.0 += level_4_prob;
                        continue;
                    }
                    let this_idx = Ik - m;
                    if this_idx > idx_s {
                        // delta right
                        inc_density.delta.1 += level_4_prob;
                    } else {
                        // func
                        inc_density.func[this_idx] += level_4_prob;
                    }
                }
            }
            // delta left 
            let delta_left = k_density.delta.0;
            let level_3_prob = prob_level2 * delta_left / bin_size; // Check if bin_size is correct here
            let level_4_prob = level_3_prob * recip_len1;
            let Ik = this_I1; // k_idx is 0
            for m in 0..len_of_1{
                let this_I2 = Ik.min(m);
                probability_I2[this_I2] += level_4_prob;
                I1_given_I2[this_I2][this_I1] += level_4_prob;

                let inc_density = &mut pk_given_preI2[this_I2];
                if m > Ik {
                    // delta left 
                    inc_density.delta.0 += level_4_prob;
                    continue;
                }
                let this_idx = Ik - m;
                if this_idx > idx_s {
                    // delta right
                    inc_density.delta.1 += level_4_prob;
                } else {
                    // func
                    inc_density.func[this_idx] += level_4_prob;
                }
            }

            // delta right 
            let delta_right = k_density.delta.1;
            let level_3_prob = prob_level2 * delta_right / bin_size; // Check if bin_size is correct here
            let level_4_prob = level_3_prob * recip_len1;
            let Ik = this_I1 + idx_s;
            for m in 0..len_of_1{
                let this_I2 = Ik.min(m);
                probability_I2[this_I2] += level_4_prob;
                I1_given_I2[this_I2][this_I1] += level_4_prob;

                let inc_density = &mut pk_given_preI2[this_I2];
                if m > Ik {
                    // delta left 
                    inc_density.delta.0 += level_4_prob;
                    continue;
                }
                let this_idx = Ik - m;
                if this_idx > idx_s {
                    // delta right
                    inc_density.delta.1 += level_4_prob;
                } else {
                    // func
                    inc_density.func[this_idx] += level_4_prob;
                }
            }

        }
    }



    // normalization of I 
    for I1_line in I1_given_I2.iter_mut(){
        let sum: f64 = I1_line.iter().sum();
        let factor = sum.recip();
        I1_line.iter_mut()
            .for_each(
                |val| *val *= factor
            );
    }

    let mut I1_given_I2_summary = vec![0.0; I1_given_I2[0].len()];
    for (I1_line, prob) in I1_given_I2.iter().zip(probability_I2.iter())
    {
        for (into, from) in I1_given_I2_summary.iter_mut().zip(I1_line.iter())
        {
            *into += from * prob;
        }
    }

    let name = "I1_I2_sum.dat";
    let mut buf = create_buf_with_command_and_version(name);
    for (i, val) in I1_given_I2_summary.iter().enumerate(){
        let x = i as f64 * bin_size;
        writeln!(
            buf,
            "{x} {val}"
        ).unwrap();
    }

    // Not sure if this is correct…
    let mut I1_given_pre_I2_maybe_incorrect_needs_checking = I1_given_I2.iter()
        .map(|line| vec![0.0; line.len()])
        .collect_vec();

    for (i1_given_pre_i2_line, i1_given_i2_line) in I1_given_pre_I2_maybe_incorrect_needs_checking.iter_mut().zip(I1_given_I2.iter())
    {
        for (&i1_given_i2_prob, i1_given_pre_i1_line) in i1_given_i2_line.iter().zip(I1_given_pre_I1){
            // now I need to calculate the next i1, I think
            for (next_I1_prob, i1_given_pre_i2_entry) in i1_given_pre_i1_line.iter().zip(i1_given_pre_i2_line.iter_mut())
            {
                let prob = i1_given_i2_prob * next_I1_prob;
                *i1_given_pre_i2_entry += prob;
            }
        }
    }
    // normalization
    normalize_prob_matrix(&mut I1_given_pre_I2_maybe_incorrect_needs_checking, bin_size);

    let mut sanity_2 = vec![0.0; I1_given_I2.len()];

    for (i1_given_pre_i2_line, i2_prob) in I1_given_pre_I2_maybe_incorrect_needs_checking.iter().zip(probability_I2.iter())
    {
        let factor = i2_prob * bin_size;
        sanity_2.iter_mut()
            .zip(i1_given_pre_i2_line)
            .for_each(
                |(res, val)|
                *res += val * factor
            );
    }

    let mut buf = create_buf_with_command_and_version("sanity_2.dat");
    for (i, val) in sanity_2.iter().enumerate()
    {
        let x = i as f64 * bin_size;
        writeln!(
            buf,
            "{x} {val}"
        ).unwrap();
    }

    let name = "N3_I_test.dat";
    let mut buf = create_buf_with_command_and_version(name);
    for (i, prob) in probability_I2.iter().enumerate()
    {
        let x = i as f64 * bin_size;
        writeln!(
            buf,
            "{x} {prob}"
        ).unwrap();
    }

    // normalization of pk
    for density in pk_given_preI2.iter_mut(){
        density.delta.0 *= bin_size;
        density.delta.1 *= bin_size;
        density.normalize(bin_size);
    }

    let mut pk_res = pk_N2_given_pre_I_N1[0].create_zeroed();
    for (prob_I2, k_density) in probability_I2.iter().zip(pk_given_preI2.iter())
    {
        for (res, from) in pk_res.func.iter_mut().zip(k_density.func.iter()){
            *res += from * prob_I2;
        }
        pk_res.delta.0 += prob_I2 * k_density.delta.0;
        pk_res.delta.1 += prob_I2 * k_density.delta.1;
    }

    pk_res.delta.0 *= bin_size;
    pk_res.delta.1 *= bin_size;
    pk_res.func
        .iter_mut()
        .for_each(
            |val| *val *= bin_size
        );

    pk_res.write("N3_pk_test", bin_size, s);


    // currently the normalization is incorrect!
    // There also seems to be another mistake
    let mut I2_given_prev_I2 = vec![vec![0.0; len_of_1]; len_of_1];

    // This calculates the quantity for which I am doing all this BS
    for (prev_I2, (line, prev_I2_prob)) in I2_given_prev_I2.iter_mut().zip(probability_I2.iter()).enumerate().progress(){
        let k_density = &pk_given_preI2[prev_I2];
        let I1_line = I1_given_pre_I2_maybe_incorrect_needs_checking[prev_I2].as_slice();
        let level_1_prob = prev_I2_prob;
        for (k_idx, k_prob) in k_density.func.iter().enumerate(){
            let level_2_prob = level_1_prob * k_prob * bin_size;
            for (idx_I1, I1_prob) in I1_line.iter().enumerate(){
                let level_3_prob = level_2_prob * I1_prob;
                let level_4_prob = level_3_prob * recip_len1;
                let Ik = idx_I1 + k_idx;
                for m in 0..len_of_1{
                    let I2 = m.min(Ik);
                    line[I2] += level_4_prob;
                }
            }
        }

        // delta left
        let level_2_prob = level_1_prob * k_density.delta.0;
        for (idx_I1, I1_prob) in I1_line.iter().enumerate(){
            let level_3_prob = level_2_prob * I1_prob;
            let level_4_prob = level_3_prob * recip_len1;
            for m in 0..len_of_1{
                let Ik = idx_I1; // k=0
                let I2 = m.min(Ik);
                line[I2] += level_4_prob;
            }
        }

        // delta right
        let level_2_prob = level_1_prob * k_density.delta.1;
        for (idx_I1, I1_prob) in I1_line.iter().enumerate(){
            let level_3_prob = level_2_prob * I1_prob;
            let level_4_prob = level_3_prob * recip_len1;
            for m in 0..len_of_1{
                let Ik = idx_I1 + idx_s;
                let I2 = m.min(Ik);
                line[I2] += level_4_prob;
            }
        }
    }

    // normalization
    normalize_prob_matrix(&mut I2_given_prev_I2, bin_size);

    // now to check if it works correctly
    let mut check_I2 = vec![0.0; len_of_1];
    for (line, prob_density) in I2_given_prev_I2.iter().zip(probability_I2.iter())
    {
        let prob = prob_density * bin_size;
        check_I2.iter_mut().zip(line.iter())
            .for_each(
                |(res, from)|
                *res += from * prob
            );
    }

    let mut buf = create_buf_with_command_and_version("CheckI2.dat");
    for (i, I2) in check_I2.iter().enumerate(){
        let x = i as f64 * bin_size;
        writeln!(
            buf,
            "{x} {I2}"
        ).unwrap();
    }

    let mut i2_sum: f64 = check_I2.iter().sum();
    i2_sum *= bin_size;
    println!("I2 sum: {i2_sum}");




    let mut I2_given_prev_I1_test = vec![vec![0.0; len_of_1]; len_of_1];

    println!("HERE");
    for (Ij_t0_idx, k_density) in pk_N2_given_pre_I_N1.iter().enumerate().progress(){
        let Ij_t1_density = I1_given_pre_I1[Ij_t0_idx].as_slice();
        let I2_given_pre_I1_line = I2_given_prev_I1_test[Ij_t0_idx].as_mut_slice();
        for (k_idx, k_prob) in k_density.func.iter().enumerate(){
            let level_1_density = k_prob * recip_len1;
            for (Ij_t1_idx, Ij_t1_prob) in Ij_t1_density.iter().enumerate(){
                let level_2_density = level_1_density * Ij_t1_prob;
                let IjK = Ij_t1_idx + k_idx;
                let end = len_of_1.min(IjK);
                for m in 0..end{ 
                    let Ii_t1 = IjK.min(m);
                    I2_given_pre_I1_line[Ii_t1] += level_2_density;
                }
                let remaining = len_of_1 - end;
                if remaining > 0{
                    I2_given_pre_I1_line[IjK] += level_2_density * remaining as f64;
                }
            }
        }

        // delta left
        let level_1_density = k_density.delta.0 * recip_len1; // I think this is correct, maybe look at bin_size again if the result is strange
        for (Ij_t1_idx, Ij_t1_prob) in Ij_t1_density.iter().enumerate(){
            let level_2_density = level_1_density * Ij_t1_prob / bin_size;
            let IjK = Ij_t1_idx; // k = 0
            for m in 0..len_of_1{ // can be optimized
                let Ii_t1 = IjK.min(m);
                I2_given_pre_I1_line[Ii_t1] += level_2_density;
            }
        }

        // delta right
        let level_1_density = k_density.delta.1 * recip_len1;
        for (Ij_t1_idx, Ij_t1_prob) in Ij_t1_density.iter().enumerate(){
            let level_2_density = level_1_density * Ij_t1_prob / bin_size;
            let IjK = Ij_t1_idx + idx_s; // k = s
            for m in 0..len_of_1{ // can be optimized
                let Ii_t1 = IjK.min(m);
                I2_given_pre_I1_line[Ii_t1] += level_2_density;
            }
        }
    }
    normalize_prob_matrix(&mut I2_given_prev_I1_test, bin_size);

    let mut sanity_5 = vec![0.0; len_of_1];
    for (density, I1_prob) in I2_given_prev_I1_test.iter().zip(I_N1){
        sanity_5
            .iter_mut()
            .zip(density)
            .for_each(
                |(r,v)|
                *r += v * I1_prob * bin_size
            );
    }

    let mut buf = create_buf_with_command_and_version("sanity_5.dat");
    for (i, val) in sanity_5.iter().enumerate(){
        let x = i as f64 * bin_size;
        writeln!(
            buf,
            "{x} {val}"
        ).unwrap();
    }
    let I2_given_I1 = reverse_prob_matrix(&I1_given_I2, &probability_I2, bin_size);
    let mut I2_given_I2 = vec![vec![0.0; len_of_1]; len_of_1];

    for (prev_I1, (density_this_I2, prev_I1_density)) in I2_given_prev_I1_test.iter().zip(I_N1.iter()).enumerate(){

        let density_prev_I2 = I2_given_I1[prev_I1].as_slice();
        for (prev_I2, density_prev_2) in density_prev_I2.iter().enumerate(){
            let level_1_density = prev_I1_density * density_prev_2;
            let I2_given_I2_line = I2_given_I2[prev_I2].as_mut_slice();
            for (this_I2_density, entry) in density_this_I2.iter().zip(I2_given_I2_line){
                *entry += level_1_density * this_I2_density;
            }
        }
    }


    normalize_prob_matrix(&mut I2_given_I2, bin_size);

    let mut sanity_4 = vec![0.0; len_of_1];
    for (density, I2_prob) in I2_given_I2.iter().zip(probability_I2.iter()){
        sanity_4.iter_mut()
            .zip(density.iter())
            .for_each(
                |(r,v)|
                *r += v * I2_prob * bin_size
            );
    }

    let mut buf = create_buf_with_command_and_version("sanity_4.dat");
    for (i, val) in sanity_4.iter().enumerate(){
        let x = i as f64 * bin_size;
        writeln!(
            buf,
            "{x} {val}"
        ).unwrap();
    }

    // normalize 
    let mut sum: f64 = probability_I2.iter().sum();
    sum *= bin_size;
    let norm_factor = sum.recip();
    probability_I2
        .iter_mut()
        .for_each(
            |val|
            {
                *val *= norm_factor;
            }
        );

    CalcResult{
        I2_density: probability_I2,
        I2_given_prev_I2
    }
}

#[allow(non_snake_case)]
fn calk_k_master_test(
    prior_pk: &Pk,
    input_P_I_given_prior_I: &[Vec<f64>],
    prior_I_for_normalization: &[f64],
    threshold: f64
) -> (Vec<ProbabilityDensity>, ProbabilityDensity)
{
    let mut current_estimate_given_prior_I = (0..input_P_I_given_prior_I.len())
        .map(|_| ProbabilityDensity::new(prior_pk.len(), prior_pk.bin_size))
        .collect_vec();

    let mut next_estimate_given_prior_I = (0..input_P_I_given_prior_I.len())
        .map(|_| ProbabilityDensity::new_zeroed(prior_pk.len()))
        .collect_vec();

    let idx_s = prior_pk.index_s;
    let len_of_1 = prior_pk.len_of_1;
    let bin_size = prior_pk.bin_size;


    let mut resulting_density = ProbabilityDensity::new_zeroed(prior_pk.len());
    let m_factor = (len_of_1 as f64).recip();

    let for_helper = |kI: usize, update_k_vec: &mut ProbabilityDensity, probability_increment: f64|
    {
        // first border exlusive
        let m_range_delta_left = kI.min(len_of_1)..len_of_1;
        let weight = if m_range_delta_left.contains(&0){
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
        let w_sum = weight + m_range_delta_right.len() + k_range.len();
        assert_eq!(
            w_sum,
            len_of_1
        );
        update_k_vec.func[k_range]
            .iter_mut()
            .for_each(
                |val| *val += probability_increment
            );
    };
    loop {
        /// Maybe there is an off by one somewhere here. Maybe the issue is instead that the density of k is 1 to long
        for (prior_I_idx, current_Ij_distribution) in input_P_I_given_prior_I.iter().enumerate().progress(){
            let current_k = &current_estimate_given_prior_I[prior_I_idx];
            let prior_I_prob = prior_I_for_normalization[prior_I_idx];
            let m_factor_times_prior_I_prob = prior_I_prob * m_factor;
            for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                let update_k_vec = &mut next_estimate_given_prior_I[Ij_idx];
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
                let update_k_vec = &mut next_estimate_given_prior_I[Ij_idx];
                let probability_increment = left_increment * Ij_prob;
                
                for_helper(kI, update_k_vec, probability_increment);
            }
    
            // right
            let right_increment = m_factor_times_prior_I_prob * current_k.delta.1 / bin_size;
            for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                let kI: usize = Ij_idx + idx_s;
                let probability_increment = right_increment * Ij_prob;
                let update_k_vec = &mut next_estimate_given_prior_I[Ij_idx];
                
                for_helper(kI, update_k_vec, probability_increment);
            }

        }
    
        next_estimate_given_prior_I.iter_mut()
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
            let delta_norm = norm * bin_size;
            resulting_density.delta.0 += estimate.delta.0 * delta_norm;
            resulting_density.delta.1 += estimate.delta.1 * delta_norm;
        }

        let estimate_diff: f64 = current_estimate_given_prior_I.iter()
            .zip(next_estimate_given_prior_I.iter())
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
        std::mem::swap(&mut next_estimate_given_prior_I, &mut current_estimate_given_prior_I);
        if estimate_diff <= threshold {
            break;
        }
        next_estimate_given_prior_I
            .iter_mut()
            .for_each(ProbabilityDensity::make_zero);
        resulting_density.make_zero();
    }
    
    (current_estimate_given_prior_I, resulting_density)

}

#[derive(Clone)]
struct Ii_given_k {
    delta_left: Vec<f64>,
    delta_right: Vec<f64>,
    func: Vec<Vec<f64>>
}

/// For now only for N-2
/// this assumes that J (jump prob) is not dependent on k
#[allow(non_snake_case)]
fn master_ansatz_i_test(
    pk: &Pk,
    prob_Ii: &[f64]
) -> Vec<Vec<f64>>
{
    let len = prob_Ii.len();
    let idx_s = pk.index_s;
    let bin_size = pk.bin_size;

    let mut Ii_given_k = Ii_given_k{
        delta_left: vec![0.0; len],
        delta_right: vec![0.0; len],
        func: vec![vec![0.0; len]; pk.k_density.func.len()]
    };

    let len_recip = (len as f64).recip();
    let len_recip2 = len_recip * len_recip;
    for (k, Ii) in Ii_given_k.func.iter_mut().enumerate()
    {
        // Ij is independent of k here, thank god
        // (its uniform)
        for Ij in 0..len {
            let kIj = k + Ij;
            let right = kIj.min(len);
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
        }
    }

    // delta left 
    for Ij in 0..len {
        let kIj = Ij;
        let right = kIj.min(len);
        Ii_given_k.delta_left[..right]
            .iter_mut()
            .for_each(
                |entry|
                {
                    *entry += len_recip2;
                }
            );
        let remaining = len - right;
        if remaining > 0{
            Ii_given_k.delta_left[kIj] += len_recip2 * remaining as f64;
        }
    }

    // delta right 
    for Ij in 0..len {
        let kIj = Ij + idx_s;
        let right = kIj.min(len);
        Ii_given_k.delta_right[..right]
            .iter_mut()
            .for_each(
                |entry|
                {
                    *entry += len_recip2;
                }
            );
        let remaining = len - right;
        if remaining > 0{
            Ii_given_k.delta_right[kIj] += len_recip2 * remaining as f64;
        }
    }

    normalize_prob_matrix(&mut Ii_given_k.func, bin_size);
    normalize_vec(&mut Ii_given_k.delta_left, bin_size);
    normalize_vec(&mut Ii_given_k.delta_right, bin_size);

    let mut  sanity_check = vec![0.0; len];

    for (prob, vec) in pk.k_density.func.iter().zip(Ii_given_k.func.iter())
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

    let prob = pk.k_density.delta.0;
    sanity_check.iter_mut()
        .zip(Ii_given_k.delta_left.iter())
        .for_each(
            |(a,b)|
            {
                *a += b * prob;
            }
        );
    let prob = pk.k_density.delta.1;
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

    let mut kij_given_prev_k_and_this_Ij = vec![vec![pk.k_density.create_zeroed(); len]; pk.k_density.func.len()];
    let mut kij_given_prev_k_delta_left_and_this_Ij = vec![pk.k_density.create_zeroed(); len];
    let mut kij_given_prev_k_delta_right_and_this_Ij = vec![pk.k_density.create_zeroed(); len];


    // func
    for (prev_k, kij_given_prev_k_Ij_unknown) in kij_given_prev_k_and_this_Ij.iter_mut().enumerate(){
        for (this_Ij, resulting_k_density) in kij_given_prev_k_Ij_unknown.iter_mut().enumerate()
        {
            let kIj = prev_k + this_Ij;
            if kIj+1 < len {
                let range_delta_left = kIj+1..len;
                let length = range_delta_left.len();
                resulting_k_density.delta.0 += len_recip * length as f64 * bin_size;
            }
            let start = if kIj >= idx_s{
                let end = kIj - idx_s;
                //let range_delta_right = 0..=end;
                let length = end + 1;
                resulting_k_density.delta.1 += len_recip * length as f64 * bin_size;
                end + 1
            } else {
                0
            };
            let end = kIj.min(len - 1);
            let range = start..=end;
            for mi in range{ // Optimizable
                let k = kIj - mi;
                resulting_k_density.func[k] += len_recip;
            }
        }
    }

    // delta left
    for (this_Ij, resulting_k_density) in kij_given_prev_k_delta_left_and_this_Ij.iter_mut().enumerate()
    {
        let kIj = this_Ij; // prev_k is 0
        if kIj+1 < len {
            let range_delta_left = kIj+1..len;
            let length = range_delta_left.len();
            resulting_k_density.delta.0 += len_recip * length as f64 * bin_size;
        }
        let start = if kIj >= idx_s{
            let end = kIj - idx_s;
            //let range_delta_right = 0..=end;
            let length = end + 1;
            resulting_k_density.delta.1 += len_recip * length as f64 * bin_size;
            end + 1
        } else {
            0
        };
        let end = kIj.min(len - 1);
        let range = start..=end;
        for mi in range{ // Optimizable
            let k = kIj - mi;
            resulting_k_density.func[k] += len_recip;
        }
    }

    // delta right
    for (this_Ij, resulting_k_density) in kij_given_prev_k_delta_right_and_this_Ij.iter_mut().enumerate()
    {
        let kIj = this_Ij + idx_s;
        if kIj+1 < len {
            let range_delta_left = kIj+1..len;
            let length = range_delta_left.len();
            resulting_k_density.delta.0 += len_recip * length as f64 * bin_size;
        }
        let start = if kIj >= idx_s{
            let end = kIj - idx_s;
            //let range_delta_right = 0..=end;
            let length = end + 1;
            resulting_k_density.delta.1 += len_recip * length as f64 * bin_size;
            end + 1
        } else {
            0
        };
        let end = kIj.min(len - 1);
        let range = start..=end;
        for mi in range{ // Optimizable
            let k = kIj - mi;
            resulting_k_density.func[k] += len_recip;
        }
    }

    let mut sanity_check_k = pk.k_density.create_zeroed();

    for (k_density, matr) in pk.k_density.func.iter().zip(kij_given_prev_k_and_this_Ij.iter())
    {
        let k_prob = k_density * bin_size;
        let prob = k_prob * len_recip;
        // all in line have same weight because Ij is uniform
        for line in matr{
            sanity_check_k.func
                .iter_mut()
                .zip(line.func.iter())
                .for_each(
                    |(a,b)|
                    {
                        *a += b * prob;
                    }
                );
            
            sanity_check_k.delta.0 += line.delta.0 * prob;
            sanity_check_k.delta.1 += line.delta.1 * prob;
        }
    }
    {
        // delta left
        let k_prob = pk.k_density.delta.0;
        let prob = k_prob * len_recip;
        // all in line have same weight because Ij is uniform
        for line in kij_given_prev_k_delta_left_and_this_Ij.iter(){
            sanity_check_k.func
                .iter_mut()
                .zip(line.func.iter())
                .for_each(
                    |(a,b)|
                    {
                        *a += b * prob;
                    }
                );
            
            sanity_check_k.delta.0 += line.delta.0 * prob;
            sanity_check_k.delta.1 += line.delta.1 * prob;
        }
    }
    {
        // delta right 
        let k_prob = pk.k_density.delta.1;
        let prob = k_prob * len_recip;
        // all in line have same weight because Ij is uniform
        for line in kij_given_prev_k_delta_right_and_this_Ij.iter(){
            sanity_check_k.func
                .iter_mut()
                .zip(line.func.iter())
                .for_each(
                    |(a,b)|
                    {
                        *a += b * prob;
                    }
                );
            
            sanity_check_k.delta.0 += line.delta.0 * prob;
            sanity_check_k.delta.1 += line.delta.1 * prob;
        }
    }
    sanity_check_k.normalize(bin_size);
    sanity_check_k.write("check_k", bin_size, pk.s);

    let mut Ii_given_this_k_and_this_Ij = vec![vec![vec![0.0; len]; len]; pk.k_density.func.len()];
    let mut Ii_given_this_k_delta_left_and_this_Ij = vec![vec![0.0; len]; len];
    let mut Ii_given_this_k_delta_right_and_this_Ij = vec![vec![0.0; len]; len];


    for (k, Ii_given_k) in Ii_given_this_k_and_this_Ij.iter_mut().enumerate() {
        for (Ij, Ii_given_k_and_Ij) in Ii_given_k.iter_mut().enumerate(){
            let kIj = k + Ij;
            for m in 0..len{
                let Ii = m.min(kIj);
                Ii_given_k_and_Ij[Ii] += len_recip;
            }
        }
    }
    // delta left 
    for (Ij, Ii_given_k_and_Ij) in Ii_given_this_k_delta_left_and_this_Ij.iter_mut().enumerate()
    {
        let kIj = Ij; // k = 0
        for m in 0..len{
            let Ii = m.min(kIj);
            Ii_given_k_and_Ij[Ii] += len_recip;
        }
    }
    // delta right 
    for (Ij, Ii_given_k_and_Ij) in Ii_given_this_k_delta_right_and_this_Ij.iter_mut().enumerate()
    {
        let kIj = Ij + idx_s; 
        for m in 0..len{
            let Ii = m.min(kIj);
            Ii_given_k_and_Ij[Ii] += len_recip;
        }
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

    let mut another_sanity = vec![0.0; len];
    for (matr, k_prob) in Ii_given_this_k_and_this_Ij.iter().zip(pk.k_density.func.iter()){
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
    let prob = pk.k_density.delta.0  * len_recip;
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
    let prob = pk.k_density.delta.1  * len_recip;
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



    // This aggregation only works for Ij uniform
    let aggregated = Ii_given_this_k_and_this_Ij
        .iter()
        .map(
            |matr|
            {
                let mut sum = matr[0].clone();
                for line in &matr[1..]
                {
                    sum.iter_mut()
                        .zip(line)
                        .for_each(|(a,b)| *a += b);
                }
                sum
            }
        ).collect_vec();
    
    let Ii_given_prev_Ii_global = Mutex::new(vec![vec![0.0; len]; len]);

    let chunk_vec = pk.k_density
        .func.iter()
        .enumerate()
        .collect_vec();

    let chunk_size = (chunk_vec.len() as f64 / 24.0).ceil() as usize;
    chunk_vec.par_chunks(chunk_size)
        .progress()
        .for_each(
        |chunk: &[(usize, &f64)]|
        {
            let mut Ii_given_prev_Ii = vec![vec![0.0; len]; len];
            for (prev_k, &k_density) in chunk{
                let prob: f64 = k_density * bin_size * len_recip2;
                for prev_Ij in 0..len{
                    let prev_kIj = prev_k + prev_Ij;

                    let end = prev_kIj.min(len);
                    let m_smaller_range = 0..end;
                    for m in m_smaller_range{
                        let prev_Ii = m;
                        let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();
                        let other_k = (prev_kIj - prev_Ii).min(idx_s);
                        if other_k < idx_s {
                            let ag = aggregated[other_k].as_slice();
                            res_Ii_vec
                                .iter_mut()
                                .zip(ag)
                                .for_each(
                                    |(res, Ii_prob)|
                                    {
                                        *res += Ii_prob * prob
                                    }
                                );
                        } else {
                            let Ii_given_k_slice = Ii_given_this_k_and_this_Ij
                                .get(other_k)
                                .unwrap_or(&Ii_given_this_k_delta_right_and_this_Ij);
        
                            for next_Ii_density in Ii_given_k_slice.iter(){
                                // iterating through next_Ij
                                // for future: If Ij depends upon the previous stuff, insert that here
                                res_Ii_vec
                                    .iter_mut()
                                    .zip(next_Ii_density)
                                    .for_each(
                                        |(res, Ii_prob)|
                                        {
                                            *res += Ii_prob * prob
                                        }
                                    );
                            }
                        }
                        
                    }
                    if len > end {
                        let kIj_range = end..len;
                        let remaining = kIj_range.len();
                        let prob = prob * remaining as f64;
                        let prev_Ii = prev_kIj;
                        let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();
                        let other_k = (prev_kIj - prev_Ii).min(idx_s);
                        let Ii_given_k_slice = Ii_given_this_k_and_this_Ij
                            .get(other_k)
                            .unwrap_or(&Ii_given_this_k_delta_right_and_this_Ij);
        
                        for next_Ii_density in Ii_given_k_slice.iter(){
                            // iterating through next_Ij
                            // for future: If Ij depends upon the previous stuff, insert that here
                            res_Ii_vec
                                .iter_mut()
                                .zip(next_Ii_density)
                                .for_each(
                                    |(res, Ii_prob)|
                                    {
                                        *res += Ii_prob * prob
                                    }
                                );
                        }
                        
                    }
                }
            }

            let mut guard = Ii_given_prev_Ii_global.lock().unwrap();
            guard.iter_mut()
                .zip(Ii_given_prev_Ii)
                .for_each(
                    |(res, input)|
                    {
                        res.iter_mut()
                            .zip(input)
                            .for_each(
                                |(a,b)|
                                *a += b
                            );
                    }
                );
            drop(guard);
            
        }
    );
    let mut Ii_given_prev_Ii = Ii_given_prev_Ii_global.into_inner().unwrap();
    // delta left
    let prev_k_prob = pk.k_density.delta.0;
    let prob = prev_k_prob * len_recip2;
    for prev_Ij in 0..len{
        let prev_kIj = prev_Ij; // k = 0
        for m in 0..len{
            let prev_Ii = m.min(prev_kIj);
            let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();

            let other_k = (prev_kIj - prev_Ii).min(idx_s);
            let Ii_given_k_slice = Ii_given_this_k_and_this_Ij
                .get(other_k)
                .unwrap_or(&Ii_given_this_k_delta_right_and_this_Ij);
            for next_Ii_density in Ii_given_k_slice.iter(){
                // iterating through next_Ij
                // for future: If Ij depends upon the previous stuff, insert that here
                res_Ii_vec
                    .iter_mut()
                    .zip(next_Ii_density)
                    .for_each(
                        |(res, Ii_prob)|
                        {
                            *res += Ii_prob * prob
                        }
                    );
            }
        }
    }
    // delta right
    let prev_k_prob = pk.k_density.delta.1;
    let prob = prev_k_prob * len_recip2;
    for prev_Ij in 0..len{
        let prev_kIj = prev_Ij + idx_s; 
        for m in 0..len{
            let prev_Ii = m.min(prev_kIj);
            let res_Ii_vec = Ii_given_prev_Ii[prev_Ii].as_mut_slice();

            let other_k = (prev_kIj - prev_Ii).min(idx_s);
            let Ii_given_k_slice = Ii_given_this_k_and_this_Ij
                .get(other_k)
                .unwrap_or(&Ii_given_this_k_delta_right_and_this_Ij);
            for next_Ii_density in Ii_given_k_slice.iter(){
                // iterating through next_Ij
                // for future: If Ij depends upon the previous stuff, insert that here
                res_Ii_vec
                    .iter_mut()
                    .zip(next_Ii_density)
                    .for_each(
                        |(res, Ii_prob)|
                        {
                            *res += Ii_prob * prob
                        }
                    );
            }
        }
    }

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

fn master_ansatz_k(
    a: &[f64], 
    s: f64, 
    threshold: f64
)-> Pk
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
    let mut k_guess = vec![guess_height; index_s]; /// TODO After optimizing I want to check if removing the 1 here improves the results. 
    /// I want to do this after optimizing, because it leads to off by one errors in the code elsewhere, i.e., runtime errors currently

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
            return Pk{
                k_density: density,
                bin_size,
                s,
                len_of_1: len,
                index_s
            };
        }
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

    let p_ka = (0..(pk.k_density.func.len() + pk.len_of_1))
        .map(
            |x|
            {
                let mut integral = 0.0;
                let start = if x < pk.len_of_1 {
                    0
                } else {
                    x - (pk.len_of_1 - 1)
                };
                let end = if x >= pk.k_density.func.len(){
                    pk.k_density.func.len() - 1
                } else {
                    x
                };
                for j in start..=end{
                    integral += pk.k_density.func[j] * a_ij[x-j];
                }
                integral *= pk.bin_size;

                if start == 0{
                    integral += pk.k_density.delta.0 * a_ij[x]; 
                }
                if x >= pk.index_s && x-pk.index_s < a_ij.len() {
                    integral += pk.k_density.delta.1 * a_ij[x-pk.index_s];
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
    let derivative_left = sampling::glue::derivative::derivative(&prob[..pk.index_s]);
    let derivative_right = sampling::glue::derivative::derivative(&prob[pk.index_s..pk.len_of_1]);
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

