use std::{io::BufReader, num::NonZeroUsize, time::Duration};
use indicatif::ProgressIterator;
use itertools::*;
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

            let P_I_N1_given_prior_I_N1 = master_ansatz_i_test(&pk_N1, &production_N0, &production_N1);

            let (pk_N2_given_I_N1, pk_N2) = calk_k_master_test(
                &pk_N1,
                &P_I_N1_given_prior_I_N1,
                &production_N1
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
            std::thread::sleep(Duration::from_secs(5));
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
    
        let pk = Pk{
            bin_size: save_state.bin_size,
            delta_left: save_state.pk_N2.delta.0,
            delta_right: save_state.pk_N2.delta.1,
            function: save_state.pk_N2.func.clone(),
            s: save_state.input.s,
            len_of_1: save_state.len_of_1,
            index_s: save_state.idx_s
        };
    
        let (pk_N3_given_I_N2, pk_N3) = calk_k_master_test(
            &pk,
            &calc_result.I2_given_prev_I2,
            &calc_result.I2_density
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

#[derive(Serialize, Deserialize)]
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
                let mut sum: f64 = line.iter().sum();
                sum *= bin_size;
                let factor = sum.recip();
                line.iter_mut()
                    .for_each(|val| *val *= factor)
            }
        );
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
    for (line, prob_density) in pre_I1_given_I1.iter().zip(I_N1.iter())
    {
        let prob = prob_density * bin_size;
        pre_I1_sanity_check.iter_mut()
            .zip(line)
            .for_each(
                |(pre, line_entry)| *pre += line_entry * prob
            )
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
    prior_I_for_normalization: &[f64]
) -> (Vec<ProbabilityDensity>, ProbabilityDensity)
{
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
    let mut counter = 0;
    loop {
        for (prior_I_idx, current_Ij_distribution) in input_P_I_given_prior_I.iter().enumerate().progress(){
            let current_k = &current_estimate_given_prior_I[prior_I_idx];
            let prior_I_prob = prior_I_for_normalization[prior_I_idx];
            let m_factor_times_prior_I_prob = prior_I_prob * m_factor;
            for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                let update_k_vec = &mut next_estimate_given_prior_I[Ij_idx];
                for (k, k_prob) in current_k.func.iter().enumerate(){
                    let probability_increment = k_prob * m_factor_times_prior_I_prob * Ij_prob;
                    let kI = Ij_idx + k;
                    for m in 0..len_of_1{
        
                        if m > kI {
                            update_k_vec.delta.0 += probability_increment;
                        } else if kI -m > idx_s {
                            update_k_vec.delta.1 += probability_increment;
                        } else {
                            update_k_vec.func[kI-m] += probability_increment;
                        }
        
                    }
                }

            }
    
            // left
            let left_increment = m_factor_times_prior_I_prob * current_k.delta.0 / bin_size; // TODO probably /bin_size or something like that missing!
            for m in 0..len_of_1{
                for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                    let update_k_vec = &mut next_estimate_given_prior_I[Ij_idx];
                    if m > Ij_idx {
                        update_k_vec.delta.0 += left_increment * Ij_prob;
                    } else if Ij_idx -m > idx_s {
                        update_k_vec.delta.1 += left_increment * Ij_prob;
                    } else {
                        update_k_vec.func[Ij_idx-m] += left_increment * Ij_prob;
                    }
                }
            }
    
            // right
            let right_increment = m_factor_times_prior_I_prob * current_k.delta.1 / bin_size;
            for (Ij_idx, Ij_prob) in current_Ij_distribution.iter().enumerate(){
                let kI: usize = Ij_idx + idx_s;
                let increment = right_increment * Ij_prob;
                let update_k_vec = &mut next_estimate_given_prior_I[Ij_idx];
                for m in 0..len_of_1{
                    if m > kI {
                        update_k_vec.delta.0 += increment;
                    } else if kI -m > idx_s {
                        update_k_vec.delta.1 += increment;
                    } else {
                        update_k_vec.func[kI-m] += increment;
                    }
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
        if counter == 20 {
            break;
        }
        next_estimate_given_prior_I
            .iter_mut()
            .for_each(ProbabilityDensity::make_zero);
        resulting_density.make_zero();
        counter += 1;



    }
    
    (current_estimate_given_prior_I, resulting_density)

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
    threshold: f64
)-> Pk
{
    let len = a.len();
    let index_s = (s * (len - 1) as f64).floor() as usize;
    let bin_size = ((len) as f64).recip();

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
    let mut k_guess = vec![guess_height; index_s+1]; // maybe I somewhere have indexmissmatch for index s?

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

