use indicatif::{ProgressIterator, ParallelProgressIterator};
use rand::RngCore;
use rand_distr::{Distribution, Exp, Exp1, Normal, Uniform};
use rand_pcg::{Pcg64, Pcg64Mcg};
use rand_xoshiro::{Xoshiro256PlusPlus, SplitMix64};
use rand_chacha::ChaCha20Rng;
use rand::{Rng, SeedableRng, rngs::mock::StepRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use crate::index_sampler::IndexSampler;
use crate::misc::*;
use std::io::{Write, stdout};
use std::num::{NonZeroU32, NonZeroUsize};
use std::sync::Mutex;
use crate::correlations::*;

pub struct WorstRng{
    rng: StepRng
}

impl RngCore for WorstRng{
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest)
    }

    fn next_u32(&mut self) -> u32 {
        self.rng.next_u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.rng.try_fill_bytes(dest)
    }
}

impl SeedableRng for WorstRng{
    type Seed = [u8;3];
    fn from_entropy() -> Self {
        unimplemented!()
    }

    fn seed_from_u64(state: u64) -> Self {
        let rng = StepRng::new(
            state.wrapping_mul(2), 
            22801763489_u64.wrapping_mul(22801763489)
        );
        Self{rng}
    }
    fn from_rng<R: RngCore>(_: R) -> Result<Self, rand::Error> {
        unimplemented!()
    }

    fn from_seed(_: Self::Seed) -> Self {
        unimplemented!()
    }
}

#[derive(Clone, Copy, Serialize, Debug, Deserialize, Default)]
pub enum RngChoice
{
    XorShift,
    #[default]
    Pcg64,
    Pcg64Mcg,
    ChaCha20,
    BadRng,
    WorstRng
}

impl RngChoice{
    pub fn check_warning(self)
    {
        match self{
            Self::BadRng | Self::WorstRng => {
                eprintln!("WARNING: You are using a very bad RNG. This is only intended for tests!");
            },
            _ => ()
        }
    }
}

impl PrintAlternatives for RngChoice{
    fn print_alternatives(layer: u8) {
        let a = RngChoice::Pcg64;
        let b = RngChoice::XorShift;
        let c = RngChoice::ChaCha20;
        let d = RngChoice::Pcg64Mcg;

        let all = [a, b, c, d];
        print_alternatives_helper(&all, layer, "RngChoice");
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy)]
pub enum SelfLinks{
    #[default]
    AllowSelfLinks,
    NoSelfLinks,
    AlwaysSelfLink,
    AllowMultiple
}

impl PrintAlternatives for SelfLinks{
    fn print_alternatives(layer: u8) {
        let a = Self::AllowSelfLinks;
        let b = Self::NoSelfLinks;
        let c = Self::AlwaysSelfLink;
        let d = Self::AllowMultiple;
        
        let all = [a, b, c, d];
        print_alternatives_helper(&all, layer, "SelfLinks")
    }
}

impl SelfLinks{
    pub fn get_step_fun<R>(&self) -> fn (&mut SubstitutingMeanField::<R>)
    where R: Rng + SeedableRng
    {
        match self{
            Self::AllowSelfLinks => SubstitutingMeanField::step_with_self_links,
            Self::NoSelfLinks => SubstitutingMeanField::step_without_self_links,
            Self::AlwaysSelfLink => SubstitutingMeanField::step_always_self_links,
            Self::AllowMultiple => SubstitutingMeanField::step_allowing_multiple
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SubstitutionVelocityVideoOpts{
    pub buffer: SampleRangeF64,
    pub substitution_prob: SampleRangeF64,
    pub opts: SubstitutingMeanFieldOpts,
    pub time_steps: usize,
    pub self_links: SelfLinks,
    pub yrange: Option<(f32, f32)>,
    pub reset_fraction: Option<f64>,
    pub samples_per_point: NonZeroU32,
    pub buffer_dist: BufferDist,
    pub sub_dist: PossibleDists,
    pub rng_choice: RngChoice
}

impl PrintAlternatives for SubstitutionVelocityVideoOpts{
    fn print_alternatives(layer: u8) {
        let this = Self::default();
        let mut stdout = stdout();
        let msg = "Serialization issue SubstitutionVelocityVideoOpts";
        print_spaces(layer);
        println!("SubstitutionVelocityVideoOpts:");
        serde_json::to_writer_pretty(&mut stdout, &this)
            .expect(msg);
        println!();
        print_spaces(layer);
        println!("Note: yrange is allowed to be null");
        print_spaces(layer);
        println!("Note: reset_fraction is allowed to be null");
        print_spaces(layer);
        println!("Alternatives for buffer_dist:");
        BufferDist::print_alternatives(layer + 1);
        print_spaces(layer);
        println!("Alternatives for sub_dist:");
        PossibleDists::print_alternatives(layer + 1);
        println!("Alternatives for rng_choice:");
        RngChoice::print_alternatives(layer + 1);
    }
}

impl Default for SubstitutionVelocityVideoOpts{
    fn default() -> Self {
        Self { 
            buffer: SampleRangeF64::default(), 
            substitution_prob: SampleRangeF64::default(), 
            opts: SubstitutingMeanFieldOpts::default(), 
            time_steps: 1000, 
            self_links: SelfLinks::default(), 
            yrange: Some((0.0,3.5)),
            reset_fraction: None,
            samples_per_point: NonZeroU32::new(1).unwrap(),
            buffer_dist: BufferDist::default(),
            sub_dist: PossibleDists::default(),
            rng_choice: RngChoice::default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct SubstitutionVelocitySampleOpts{
    pub buffer: SampleRangeF64,
    pub opts: SubstitutingMeanFieldOpts,
    pub time_steps: usize,
    pub self_links: SelfLinks,
    pub rng_choice: RngChoice
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct SubstitutingMeanFieldOpts{
    pub buffer: f64,
    pub substitution_prob: f64,
    pub seed: u64,
    pub k: usize,
    pub n: usize,
    pub lambda: f64
}

impl SubstitutingMeanFieldOpts{
    pub fn get_buffers(&self) -> Vec<f64>
    {
        vec![self.buffer; self.n]
    }

    pub fn get_substitution_prob(&self) -> Vec<f64>
    {
        vec![self.substitution_prob; self.n]
    }
}

pub struct SubstitutingMeanField<RNG>{
    current_delays: Vec<f64>,
    buffers: Vec<f64>,
    substitution_prob: Vec<f64>,
    next_delays: Vec<f64>,
    k: usize,
    rng: RNG,
    index_sampler: IndexSampler,
    dist: Exp1
}

impl<R> SubstitutingMeanField<R>
where R: Rng + SeedableRng{

    pub fn reseed_sub_prob(&mut self, sub_prob: f64, fraction: f64)
    {
        let mut amount = (self.substitution_prob.len() as f64 * fraction)
            .round() as usize;
        amount = amount.min(self.substitution_prob.len());
        let (to_set, rest) = self.index_sampler
            .sample_inplace_amount_rest(&mut self.rng, amount);
        for &i in to_set{
            let index = i as usize;
            self.substitution_prob[index] = sub_prob;
        }
        for &i in rest{
            let index = i as usize;
            self.substitution_prob[index] = 0.0;
        }
    }

    pub fn reset_delays(&mut self)
    {
        self.current_delays.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn change_buffer_to_const(&mut self, const_val: f64)
    {
        self.buffers.iter_mut()
            .for_each(|v| *v = const_val)
    }

    pub fn change_substitution_prob_to_const(&mut self, const_val: f64)
    {
        self.substitution_prob
            .iter_mut()
            .for_each(|v| *v = const_val)
    }

    pub fn change_substitution_prob<D>(
        &mut self, 
        sub_dist: D
    )
    where D: Distribution<f64>
    {
        self.substitution_prob
            .iter_mut()
            .zip(sub_dist.sample_iter(&mut self.rng))
            .for_each(
                |(sub, rand_val)|
                {
                    *sub = rand_val;
                    if *sub < 0.0 {
                        *sub = 0.0;
                    }
                    if *sub > 1.0 {
                        *sub = 1.0;
                    }
                }
            )
    }

    pub fn change_buffer_dist_min_max<D>(
        &mut self,
        buffer_dist: D, 
        min_buf: f64, 
        max_buf: f64
    )
    where D: Distribution<f64>
    {
        self.buffers
            .iter_mut()
            .zip(buffer_dist.sample_iter(&mut self.rng))
            .for_each(
                |(buffer, rand_val)|
                {
                    *buffer = rand_val;
                    // interestingly two ifs are faster than if - else
                    // because we do not need any jumps in assembly
                    if *buffer > max_buf{
                        *buffer = max_buf;
                    }
                    if *buffer < min_buf {
                        *buffer = min_buf
                    }
                }
            );
    }

    pub fn get_k(&self) -> usize {
        self.k
    }

    pub fn new(opt: &SubstitutingMeanFieldOpts) -> Self
    {
        let current_delays = vec![0.0; opt.n];
        let next_delays = vec![0.0; opt.n];
        let mut rng = R::seed_from_u64(opt.seed);
        let index_sampler = IndexSampler::measure_which(
            opt.n, 
            opt.k, 
            &mut rng
        );
        assert_eq!(opt.lambda, 1.0, "For optimization reasons other lamba are currently not implemented!");
        // let exp = Exp::new(opt.lambda)
        //    .unwrap();
        let exp = Exp1;
        Self{
            current_delays,
            next_delays,
            buffers: opt.get_buffers(),
            k: opt.k,
            index_sampler,
            substitution_prob: opt.get_substitution_prob(),
            rng,
            dist: exp
        }
    }

    pub fn step_without_self_links(&mut self)
    {
        self.next_delays.iter_mut()
            .enumerate()
            .for_each(
                |(index, n_delay)|
                {
                    if self.rng.gen::<f64>() < self.substitution_prob[index]{
                        *n_delay = self.dist.sample(&mut self.rng);
                    } else {
                        let mut current = 0.0_f64;
                        for i in self.index_sampler.sample_indices_without(&mut self.rng, index as u32){
                            let i = *i as usize;
                            current = current.max(self.current_delays[i]);
                        }
                        let e_sample: f64 = self.dist.sample(&mut self.rng);
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + e_sample;
                        
                    }
                    
                }
            );
        std::mem::swap(&mut self.current_delays, &mut self.next_delays);
    }

    pub fn step_always_self_links(&mut self)
    {
        self.next_delays.iter_mut()
            .enumerate()
            .for_each(
                |(index, n_delay)|
                {
                    if self.rng.gen::<f64>() < self.substitution_prob[index]{
                        *n_delay = self.dist.sample(&mut self.rng);
                    } else {
                        let mut current = 0.0_f64;
                        for i in self.index_sampler
                            .sample_indices_without(&mut self.rng, index as u32)
                            .iter()
                            .skip(1)
                            .copied()
                            .chain(std::iter::once(index as u32))
                        {
                            let i = i as usize;
                            current = current.max(self.current_delays[i]);
                        }
                        let e_sample: f64 = self.dist.sample(&mut self.rng);
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + e_sample;
                        
                    }
                    
                }
            );
        std::mem::swap(&mut self.current_delays, &mut self.next_delays);
    }

    pub fn step_with_self_links(&mut self)
    {
        self.next_delays.iter_mut()
            .enumerate()
            .for_each(
                |(index, n_delay)|
                {
                    if self.rng.gen::<f64>() < self.substitution_prob[index]{
                        *n_delay = self.dist.sample(&mut self.rng);
                    } else {
                        let mut current = 0.0_f64;
                        for i in self.index_sampler.sample_indices(&mut self.rng){
                            let i = *i as usize;
                            current = current.max(self.current_delays[i]);
                        }
                        let e_sample: f64 = self.dist.sample(&mut self.rng);
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + e_sample;
                    }
                }
            );
        std::mem::swap(&mut self.current_delays, &mut self.next_delays);
    }

    pub fn step_allowing_multiple(&mut self)
    {
        let uni = Uniform::new(0, self.next_delays.len());
        self.next_delays.iter_mut()
            .enumerate()
            .for_each(
                |(index, n_delay)|
                {
                    if self.rng.gen::<f64>() < self.substitution_prob[index]{
                        *n_delay = self.dist.sample(&mut self.rng);
                    } else {
                        let mut current = 0.0_f64;
                        for i in uni.sample_iter(&mut self.rng).take(self.k){
                            current = current.max(self.current_delays[i]);
                        }
                        let e_sample: f64 = self.dist.sample(&mut self.rng);
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + e_sample;
                    }
                }
            );
        std::mem::swap(&mut self.current_delays, &mut self.next_delays);
    }

    pub fn average_delay(&self) -> f64{
        self.current_delays.iter().sum::<f64>() / self.current_delays.len() as f64
    }
}


pub fn sample_velocity(opt: &SubstitutionVelocitySampleOpts, out_stub: &str){
    opt.rng_choice.check_warning();
    match opt.rng_choice{
        RngChoice::Pcg64 => {
            sample_velocity_helper::<Pcg64>(opt, out_stub)
        },
        RngChoice::Pcg64Mcg => {
            sample_velocity_helper::<Pcg64Mcg>(opt, out_stub)
        }
        RngChoice::XorShift => {
            sample_velocity_helper::<Xoshiro256PlusPlus>(opt, out_stub)
        },
        RngChoice::ChaCha20 => {
            sample_velocity_helper::<ChaCha20Rng>(opt, out_stub)
        },
        RngChoice::BadRng => {
            sample_velocity_helper::<SplitMix64>(opt, out_stub)
        },
        RngChoice::WorstRng => {
            sample_velocity_helper::<WorstRng>(opt, out_stub)
        }
    }
}

fn sample_velocity_helper<R>(opt: &SubstitutionVelocitySampleOpts, out_stub: &str)
where R: Rng + SeedableRng
{
    let name = format!("{out_stub}.dat");
    let mut writer = create_buf_with_command_and_version(name);
    let header = ["B", "Velocity"];
    write_slice_head(&mut writer, header).unwrap();

    let mut model = SubstitutingMeanField::<R>::new(&opt.opts);

    let fun = opt.self_links.get_step_fun();

    let bar = crate::misc::indication_bar(opt.buffer.samples as u64);

    for b in opt.buffer.get_iter().progress_with(bar){
        model.change_buffer_to_const(b);
        model.reset_delays();
        for _ in 0..opt.time_steps{
            fun(&mut model);
        }
        let velocity = model.average_delay() / opt.time_steps as f64;
        writeln!(writer, "{b} {velocity}").unwrap();
    }
}

#[derive(Default)]
pub struct Cleaner{
    list: Mutex<Vec<String>>
}

impl Cleaner{
    pub fn new() -> Self{
        Self::default()
    }

    pub fn add(&self, s: String)
    {
        let mut lock = self.list.lock().unwrap();
        lock.push(s);
        drop(lock);
    }

    pub fn add_multi<I>(&self, iter: I)
    where I: IntoIterator<Item = String>
    {
        let mut lock = self.list.lock().unwrap();
        lock.extend(iter);
        drop(lock);
    }

    pub fn clean(self){
        let list = self.list
            .into_inner()
            .unwrap();
        for s in list{
            let _ = std::fs::remove_file(&s);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct UniformAround{
    interval_length_half: f64
}

impl UniformAround{
    fn get_uniform(&self, mid: f64) -> Uniform<f64>
    {
        let left = mid - self.interval_length_half;
        let right = mid + self.interval_length_half;
        Uniform::new_inclusive(left, right)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Gauss{
    pub std_dev: f64,
}

impl Gauss{
    pub fn get_gauss(&self, mean: f64) -> Normal<f64>
    {
        Normal::new(mean, self.std_dev).unwrap()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum PossibleDists{
    #[default]
    Const,
    Exp,
    UniformAround(UniformAround),
    Gauss(Gauss),
    Beta,
    Pow
}

impl PossibleDists{

    pub fn is_reset_fraction_allowed(self) -> bool
    {
        !matches!(self, Self::Beta)
    }

    pub fn gnuplot_x_axis_name(self) -> &'static str
    {
        match self {
            Self::Beta | Self::Pow => "α",
            _ => "<p_s>"
        }
    }
   
    pub fn is_const(&self) -> bool
    {
        matches!(self, Self::Const)
    }

    #[allow(clippy::type_complexity)]
    pub fn get_sub_fun<'a, R>(&'a self) -> Box<dyn Fn (&mut SubstitutingMeanField::<R>, f64) + Sync + 'a>
    where R: Rng + SeedableRng
    {
        match self{
            Self::Const =>
            {
                Box::new(|_,_|{})
            },
            Self::Exp => {
                let fun = |model: &mut SubstitutingMeanField::<R>, lambda: f64|
                {
                    let dist = Exp::new(lambda).unwrap();
                    model.change_substitution_prob(
                        dist
                    );
                };
                Box::new(fun)
            },
            PossibleDists::UniformAround(uni) => {
                let fun = move |model: &mut SubstitutingMeanField::<R>, mid: f64|
                {
                    let dist = uni.get_uniform(mid);
                    model.change_substitution_prob(
                        dist
                    );
                };
                Box::new(fun)
            },
            PossibleDists::Gauss(gauss) => {
                let fun = move |model: &mut SubstitutingMeanField::<R>, mean: f64|
                {
                    let dist = gauss.get_gauss(mean);
                    model.change_substitution_prob(
                        dist
                    );
                };
                Box::new(fun)
            },
            Self::Beta => {
                // \frac{x^{\left(l-1\right)}\left(1-x\right)^{\left(1-l\right)}}{-(-1+l)\pi\csc(l\pi)}
                let fun = |model: &mut SubstitutingMeanField::<R>, alpha: f64|
                {
                    let beta = 2.0 - alpha;
                    let dist = rand_distr::Beta::new(alpha, beta)
                        .unwrap();
                    model.change_substitution_prob(
                        dist
                    );
                };
                Box::new(fun)
            },
            Self::Pow =>
            {
                // alpha to p_s:
                // f(alpha)=(alpha+1.0)/(alpha+2.0)
                let uni: Uniform<f64> = Uniform::new_inclusive(0.0, 1.0);
                let fun = move |model: &mut SubstitutingMeanField::<R>, alpha: f64|
                {
                    assert!(
                        alpha >= -1.0, 
                        "Integral does not converge for x^a for a <= 1.0, So it is not a valid probability distribution. Thus: ERROR!"
                    );
                    if alpha == -1.0 {
                        model.change_substitution_prob_to_const(0.0);
                    } else {
                        let exponent = 1.0 /(1.0 + alpha);
                        let dist = uni.map(
                            |val|
                            val.powf(exponent)
                        );
                        model.change_substitution_prob(
                            dist
                        );
                    }
                    
                };
                Box::new(fun)
            }
        }
    }
}

impl PrintAlternatives for PossibleDists{
    fn print_alternatives(layer: u8) {
        let a = Self::Const;
        let b = Self::Exp;
        let c = Self::UniformAround(UniformAround { interval_length_half: 0.2 });
        let d = Self::Gauss(Gauss { std_dev: 0.2 });
        let e = Self::Beta;
        let f = Self::Pow;

        let all = [a, b, c, d, e, f];

        print_alternatives_helper(&all, layer, "PossibleDists");
    }
}


#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct BufferDist{
    pub min: f64,
    pub max: f64,
    pub dist: PossibleDists
}

impl PrintAlternatives for BufferDist {
    fn print_alternatives(layer: u8) {
        let mut stdout = stdout();
        let msg = "Serialization issue BufferDist";
        let this = Self::default();
        print_spaces(layer);
        println!("BufferDist:");
        serde_json::to_writer_pretty(&mut stdout, &this)
            .expect(msg);
        print_spaces(layer);
        println!();
        println!("Alternatives PossibleDists:");
        PossibleDists::print_alternatives(layer + 1);
    }
}


impl BufferDist{

    fn assert_unequal(&self)
    {
        assert_ne!(
            self.min, 
            self.max, 
            "Equal buffer thresholds do not make sense in this context"
        ); 
    }



    #[allow(clippy::type_complexity)]
    pub fn change_buffers_fun<'a, R>(&'a self) -> Box<dyn Fn(&mut SubstitutingMeanField::<R>, f64) + 'a + Sync>
    where R: SeedableRng + Rng + 'a
    {
        match self.dist{
            PossibleDists::Const =>
            {
                Box::new(SubstitutingMeanField::<R>::change_buffer_to_const)
            },
            PossibleDists::Exp => {
                self.assert_unequal();
                let fun = |model: &mut SubstitutingMeanField::<R>, lambda: f64|
                {
                    let dist = Exp::new(lambda).unwrap();
                    model.change_buffer_dist_min_max(
                        dist,
                        self.min, 
                        self.max
                    );
                };
                Box::new(fun)
            },
            PossibleDists::UniformAround(uni) => {
                self.assert_unequal();
                let fun = move |model: &mut SubstitutingMeanField::<R>, mid: f64|
                {
                    let dist = uni.get_uniform(mid);
                    model.change_buffer_dist_min_max(
                        dist,
                        self.min, 
                        self.max
                    );
                };
                Box::new(fun)
            },
            PossibleDists::Gauss(gauss) => {
                self.assert_unequal();
                let fun = move |model: &mut SubstitutingMeanField::<R>, mean: f64|
                {
                    let dist = gauss.get_gauss(mean);
                    model.change_buffer_dist_min_max(
                        dist,
                        self.min, 
                        self.max
                    );
                };
                Box::new(fun)
            },
            PossibleDists::Beta | PossibleDists::Pow => unimplemented!()
            
        }
    }
}

pub fn sample_velocity_video(opt: &SubstitutionVelocityVideoOpts, out_stub: &str, frametime: u8)
{
    opt.rng_choice.check_warning();
    match opt.rng_choice{
        RngChoice::Pcg64 => {
            sample_velocity_video_helper::<Pcg64>(opt, out_stub, frametime)
        },
        RngChoice::Pcg64Mcg => {
            sample_velocity_video_helper::<Pcg64Mcg>(opt, out_stub, frametime)
        },
        RngChoice::XorShift => {
            sample_velocity_video_helper::<Xoshiro256PlusPlus>(opt, out_stub, frametime)
        },
        RngChoice::ChaCha20 => {
            sample_velocity_video_helper::<ChaCha20Rng>(opt, out_stub, frametime)
        },
        RngChoice::BadRng => {
            sample_velocity_video_helper::<SplitMix64>(opt, out_stub, frametime)
        },
        RngChoice::WorstRng => {
            sample_velocity_video_helper::<WorstRng>(opt, out_stub, frametime)
        }
    }
}

fn sample_velocity_video_helper<R>(opt: &SubstitutionVelocityVideoOpts, out_stub: &str, frametime: u8)
where R: Rng + SeedableRng
{
    if opt.reset_fraction.is_some(){
        assert!(opt.sub_dist.is_reset_fraction_allowed());
    }
    let fun = opt.self_links.get_step_fun();
    if !opt.sub_dist.is_const(){
        assert!(
            opt.reset_fraction.is_none(),
            "Reset fraction not implemented with sub dists"
        );
    }
    let sub_fun = opt.sub_dist.get_sub_fun::<R>();
    let all_sub_probs: Vec<_> = opt.substitution_prob
        .get_iter()
        .collect();

    let zeros = "000000000";

    let cleaner = Cleaner::new();

    let bar = crate::misc::indication_bar(all_sub_probs.len() as u64);

    let change_buffers_fun = opt.buffer_dist.change_buffers_fun();

    let criticals: Vec<_> = all_sub_probs
        .par_iter()
        .enumerate()
        .filter_map(
            |(index, &sub_prob)|
            {
                let mut model_opt = opt.opts.clone();
                model_opt.substitution_prob = sub_prob;
                model_opt.seed = index as u64;

                let mut model = SubstitutingMeanField::new(&model_opt);

                let i_name = index.to_string();
                let start = i_name.len();
                let zeros = &zeros[start..];
                let stub = format!("TMP_{zeros}{i_name}{out_stub}");
                let w_name = format!("{stub}.dat");
                let mut writer = create_buf(&w_name);

                for b in opt.buffer.get_iter(){
                    let mut velocity_sum = 0.0;
                    (0..opt.samples_per_point.get())
                        .for_each(
                            |_|
                            {
                                change_buffers_fun(&mut model, b);
                                
                                if let Some(f) = opt.reset_fraction{
                                    model.reseed_sub_prob(sub_prob, f);
                                } else {
                                    // Sub fun currently not compatible with reset_fraction
                                    sub_fun(&mut model, sub_prob);
                                }
                                model.reset_delays();
                                
                                for _ in 0..opt.time_steps{
                                    fun(&mut model);
                                }
                                let velocity = model.average_delay() / opt.time_steps as f64;
                                velocity_sum += velocity;
                            }
                        );
                    let average_velocity = velocity_sum / opt.samples_per_point.get() as f64;
                    
                    writeln!(writer, "{b} {average_velocity}").unwrap();
                }
                drop(writer);
                let gp_name = format!("{stub}.gp");
                let mut gp_writer = create_gnuplot_buf(&gp_name);
                let png = format!("{stub}.png");
                writeln!(gp_writer, "set t pngcairo").unwrap();
                writeln!(gp_writer, "set output '{png}'").unwrap();
                writeln!(gp_writer, "set ylabel 'v'").unwrap();
                writeln!(gp_writer, "set xlabel 'B'").unwrap();
                writeln!(gp_writer, "set fit quiet").unwrap();
                writeln!(gp_writer, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
                writeln!(gp_writer, "f(x)=a*x+b").unwrap();
                writeln!(gp_writer, "fit f(x) '{w_name}' u 1:2:(t($2)) yerr via a,b").unwrap();
                writeln!(gp_writer, "set label 'p={sub_prob}' at screen 0.4,0.9").unwrap();
                if let Some((min, max)) = &opt.yrange{
                    writeln!(gp_writer, "set yrange [{min}:{max}]").unwrap();
                }
                writeln!(gp_writer, "p '{w_name}' t '', f(x)").unwrap();
                writeln!(gp_writer, "print(b)").unwrap();
                writeln!(gp_writer, "print(a)").unwrap();
                writeln!(gp_writer, "set output").unwrap();
                drop(gp_writer);
                let out = call_gnuplot(&gp_name);
                if out.status.success(){
                    let s = String::from_utf8(out.stderr)
                        .unwrap();
                
                    let mut iter = s.lines();

                    let b: f64 = iter.next().unwrap().parse().unwrap();
                    let a: f64 = iter.next().unwrap().parse().unwrap();
                    let crit = -b/a;
                    
                    cleaner.add_multi([w_name, gp_name, png]);
                    Some([sub_prob, a, b, crit])
                } else {
                    None
                }
                
            }
        ).progress_with(bar)
        .collect();

    let crit_stub = format!("{out_stub}_crit");
    let crit_name = format!("{crit_stub}.dat");
    let mut buf = create_buf_with_command_and_version(&crit_name);
    let header = ["sub_prob", "a", "b", "critical"];
    write_slice_head(&mut buf, header).unwrap();
    for s in criticals.iter(){
        writeln!(
            buf, 
            "{} {} {} {}", 
            s[0], 
            s[1], 
            s[2], 
            s[3]
        ).unwrap();
    }
    drop(buf);
    enum How{
        Linear,
        Complex,
        NoFit
    }

    impl How {
        fn is_no_fit(&self) -> bool
        {
            matches!(self, How::NoFit)
        }
    }

    let crit_gp_write = |how: How|
    {
        let crit_gp = format!("{crit_stub}.gp");
        let mut gp = create_gnuplot_buf(&crit_gp);
        writeln!(gp, "set t pdfcairo").unwrap();
        writeln!(gp, "set output '{crit_stub}.pdf'").unwrap();
        
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &val in criticals[0].iter().skip(1){
            if val > max{
                max = val;
            } 
            if val < min {
                min = val;
            }
        }
        writeln!(gp, "set yrange[{min}:{max}]").unwrap();
        writeln!(gp, "set ylabel 'B'").unwrap();
        match how{
            How::Complex => {
                writeln!(gp, "f(x)= a*x+b+k*x**l").unwrap();
            },
            How::Linear => {
                writeln!(gp, "f(x)= a*x+b").unwrap();
            },
            How::NoFit => ()
        }
        
        if !how.is_no_fit(){
            writeln!(gp, "t(x)=abs(x)>0.1?0.00000000001:10000000").unwrap();
            writeln!(gp, "g(x)= c*x+d").unwrap();
        }
    
        let s = opt.sub_dist.gnuplot_x_axis_name();

        let using = if let Some(f) = opt.reset_fraction{
            writeln!(gp, "set xlabel '{s} f'").unwrap();
            format!("($1*{f})")
        } else {
            writeln!(gp, "set xlabel '{s}'").unwrap();
            "1".to_owned()
        };
        match how{
            How::Complex => {
                writeln!(gp, "fit f(x) '{crit_name}' u {using}:2:(t({using})) yerr via a,b,k,l").unwrap();
            },
            How::Linear => {
                writeln!(gp, "fit f(x) '{crit_name}' u {using}:2:(t({using})) yerr via a,b").unwrap();
            },
            How::NoFit => ()
        }
        if !how.is_no_fit(){
            writeln!(gp, "fit g(x) '{crit_name}' u {using}:3:(t($3)) yerr via c,d").unwrap();
            writeln!(gp, "h(x)=-g(x)/f(x)").unwrap();
        }
        
        write!(
            gp, 
            "p '{crit_name}' u {using}:2 t 'a', '' u {using}:3 t 'b', '' u {using}:4 t 'Crit B'"
        ).unwrap();
        if how.is_no_fit(){
            writeln!(gp)
        } else {
            writeln!(gp, ", f(x) t 'fit a', g(x) t 'fit b', h(x) t 'approx'")
        }.unwrap();
        writeln!(gp, "set output").unwrap();
        drop(gp);
        crit_gp
    };

    let crit_gp = crit_gp_write(How::Complex);
    let out = call_gnuplot(&crit_gp);
    if !out.status.success(){
        eprintln!("CRIT GNUPLOT ERROR! Trying to recover by using linear function instead!");
        let crit_gp = crit_gp_write(How::Linear);
        let out = call_gnuplot(&crit_gp);
        if !out.status.success(){
            eprintln!("RECOVERY also failed :( Removing fit!");
            let crit_gp = crit_gp_write(How::NoFit);
            let out = call_gnuplot(&crit_gp);
            if !out.status.success(){
                eprintln!("This also failed...");
                dbg!(out);
            }
        } else {
            eprintln!("RECOVERY SUCCESS!");
        }
    }

    create_video("TMP_*.png", out_stub, frametime);
    cleaner.clean();
}

#[derive(Deserialize, Serialize)]
pub struct AutoOpts{
    mean_opts: SubstitutingMeanFieldOpts,
    total_steps: u32,
    self_links: SelfLinks,
    samples_per_seed: NonZeroU32,
    num_seeds: NonZeroU32,
    rng_choice: RngChoice
}

impl PrintAlternatives for AutoOpts
{
    fn print_alternatives(layer: u8) {
        let this = Self::default();
        let mut stdout = stdout();
        let msg = "Serialization issue AutoOpts";
        print_spaces(layer);
        println!("AutoOpts:");
        serde_json::to_writer_pretty(&mut stdout, &this)
            .expect(msg);
        println!();
        print_spaces(layer);
        println!("Alternatives for self_links:");
        SelfLinks::print_alternatives(layer + 1);
        print_spaces(layer);
        println!("Alternatives for rng_choice:");
        RngChoice::print_alternatives(layer + 1);
    }
}

impl Default for AutoOpts{
    fn default() -> Self {
        Self{
            mean_opts: SubstitutingMeanFieldOpts::default(),
            total_steps: 500000,
            samples_per_seed: NonZeroU32::new(1).unwrap(),
            num_seeds: NonZeroU32::new(1).unwrap(),
            self_links: SelfLinks::default(),
            rng_choice: RngChoice::default()
        }
    }
}

pub fn auto(opt: &AutoOpts, output: &str, disabled_auto_calc: bool, j: Option<NonZeroUsize>)
{
    opt.rng_choice.check_warning();
    match opt.rng_choice{
        RngChoice::Pcg64 => {
            auto_helper::<Pcg64>(opt, output, disabled_auto_calc, j)
        },
        RngChoice::Pcg64Mcg => {
            auto_helper::<Pcg64Mcg>(opt, output, disabled_auto_calc, j)
        },
        RngChoice::XorShift => {
            auto_helper::<Xoshiro256PlusPlus>(opt, output, disabled_auto_calc, j)
        },
        RngChoice::ChaCha20 => {
            auto_helper::<ChaCha20Rng>(opt, output, disabled_auto_calc, j)
        },
        RngChoice::BadRng => {
            auto_helper::<SplitMix64>(opt, output, disabled_auto_calc, j)
        },
        RngChoice::WorstRng => {
            auto_helper::<WorstRng>(opt, output, disabled_auto_calc, j)
        }
    }
}

fn auto_helper<R>(opt: &AutoOpts, output: &str, disabled_auto_calc: bool, j: Option<NonZeroUsize>)
where R: Rng + SeedableRng
{
    if let Some(j) = j {
        rayon::ThreadPoolBuilder::new()
            .num_threads(j.get())
            .build_global()
            .unwrap();
    }
    let fun = opt.self_links.get_step_fun();
    (0..opt.num_seeds.get())
        .into_par_iter()
        .progress()
        .for_each(
            |seed_offset|
            {
                
            
                let mut time_series = vec![0.0; opt.total_steps as usize];
                let seed = opt.mean_opts.seed + seed_offset as u64;
            
                let mut mopt = opt.mean_opts.clone();
                mopt.seed = seed;
                let mut model = SubstitutingMeanField::<R>::new(&mopt);

                (0..opt.samples_per_seed.get())
                    .for_each(
                        |_|
                        {
                            model.reset_delays();
                            let v: Vec<_> = (0..opt.total_steps)
                                .map(
                                    |_|
                                    {
                                        let d = model.average_delay();
                                        fun(&mut model);
                                        d
                                    }
                                ).collect();
                            
                            time_series
                                .iter_mut()
                                .zip(v)
                                .for_each(|(this, other)| *this += other);
                        }
                    );
                time_series.iter_mut()
                    .for_each(|v| *v /= opt.samples_per_seed.get() as f64);
                //remove_mean(&mut time_series);
            
                if !disabled_auto_calc{
                    let auto = cross_correlation(&time_series, &time_series);
                    let auto_2 = cross_correlation_alt(&time_series, &time_series);
                    let mut buf = create_buf_with_command_and_version(output);
                    let header = ["delay", "auto", "auto_pear"];
                    write_slice_head(&mut buf, header).unwrap();
                    for (i, (auto, auto2)) in auto.iter().zip(auto_2).enumerate()
                    {
                        writeln!(buf, "{i} {auto} {auto2}").unwrap();
                    }
                }
                
                
            
                let name = format!("{output}_seed{seed}.time");
                let mut buf = create_buf_with_command_and_version(name);
            
                for (i, v) in time_series.iter().enumerate(){
                    writeln!(buf, "{i} {v}").unwrap();
                }
            }
        )
    
}

