use indicatif::{ProgressIterator, ParallelProgressIterator};
use rand_distr::{Exp, Distribution};
use rand_pcg::Pcg64;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};
use serde::{Serialize, Deserialize};
use crate::index_sampler::IndexSampler;
use crate::misc::*;
use std::io::Write;
use std::sync::Mutex;

#[derive(Debug, Serialize, Deserialize, Default)]
pub enum SelfLinks{
    #[default]
    AllowSelfLinks,
    NoSelfLinks
}

impl SelfLinks{
    pub fn get_step_fun(&self) -> fn (&mut SubstitutingMeanField)
    {
        match self{
            Self::AllowSelfLinks => SubstitutingMeanField::step_with_self_links,
            Self::NoSelfLinks => SubstitutingMeanField::step_without_self_links
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
    pub yrange: Option<(f32, f32)>
}

impl Default for SubstitutionVelocityVideoOpts{
    fn default() -> Self {
        Self { 
            buffer: SampleRangeF64::default(), 
            substitution_prob: SampleRangeF64::default(), 
            opts: SubstitutingMeanFieldOpts::default(), 
            time_steps: 1000, 
            self_links: SelfLinks::default(), 
            yrange: Some((0.0,3.5)) 
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct SubstitutionVelocitySampleOpts{
    pub buffer: SampleRangeF64,
    pub opts: SubstitutingMeanFieldOpts,
    pub time_steps: usize,
    pub self_links: SelfLinks
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct SubstitutingMeanFieldOpts{
    buffer: f64,
    substitution_prob: f64,
    seed: u64,
    k: usize,
    n: usize,
    lambda: f64
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

pub struct SubstitutingMeanField{
    current_delays: Vec<f64>,
    buffers: Vec<f64>,
    substitution_prob: Vec<f64>,
    next_delays: Vec<f64>,
    k: usize,
    rng: Pcg64,
    index_sampler: IndexSampler,
    dist: Exp<f64>
}

impl SubstitutingMeanField{

    pub fn reset_delays(&mut self)
    {
        self.current_delays.iter_mut().for_each(|v| *v = 0.0);
    }

    pub fn change_buffer_to_const(&mut self, const_val: f64)
    {
        self.buffers.iter_mut()
            .for_each(|v| *v = const_val)
    }

    pub fn get_k(&self) -> usize {
        self.k
    }

    pub fn new(opt: &SubstitutingMeanFieldOpts) -> Self
    {
        let current_delays = vec![0.0; opt.n];
        let next_delays = vec![0.0; opt.n];
        let mut rng = Pcg64::seed_from_u64(opt.seed);
        let index_sampler = IndexSampler::measure_which(
            opt.n, 
            opt.k, 
            &mut rng
        );
        let exp = Exp::new(opt.lambda)
            .unwrap();
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
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + self.dist.sample(&mut self.rng);
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
                        *n_delay = (current - self.buffers[index]).max(0.0) 
                            + self.dist.sample(&mut self.rng);
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
    let name = format!("{out_stub}.dat");
    let mut writer = create_buf_with_command_and_version(name);
    let header = ["B", "Velocity"];
    write_slice_head(&mut writer, header).unwrap();

    let mut model = SubstitutingMeanField::new(&opt.opts);

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

pub fn sample_velocity_video(opt: &SubstitutionVelocityVideoOpts, out_stub: &str, frametime: u8)
{
    let fun = opt.self_links.get_step_fun();

    let all_sub_probs: Vec<_> = opt.substitution_prob
        .get_iter()
        .collect();

    let zeros = "000000000";

    let cleaner = Cleaner::new();

    let bar = crate::misc::indication_bar(all_sub_probs.len() as u64);

    all_sub_probs.par_iter()
        .enumerate()
        .progress_with(bar)
        .for_each(
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
                    model.change_buffer_to_const(b);
                    model.reset_delays();
                    for _ in 0..opt.time_steps{
                        fun(&mut model);
                    }
                    let velocity = model.average_delay() / opt.time_steps as f64;
                    writeln!(writer, "{b} {velocity}").unwrap();
                }
                drop(writer);
                let gp_name = format!("{stub}.gp");
                let mut gp_writer = create_gnuplot_buf(&gp_name);
                let png = format!("{stub}.png");
                writeln!(gp_writer, "set t pngcairo").unwrap();
                writeln!(gp_writer, "set output '{png}'").unwrap();
                writeln!(gp_writer, "set ylabel 'v'").unwrap();
                writeln!(gp_writer, "set xlabel 'B'").unwrap();
                writeln!(gp_writer, "set label 'p={sub_prob}' at screen 0.4,0.9").unwrap();
                if let Some((min, max)) = &opt.yrange{
                    writeln!(gp_writer, "set yrange [{min}:{max}]").unwrap();
                }
                writeln!(gp_writer, "p '{w_name}' t ''").unwrap();
                writeln!(gp_writer, "set output").unwrap();
                drop(gp_writer);
                call_gnuplot(&gp_name);
                
                cleaner.add_multi([w_name, gp_name, png])
                
            }
        );
    create_video("TMP_*.png", out_stub, frametime);
    cleaner.clean();
}