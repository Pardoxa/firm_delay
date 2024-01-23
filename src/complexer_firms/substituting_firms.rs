use indicatif::{ProgressIterator, ParallelProgressIterator};
use rand_distr::{Exp, Distribution, Uniform, Normal};
use rand_pcg::Pcg64;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};
use serde::{Serialize, Deserialize};
use crate::index_sampler::IndexSampler;
use crate::misc::*;
use std::io::{Write, stdout};
use std::num::NonZeroU32;
use std::sync::Mutex;

#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy)]
pub enum SelfLinks{
    #[default]
    AllowSelfLinks,
    NoSelfLinks,
    AlwaysSelfLink
}

impl SelfLinks{
    pub fn get_step_fun(&self) -> fn (&mut SubstitutingMeanField)
    {
        match self{
            Self::AllowSelfLinks => SubstitutingMeanField::step_with_self_links,
            Self::NoSelfLinks => SubstitutingMeanField::step_without_self_links,
            Self::AlwaysSelfLink => SubstitutingMeanField::step_always_self_links
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
    pub sub_dist: PossibleDists
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
            sub_dist: PossibleDists::default()
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
    PowNeg
}

impl PossibleDists{

    pub fn is_reset_fraction_allowed(self) -> bool
    {
        !matches!(self, Self::Beta)
    }

    pub fn gnuplot_x_axis_name(self) -> &'static str
    {
        match self {
            Self::Beta | Self::PowNeg => "α",
            _ => "<p_s>"
        }
    }
   
    pub fn is_const(&self) -> bool
    {
        matches!(self, Self::Const)
    }

    #[allow(clippy::type_complexity)]
    pub fn get_sub_fun<'a>(&'a self) -> Box<dyn Fn (&mut SubstitutingMeanField, f64) + Sync + 'a>
    {
        match self{
            Self::Const =>
            {
                Box::new(|_,_|{})
            },
            Self::Exp => {
                let fun = |model: &mut SubstitutingMeanField, lambda: f64|
                {
                    let dist = Exp::new(lambda).unwrap();
                    model.change_substitution_prob(
                        dist
                    );
                };
                Box::new(fun)
            },
            PossibleDists::UniformAround(uni) => {
                let fun = move |model: &mut SubstitutingMeanField, mid: f64|
                {
                    let dist = uni.get_uniform(mid);
                    model.change_substitution_prob(
                        dist
                    );
                };
                Box::new(fun)
            },
            PossibleDists::Gauss(gauss) => {
                let fun = move |model: &mut SubstitutingMeanField, mean: f64|
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
                let fun = |model: &mut SubstitutingMeanField, alpha: f64|
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
            Self::PowNeg =>
            {
                let uni: Uniform<f64> = Uniform::new_inclusive(0.0, 1.0);
                let fun = move |model: &mut SubstitutingMeanField, alpha: f64|
                {
                    assert!(alpha < 0.0, "Only valid for negative alpha!");
                    assert!(
                        alpha > -1.0, 
                        "Integral does not converge for x^a for a <= 1.0, So it is not a valid probability distribution. Thus: ERROR!"
                    );
                    let exponent = 1.0 /(1.0 + alpha);
                    let dist = uni.map(
                        |val|
                        val.powf(exponent)
                    );
                    model.change_substitution_prob(
                        dist
                    );
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

        let all = [a, b, c, d, e];

        let mut stdout = stdout();
        let msg = "Serialization issue PossibleDists";

        for (idx, dist) in all.iter().enumerate(){
            print_spaces(layer);
            println!("PossibleDists {idx})");
            serde_json::to_writer_pretty(&mut stdout, dist)
                .expect(msg);
            println!();
        }
        
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
    pub fn change_buffers_fun<'a>(&'a self) -> Box<dyn Fn(&mut SubstitutingMeanField, f64) + 'a + Sync>
    {
        match self.dist{
            PossibleDists::Const =>
            {
                Box::new(SubstitutingMeanField::change_buffer_to_const)
            },
            PossibleDists::Exp => {
                self.assert_unequal();
                let fun = |model: &mut SubstitutingMeanField, lambda: f64|
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
                let fun = move |model: &mut SubstitutingMeanField, mid: f64|
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
                let fun = move |model: &mut SubstitutingMeanField, mean: f64|
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
            PossibleDists::Beta | PossibleDists::PowNeg => unimplemented!()
            
        }
    }
}

pub fn sample_velocity_video(opt: &SubstitutionVelocityVideoOpts, out_stub: &str, frametime: u8)
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
    let sub_fun = opt.sub_dist.get_sub_fun();
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
                writeln!(gp, "fit f(x) '{crit_name}' u {using}:2:(t(${using})) yerr via a,b,k,l").unwrap();
            },
            How::Linear => {
                writeln!(gp, "fit f(x) '{crit_name}' u {using}:2:(t(${using})) yerr via a,b").unwrap();
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