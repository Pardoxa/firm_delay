use indicatif::ParallelProgressIterator;
use rand_distr::{Exp, Distribution};
use rand_pcg::Pcg64;
use rand::{Rng, SeedableRng};
use rayon::iter::{IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator};
use crate::index_sampler::IndexSampler;
use crate::misc::*;
use std::io::Write;
use super::{
    SelfLinks, 
    SubstitutionVelocityVideoOpts, 
    SubstitutingMeanFieldOpts,
    Cleaner,
    PossibleDists
};


pub struct SubstitutingQuenchedMeanField{
    current_delays: Vec<f64>,
    buffers: Vec<f64>,
    substitution_prob: Vec<f64>,
    next_delays: Vec<f64>,
    k: usize,
    rng: Pcg64,
    index_sampler: IndexSampler,
    dist: Exp<f64>,
    links: Vec<Vec<usize>>
}

impl SubstitutingQuenchedMeanField{
    fn choose_links(&mut self, self_links: SelfLinks)
    {
        match self_links{
            SelfLinks::AllowSelfLinks => {
                self.links
                    .iter_mut()
                    .for_each(
                        |vec|
                        {
                            vec.clear();
                            let iter = self.index_sampler
                                .sample_inplace_amount(&mut self.rng, self.k)
                                .iter().map(|val| *val as usize);
                            vec.extend(iter);
                        }
                    );
            },
            SelfLinks::NoSelfLinks =>
            {
                self.links
                    .iter_mut()
                    .zip(0..)
                    .for_each(
                        |(vec, index)|
                        {
                            vec.clear();
                            let iter = self.index_sampler
                                .sample_inplace_amount_without(&mut self.rng, index, self.k)
                                .iter().map(|val| *val as usize);
                            vec.extend(iter);
                        }
                    );
            },
            SelfLinks::AlwaysSelfLink =>
            {
                unimplemented!()
            },
            SelfLinks::AllowMultiple => {
                unimplemented!()
            }
        }
        
    }

    pub fn reseed_sub_prob(
        &mut self, 
        sub_prob: f64, 
        fraction: f64
    )
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

    pub fn get_k(&self) -> usize {
        self.k
    }

    pub fn new(opt: &SubstitutingMeanFieldOpts, self_links: SelfLinks) -> Self
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
        let mut s = Self{
            current_delays,
            next_delays,
            buffers: opt.get_buffers(),
            k: opt.k,
            index_sampler,
            substitution_prob: opt.get_substitution_prob(),
            rng,
            dist: exp,
            links: vec![Vec::new(); opt.n]
        };
        s.choose_links(self_links);
        s
    }


    pub fn step(&mut self)
    {
        self.next_delays
            .iter_mut()
            .enumerate()
            .for_each(
                |(index, n_delay)|
                {
                    if self.rng.gen::<f64>() < self.substitution_prob[index]{
                        *n_delay = self.dist.sample(&mut self.rng);
                    } else {
                        let mut current = 0.0_f64;
                        for &i in self.links[index].iter(){
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

pub fn sample_velocity_video(opt: &SubstitutionVelocityVideoOpts, out_stub: &str, frametime: u8)
{

    assert!(
        matches!(
            opt.buffer_dist.dist,
            PossibleDists::Const
        ),
        "ERROR: Only const buffer dist is implemented for Quenched. Others are implemented for non-quenched. If you want to use it for quenched, look there and copy the nessessary parts over"
    );

    let all_sub_probs: Vec<_> = opt.substitution_prob
        .get_iter()
        .collect();

    let zeros = "000000000";

    let cleaner = Cleaner::new();

    let bar = crate::misc::indication_bar(all_sub_probs.len() as u64);

    let criticals: Vec<_> = all_sub_probs
        .par_iter()
        .enumerate()
        .filter_map(
            |(index, &sub_prob)|
            {
                let mut model_opt = opt.opts.clone();
                model_opt.substitution_prob = sub_prob;
                model_opt.seed = index as u64;

                let mut model = SubstitutingQuenchedMeanField::new(
                    &model_opt, 
                    opt.self_links
                );

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
                                model.change_buffer_to_const(b);
                                model.choose_links(opt.self_links);
                                if let Some(f) = opt.reset_fraction{
                                    model.reseed_sub_prob(
                                        sub_prob, 
                                        f
                                    );
                                }
                                model.reset_delays();
                                for _ in 0..opt.time_steps{
                                    model.step();
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
        writeln!(buf, "{} {} {} {}", s[0], s[1], s[2], s[3]).unwrap();
    }
    drop(buf);
    enum How{
        Linear,
        Complex
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
                writeln!(gp, "f(x)= a*x+b+k*x**l")
            },
            How::Linear => {
                writeln!(gp, "f(x)= a*x+b")
            }
        }.unwrap();
        
        writeln!(gp, "g(x)= c*x+d").unwrap();
    
        let using = if let Some(f) = opt.reset_fraction{
            writeln!(gp, "set xlabel 'p_s f'").unwrap();
            
            format!("($1*{f})")
        } else {
            writeln!(gp, "set xlabel 'p_s'").unwrap();
            
            "1".to_owned()
        };
        match how{
            How::Complex => {
                writeln!(gp, "fit f(x) '{crit_name}' u {using}:2 via a,b,k,l")
            },
            How::Linear => {
                writeln!(gp, "fit f(x) '{crit_name}' u {using}:2 via a,b")
            }
        }.unwrap();
        
        writeln!(gp, "fit g(x) '{crit_name}' u {using}:3 via c,d").unwrap();
        writeln!(gp, "h(x)=-g(x)/f(x)").unwrap();
        
        writeln!(
            gp, 
            "p '{crit_name}' u {using}:2 t 'a', '' u {using}:3 t 'b', '' u {using}:4 t 'Crit B', f(x) t 'fit a', g(x) t 'fit b', h(x) t 'approx'"
        ).unwrap();
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
            eprintln!("RECOVERY also failed :(");
        } else {
            eprintln!("RECOVERY SUCCESS!");
        }
    }

    create_video("TMP_*.png", out_stub, frametime);
    cleaner.clean();
}