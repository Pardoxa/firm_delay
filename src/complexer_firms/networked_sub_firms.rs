use {
    super::{network_helper::ImpactNetworkHelper, substituting_firms::*}, 
    crate::misc::*, 
    clap::{Parser, Subcommand}, 
    indicatif::ParallelProgressIterator, 
    rand::{seq::SliceRandom, Rng, RngCore, SeedableRng}, 
    rand_chacha::ChaCha20Rng, 
    rand_distr::{Distribution, Uniform}, 
    rand_pcg::{Pcg64, Pcg64Mcg}, 
    rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus}, 
    rayon::prelude::*, 
    std::{io::Write, sync::RwLock}
};

static GLOBAL_NETWORK: RwLock<Vec<Vec<usize>>> = RwLock::new(Vec::new());

fn set_global_network(k: usize, n: usize, iterations: usize, markov_steps: usize)
{
    let mut impact = ImpactNetworkHelper::new_both_dirs(k, n);
    impact.rebuild_both_dirs(iterations);
    impact.markov_greed(markov_steps);
    let network = impact.into_inner_network();
    let mut lock = GLOBAL_NETWORK.write().unwrap();
    *lock = network;
    drop(lock);
}

pub struct NetworkedSubFirms<R>{
    firms: SubstitutingMeanField<R>,
    pub network: Vec<Vec<usize>>
}

impl<R> NetworkedSubFirms<R>
where R: Rng + RngCore
{
    pub fn new_empty(firms: SubstitutingMeanField<R>) -> Self
    {
        let n = firms.current_delays.len();
        let network = vec![Vec::with_capacity(firms.k); n];
        Self { firms, network }
    }
    
    fn clear_network(&mut self)
    {
        self.network.iter_mut().for_each(Vec::clear);
    }

    pub fn independent_hub(&mut self)
    {
        self.clear_network();
        let k = self.firms.k;
        self.network[k..]
            .iter_mut()
            .for_each(
                |adj|
                {
                    adj.extend(0..k)
                }
            );
    }

    fn add_self_links_unchecked(&mut self)
    {
        self.network
            .iter_mut()
            .enumerate()
            .for_each(|(index, adj)| adj.push(index));
    }

    pub fn line(&mut self)
    {
        self.clear_network();
        let lenm1 = self.network.len() - 1;
        self.network[..lenm1]
            .iter_mut()
            .zip(1..)
            .for_each(
                |(adj, idx)|
                {
                    adj.push(idx);
                }
            )
    }

    fn line_like(&mut self)
    {
        self.clear_network();
        let k = self.firms.k;
        let len = self.network.len();
        self.network[..len-k]
            .iter_mut()
            .enumerate()
            .for_each(
                |(idx, adj)|
                {
                    adj.extend(idx..idx+k);
                }
            );
        self.network[len-k..]
            .iter_mut()
            .for_each(
                |adj|
                {
                    adj.extend(len-k..len)
                }
            );
    }

    pub fn loop_network(&mut self, frac: f64){
        self.line_like();
        if frac > 0.0 {
            let k = self.firms.k;
            let n = self.network.len();
            let mut possibility: Vec<_> = (k..n-k)
                .flat_map(
                    |i|
                    {
                        (1..k)
                            .map(
                                move |o|
                                {
                                    (i, o)
                                }
                            )
                    }
                ).collect();
            possibility.shuffle(&mut self.firms.rng);
            let bound = (frac * n as f64).floor() as usize;
            possibility[0..bound]
                .iter()
                .for_each(
                    |&(i, o)|
                    {
                        loop{
                            let new_val = self.firms.rng.gen_range(0..i);
                            if !self.network[i].contains(&new_val){
                                self.network[i][o] = new_val;
                                break;
                            }
                        }
                    }
                )
        }
    }

    pub fn separated_complete_graphs(&mut self)
    {
        self.clear_network();
        let k = self.firms.k;
        let mut iter = self.network
            .chunks_exact_mut(k);
        let mut counter = 0;
        (&mut iter)
            .for_each(
                |chunk|
                {
                    let start = counter * k;
                    let end = start + k;
                    chunk.iter_mut()
                        .for_each(|adj| adj.extend(start..end));
                    counter += 1;
                }
            );
        let rest = iter.into_remainder();
        let start = counter * k;
        let amount = rest.len();
        let end = start + amount;
        rest.iter_mut()
            .for_each(
                |adj| adj.extend(start..end)
            );
    }

    pub fn small_world(&mut self, p: f64, offset: isize)
    {
        self.make_ring(offset);
        let k = self.firms.k;
        let uni = Uniform::new(0, self.network.len());
        self.network
            .iter_mut()
            .for_each(
                |adj|
                {
                    // randomly remove nodes
                    (0..adj.len())
                        .rev()
                        .for_each(
                            |i|
                            {
                                if self.firms.rng.gen::<f64>() < p {
                                    adj.swap_remove(i);
                                }
                            }
                        );
                    // randomly add nodes
                    while adj.len() < k {
                        let idx = uni.sample(&mut self.firms.rng);
                        if !adj.contains(&idx)
                        {
                            adj.push(idx);
                        }
                    }
                }
            );
    }

    pub fn independent_hub_self_links(&mut self)
    {
        self.independent_hub();
        self.add_self_links_unchecked();
    }

    pub fn dependent_hub(&mut self)
    {
        let k = self.firms.k;
        self.independent_hub();
        self.network[..k]
            .iter_mut()
            .enumerate()
            .for_each(
                |(index, adj)|
                {
                    adj.extend(0..index);
                    adj.extend(index+1..k);
                }
            )
    }

    pub fn random_network(&mut self)
    {
        self.network
            .iter_mut()
            .for_each(
                |adj|
                {
                    adj.clear();
                    adj.extend(
                        self.firms
                            .index_sampler
                            .sample_indices(&mut self.firms.rng)
                            .iter()
                            .map(|&val| val as usize)
                    )
                }
            );
    }

    pub fn impact_network(&mut self)
    {
        self.clear_network();
        self.network
            .iter_mut()
            .enumerate()
            .for_each(
                |(idx, adj)|
                {
                    adj.push(idx);
                }
            );
        
    }

    pub fn make_ring(&mut self, offset: isize)
    {
        self.clear_network();
        let k = self.firms.k;
        let i_len = self.network.len() as isize;

        self.network
            .iter_mut()
            .zip(0_isize..)
            .for_each(
                |(adj, index)|
                {
                    adj.clear();
                    adj.extend(
                        (index..(index+k as isize))
                            .map(
                                |idx|
                                {
                                    let val = idx + offset + i_len;
                                    (val % i_len) as usize
                                }
                            )
                    );
                }
            )
    }

    pub fn step(&mut self)
    {
        self.firms
            .next_delays
            .iter_mut()
            .enumerate()
            .for_each(
                |(index, n_delay)|
                {
                    if self.firms.rng.gen::<f64>() < self.firms.substitution_prob[index]{
                        *n_delay = self.firms.dist.sample(&mut self.firms.rng);
                    } else {
                        let mut current = 0.0_f64;
                        for &i in &self.network[index]{
                            current = current.max(self.firms.current_delays[i]);
                        }
                        let e_sample: f64 = self.firms
                            .dist
                            .sample(&mut self.firms.rng);
                        *n_delay = (current - self.firms.buffers[index]).max(0.0) 
                            + e_sample;
                    }
                    
                }
            );
        std::mem::swap(&mut self.firms.current_delays, &mut self.firms.next_delays);
    }
}

#[derive(Debug, Clone, Copy, Parser, PartialEq, Eq)]
pub struct RingOpt{
    /// Offset for ring
    offset: isize
}

#[derive(Debug, Clone, Copy, Parser, PartialEq)]
pub struct RandomizedRingOpt{
    /// Offset for ring
    offset: isize,

    /// rewire probability
    p: f64
}


#[derive(Debug, Clone, Copy, Parser, PartialEq)]
pub struct LoopTest{
    p: f64
}

#[derive(Debug, Clone, Copy, Parser, PartialEq)]
pub struct DistHeur{
    iterations: usize,
    markov_steps: usize
}

#[derive(Subcommand, Debug, Clone, PartialEq)]
/// Which network structure to use?
pub enum NetworkStructure{
    /// Ring structure for network
    Ring(RingOpt),
    /// Ring structure but randomized
    RandomizedRing(RandomizedRingOpt),
    /// Random network
    Random,
    /// K nodes depend on nothing, everything else depends on these k nodes. NO SELF LINKS
    IndependentHub,
    /// K nodes depend on nothing, everything else depends on these k nodes. Every node has a self link
    IndependentHubSelfLinks,
    /// K nodes depend on each other, everything else depends on these k nodes. NO SELF LINKS
    DependentHub,
    /// Line, just a test
    Line,
    /// Loop tests
    LoopTest(LoopTest),
    /// Complete graphs that are separated from one another
    CompleteChunks,
    /// Using a distance Heuristic
    DistanceHeuristic(DistHeur)
}

pub fn sample_ring_velocity_video(
    opt: &SubstitutionVelocityVideoOpts, 
    out_stub: &str, 
    frametime: u8, 
    no_clean: bool,
    convert_video: bool,
    structure: NetworkStructure
)
{
    opt.rng_choice.check_warning();
    match opt.rng_choice{
        RngChoice::Pcg64 => {
            sample_network_velocity_video_helper::<Pcg64>(opt, out_stub, frametime, no_clean, convert_video, structure)
        },
        RngChoice::Pcg64Mcg => {
            sample_network_velocity_video_helper::<Pcg64Mcg>(opt, out_stub, frametime, no_clean, convert_video, structure)
        },
        RngChoice::XorShift => {
            sample_network_velocity_video_helper::<Xoshiro256PlusPlus>(opt, out_stub, frametime, no_clean, convert_video, structure)
        },
        RngChoice::ChaCha20 => {
            sample_network_velocity_video_helper::<ChaCha20Rng>(opt, out_stub, frametime, no_clean, convert_video, structure)
        },
        RngChoice::BadRng => {
            sample_network_velocity_video_helper::<SplitMix64>(opt, out_stub, frametime, no_clean, convert_video, structure)
        },
        RngChoice::WorstRng => {
            sample_network_velocity_video_helper::<WorstRng>(opt, out_stub, frametime, no_clean, convert_video, structure)
        }
    }
}

fn sample_network_velocity_video_helper<R>(
    opt: &SubstitutionVelocityVideoOpts, 
    out_stub: &str, 
    frametime: u8, 
    no_clean: bool,
    convert_video: bool,
    structure: NetworkStructure
)
where R: Rng + SeedableRng + 'static
{
    if opt.reset_fraction.is_some(){
        assert!(opt.sub_dist.is_reset_fraction_allowed());
    }
    
    if !opt.sub_dist.is_const(){
        assert!(
            opt.reset_fraction.is_none(),
            "Reset fraction not implemented with sub dists"
        );
    }
    let sub_fun = opt.sub_dist.get_sub_fun::<R>();

    let (iter, what_type) = opt.get_iter_help(); 
    

    let zeros = "000000000";

    let cleaner = Cleaner::new();

    let bar = crate::misc::indication_bar(iter.len() as u64);

    if let NetworkStructure::DistanceHeuristic(d) = &structure
    {
        set_global_network(opt.opts.k, opt.opts.n, d.iterations, d.markov_steps);
    }

    

    let criticals: Vec<_> = iter
        .par_iter()
        .enumerate()
        .filter_map(
            |(index, &item)|
            {
                let mut model_opt = opt.opts.clone();
                if let ChangeType::SubProb = what_type{
                    model_opt.substitution_prob = item;
                }
                
                model_opt.seed = index as u64;

                let model = SubstitutingMeanField::new(&model_opt);
                let mut network_model = NetworkedSubFirms::new_empty(model);
                match &structure{
                    NetworkStructure::Ring(offset) => network_model.make_ring(offset.offset),
                    NetworkStructure::Random => network_model.random_network(),
                    NetworkStructure::IndependentHub => network_model.independent_hub(),
                    NetworkStructure::IndependentHubSelfLinks => network_model.independent_hub_self_links(),
                    NetworkStructure::DependentHub => network_model.dependent_hub(),
                    NetworkStructure::RandomizedRing(opt) => network_model.small_world(opt.p, opt.offset),
                    NetworkStructure::Line => network_model.line(),
                    NetworkStructure::LoopTest(opt) => network_model.loop_network(opt.p),
                    NetworkStructure::CompleteChunks => network_model.separated_complete_graphs(),
                    NetworkStructure::DistanceHeuristic(_) => {
                        let global = GLOBAL_NETWORK.read().unwrap();
                        network_model.network = global.clone();
                        drop(global);
                    }
                }

                let i_name = index.to_string();
                let start = i_name.len();
                let zeros = &zeros[start..];
                let stub = format!("TMP_{zeros}{i_name}{out_stub}");
                let w_name = format!("{stub}.dat");
                let mut writer = create_buf(&w_name);

                let sub_prob = match what_type{
                    ChangeType::SubProb =>{
                        item
                    },
                    ChangeType::DistUniMid => {
                        opt.opts.substitution_prob
                    }
                };

                for b in opt.buffer.get_iter(){
                    let mut velocity_sum = 0.0;
                    let change_buffers_fun: CBufferFun<R> = if let ChangeType::DistUniMid = what_type
                    {
                        match &opt.buffer_dist.dist {
                            PossibleDists::UniformAround(around) => {
                                let mut uni = *around;
                                uni.interval_length_half = item;
                                let dist = PossibleDists::UniformAround(uni);
                                let dist = BufferDist{
                                    min: opt.buffer_dist.min,
                                    max: opt.buffer_dist.max,
                                    dist
                                };
                                dist.change_buffers_fun()
                            },
                            _ => unreachable!()
                        }
                    } else {
                        opt.buffer_dist.change_buffers_fun()
                    };
                    
                    (0..opt.samples_per_point.get())
                        .for_each(
                            |_|
                            {
                                change_buffers_fun(&mut network_model.firms, b);
                                

                                match opt.reset_fraction{
                                    Some(f) => {
                                        network_model.firms.reseed_sub_prob(sub_prob, f);
                                    },
                                    None => {
                                        // Sub fun currently not compatible with reset_fraction
                                        sub_fun(&mut network_model.firms, sub_prob);
                                    }
                                }
                                
                                network_model.firms.reset_delays();
                                
                                for _ in 0..opt.time_steps{
                                    network_model.step();
                                }
                                let velocity = network_model.firms.average_delay() / opt.time_steps as f64;
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
                match what_type{
                    ChangeType::SubProb => {
                        writeln!(gp_writer, "set label 'p={sub_prob}' at screen 0.4,0.9").unwrap();
                    },
                    ChangeType::DistUniMid => {
                        writeln!(gp_writer, "set label '{DBSTRING}={item}' at screen 0.4,0.9").unwrap();
                    }
                }
                
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
                    let first = match what_type{
                        ChangeType::SubProb => {
                            sub_prob
                        },
                        ChangeType::DistUniMid => {
                            item
                        }
                    };
                    Some([first, a, b, crit])
                } else {
                    None
                }
                
            }
        ).progress_with(bar)
        .collect();

    let crit_stub = format!("{out_stub}_crit");
    let crit_name = format!("{crit_stub}.dat");
    let mut buf = create_buf_with_command_and_version(&crit_name);
    let mut header = Vec::new();
    let s = match what_type
    {
        ChangeType::SubProb => {
            header.push("sub_prob");
            opt.sub_dist.gnuplot_x_axis_name()
        }, 
        ChangeType::DistUniMid => {
            header.push("half_width_buf_dist");
            DBSTRING
        }
    };
    header.extend_from_slice(&["a", "b", "critical"]);
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
        let start_idx = match what_type{
            ChangeType::SubProb => {
                1
            },
            ChangeType::DistUniMid => {
                3
            }
        };

        for &val in criticals.iter().flat_map(|slice| &slice[start_idx..]){
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

        match what_type{
            ChangeType::SubProb => {
                write!(
                    gp, 
                    "p '{crit_name}' u {using}:2 t 'a', '' u {using}:3 t 'b', '' u {using}:4 t 'Crit B'"
                ).unwrap();
            },
            ChangeType::DistUniMid => {
                write!(
                    gp, 
                    "p '{crit_name}' u {using}:4 t 'Crit B'"
                ).unwrap();
            }
        }
        
        
        if how.is_no_fit(){
            writeln!(gp)
        } else {
            writeln!(gp, ", f(x) t 'fit a', g(x) t 'fit b', h(x) t 'approx'")
        }.unwrap();
        writeln!(gp, "set output").unwrap();
        drop(gp);
        crit_gp
    };

    let how = match what_type{
        ChangeType::SubProb => How::Complex,
        ChangeType::DistUniMid => How::NoFit
    };
    let crit_gp = crit_gp_write(how);
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

    create_video(
        "TMP_*.png", 
        out_stub, 
        frametime,
        convert_video
    );
    if !no_clean{
        cleaner.clean();
    }
}