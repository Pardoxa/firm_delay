use {
    super::{network_helper::{write_digraph, ImpactNetworkHelper}, substituting_firms::*}, crate::misc::*, camino::Utf8PathBuf, clap::{Parser, Subcommand}, indicatif::ParallelProgressIterator, itertools::Itertools, rand::{seq::SliceRandom, Rng, RngCore, SeedableRng}, rand_chacha::ChaCha20Rng, rand_distr::{Distribution, Uniform}, rand_pcg::{Pcg64, Pcg64Mcg}, rand_xoshiro::{SplitMix64, Xoshiro256PlusPlus}, rayon::prelude::*, std::{collections::BTreeSet, io::Write}
};

// Not actually what I originally planned, but it leads to interesting 
// results, so I'm keeping it.
//
// Note: Currently k != 5. Even if everything worked like I originally
// planned it would have been k == 4, but the way it is implemented 
// now, k is decreasing with rec_count.
fn recursive_interweaving_k5(rec_count: u8) -> Vec<Vec<usize>>
{
    let mut origin = (0_usize..4)
        .map(
            |i|
            {
                (0..i).chain(i+1..4)
                    .collect_vec()
            }
        ).collect_vec();
    for _ in 0..rec_count
    {
        let new_len = origin.len() * 4;
        let mut next_nodes = vec![Vec::with_capacity(5); new_len];
        let mut next_node_count = origin.len();
        for (idx, adj) in origin.iter().enumerate()
        {
            for &j in adj {
                if j > idx {
                    let k1 = next_node_count;
                    let k2 = next_node_count + 1;
                    next_node_count += 2;
                    let j_adj = &mut next_nodes[j];
                    j_adj.push(k1);
                    let i_adj = &mut next_nodes[idx];
                    i_adj.push(k2);
                    let k1_adj = &mut next_nodes[k1];
                    k1_adj.extend_from_slice(&[idx, j, k2]);
                    let k2_adj = &mut next_nodes[k2];
                    k2_adj.extend_from_slice(&[idx, j, k1]);
                }
            }
        }
        origin = next_nodes;
    }
    origin.iter_mut()
        .enumerate()
        .for_each(
            |(idx, adj)| adj.push(idx)
        );
    origin
}

fn get_heur_network(k: usize, n: usize, d: &DistHeur) -> Vec<Vec<usize>>
{
    let mut impact = ImpactNetworkHelper::new_both_dirs(k, n);
    impact.rebuild_both_dirs(d.iterations);
    impact.markov_greed(d.markov_steps, d.step_size);
    let network = impact.into_inner_network();
    let correct_dim = network
        .iter()
        .map(|adj| adj.len())
        .all(|len| len == k);
    assert!(correct_dim);
    network
}

pub struct NetworkedSubFirms<R>{
    firms: SubstitutingMeanField<R>,
    pub network: Vec<Vec<usize>>
}

fn ratios_to_layer_count(r: &[f64], len: usize) -> Vec<usize>
{
    let sum: f64 = r.iter().copied().sum();
    let recip = sum.recip();
    let percentages = r.iter()
        .copied()
        .map(|r| r * recip)
        .collect_vec();
    let len_f = len as f64;
    let mut layer = percentages.iter()
        .map(|&p| (p * len_f).floor() as usize)
        .collect_vec();
    let mut total: usize = layer.iter().sum();
    while total < len {
        let mut idx_max_missmatch = 0;
        let mut max_missmatch = f64::NEG_INFINITY;
        // this can be done more efficient, but I don't care as this is only called once per simulation
        for i in 0..layer.len()
        {
            let current = layer[i] as f64 / len_f;
            let missmatch = percentages[i] - current;
            if missmatch > max_missmatch{
                max_missmatch = missmatch;
                idx_max_missmatch = i;
            }
        }
        layer[idx_max_missmatch] += 1;
        total += 1;
    }
    layer
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

    fn tree_like_network(&mut self, ratios: &[f64])
    {
        let k = self.firms.k;
        self.clear_network();
        let layer_count = ratios_to_layer_count(ratios, self.network.len());
        assert!(
            layer_count[0] >= self.firms.k,
            "First layer needs to consist of at least k firms!"
        );
        let mut nodes = self.network.as_mut_slice();
        // first nodes are of layer 0. They are only connected to themselves!
        let mut current_layer;
        let mut layer_iter = layer_count.iter();
        let count = *layer_iter.next().unwrap();
        (current_layer, nodes) = nodes.split_at_mut(count);
        for (idx, node) in current_layer.iter_mut().enumerate()
        {
            node.push(idx);
        }
        let mut end = count;
        // All other nodes are connected to k-1 nodes in layers above + to themselves
        let km1 = k - 1;
        for &count in layer_iter
        {
            (current_layer, nodes) = nodes.split_at_mut(count);
            let mut set = BTreeSet::new();
            let uni = Uniform::new(0, end);
            for (node, idx) in current_layer.iter_mut().zip(end..)
            {
                set.clear();
                while set.len() != km1 {
                    set.extend(
                        uni.sample_iter(&mut self.firms.rng)
                            .take(km1 - set.len())
                    );
                }
                node.extend(
                    set.iter()
                );
                node.push(idx);
            }
            end += count;
        }
    }

    fn tree_like_network_var(&mut self, ratios: &[f64])
    {
        let k = self.firms.k;
        self.clear_network();
        let layer_count = ratios_to_layer_count(ratios, self.network.len());
        assert!(
            layer_count[0] >= self.firms.k,
            "First layer needs to consist of at least k firms!"
        );
        let mut nodes = self.network.as_mut_slice();
        // first nodes are of layer 0. They are only connected to themselves!
        let mut current_layer;
        let mut layer_iter = layer_count.iter();
        let count = *layer_iter.next().unwrap();
        (current_layer, nodes) = nodes.split_at_mut(count);
        for (idx, node) in current_layer.iter_mut().enumerate()
        {
            node.push(idx);
        }
        let mut end = count;
        // All other nodes are connected to k-1 nodes in their + layer above and also to themselves
        for &count in layer_iter
        {
            (current_layer, nodes) = nodes.split_at_mut(count);
            let mut set = BTreeSet::new();
            let uni = Uniform::new(0, end + count);
            for (node, idx) in current_layer.iter_mut().zip(end..)
            {
                set.clear();
                set.insert(idx);
                while set.len() != k {
                    set.extend(
                        uni.sample_iter(&mut self.firms.rng)
                            .take(k - set.len())
                    );
                }
                node.extend(
                    set.iter()
                );
            }
            end += count;
        }
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

#[derive(Debug, Clone, Parser, PartialEq)]
pub struct TreeLikeNetwork{
    pub ratios: Vec<f64>
}

#[derive(Debug, Clone, Parser, PartialEq)]
pub struct RecursiveK5{
    pub recursions: u8,

    /// write the network
    #[arg(long, short)]
    pub dot_file: Option<Utf8PathBuf>
}


#[derive(Debug, Clone, Copy, Parser, PartialEq)]
pub struct LoopTest{
    p: f64
}

#[derive(Debug, Clone, Copy, Parser, PartialEq)]
pub struct DistHeur{
    iterations: usize,
    markov_steps: usize,
    #[arg(default_value_t=1)]
    step_size: usize
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
    DistanceHeuristic(DistHeur),
    /// Tree like networks
    TreeLike(TreeLikeNetwork),
    /// Variant of tree like where each node may also be connected to nodes on the same layer 
    /// (except for layer 0, which is only connected to themselves)
    TreeLikeVar(TreeLikeNetwork),
    /// Recursively created highly connected graph. Ignores N. k has to be 5
    RecursiveK5(RecursiveK5)
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
    let (opt, network) = match &structure
    {
        NetworkStructure::RecursiveK5(r) => {
            let n = 4_usize.pow(r.recursions as u32 + 1);
            let mut opt: SubstitutionVelocityVideoOpts = opt.clone();
            opt.opts.n = n;
            println!("Setting n to {n}");
            assert_eq!(
                opt.opts.k,
                5,
                "K needs to be five here!"
            );
            let network = recursive_interweaving_k5(r.recursions);
            let total_edges = network.iter()
                .map(|adj| adj.len())
                .sum::<usize>();
            let average_edges = total_edges as f64 / network.len() as f64;
            println!("Average edges: {average_edges}");
            if let Some(path) = &r.dot_file 
            {
                let writer = create_buf_with_command_and_version(path);
                write_digraph(writer, &network);
            }
            (opt, Some(network))
        },
        _ => (opt.to_owned(), None)
    };
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

    let g_network = match &structure{
        NetworkStructure::DistanceHeuristic(d) => {
            Some(get_heur_network(opt.opts.k, opt.opts.n, d))
        },
        _ => None
    };

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

                let model = match &structure{
                    NetworkStructure::RecursiveK5(_) =>  {
                        SubstitutingMeanField::new_empty_index_sampler(&model_opt)
                    },
                    _ => SubstitutingMeanField::new(&model_opt)
                };
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
                        let network = g_network.as_ref().unwrap();
                        network_model.network.clone_from(network);
                    },
                    NetworkStructure::TreeLike(opt) => {
                        network_model.tree_like_network(&opt.ratios)
                    },
                    NetworkStructure::TreeLikeVar(opt) => {
                        network_model.tree_like_network_var(&opt.ratios)
                    },
                    NetworkStructure::RecursiveK5(_) => {
                        match &network{
                            Some(network) => {
                                network_model.network.clone_from(network);
                            },
                            None => unreachable!()
                        }
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