use std::{io::Write, num::*, path::Path};
use camino::Utf8PathBuf;
use itertools::Itertools;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use sampling::HistF64;
use super::{opts::*, model::*};
use crate::misc::*;

const ZEROS: &str = "000000000";

pub fn test_profile(opt: ChainProfileOpts, out: Utf8PathBuf)
{
    let chain_len = opt.total_len.get() - 1;
    let rng = Pcg64::seed_from_u64(opt.seed);
    let mut model = Model::new_multi_chain_from_rng(
        opt.num_chains,
        chain_len,
        rng, 
        opt.root_demand, 
        opt.max_stock
    );
    let mut header = vec![
        "time_step".to_string(),
        "d0_c0".to_owned(),
        "I0_c0".to_owned(),
        "k0_c0".to_owned(),
        "a0_c0".to_owned()
    ];
    for j in 1..opt.num_chains.get(){
        header.push(format!("k0_c{j}"));
        header.push(format!("a0_c{j}"));
    }
    for j in 0..opt.num_chains.get(){
        let addition = format!("_c{j}");
        for i in 1..=chain_len{
            header.push(format!("d{i}{addition}"));
            header.push(format!("I{i}{addition}"));
            header.push(format!("k{i}{addition}"));
            header.push(format!("a{i}{addition}"));
        }
    }
    let mut averages = vec![vec![0.0;header.len()-1]; opt.time_steps.get() as usize];

    for _ in 0..opt.average_over_samples.get(){
        model.reset_delays();
        for av_slice in averages.iter_mut(){

            model.update_demand();
            model.update_production();

            let mut iter = av_slice.iter_mut();
    
            for i in 0..model.stock_avail.len(){
                *iter.next().unwrap() += model.current_demand[i];
                *iter.next().unwrap() += model.currently_produced[i];
                let slice = &model.stock_avail[i];
                if slice.is_empty(){
                    *iter.next().unwrap() = f64::NAN;
                    *iter.next().unwrap() = f64::NAN;
                } else {
                    for s in slice {
                        *iter.next().unwrap() += s.stock;
                        *iter.next().unwrap() += s.currently_avail;
                    }
                }
            }
        }
    }

    let factor = 1.0 / opt.average_over_samples.get() as f64;
    if !opt.output_only_production_profile{
        let mut buf = create_buf_with_command_and_version_and_header(&out, header);
        
        for (line, t) in averages.iter().zip(1..)
        {
            write!(buf, "{t}").unwrap();
            for val in line {
                let val = val * factor;
                write!(buf, " {val}").unwrap();
            }
            writeln!(buf).unwrap();
        }
    }

    let header = [
        "Distance_from_leaf",
        "I",
        "k",
        "D"
    ];
    let name = format!("{out}.profile");
    let mut buf = create_buf_with_command_and_version_and_header(
        name, 
        header
    );

    let last_line = averages.last_mut().unwrap();
    last_line.reverse();
    for (leaf_distance, chunk) in last_line.chunks_exact(4).enumerate(){
        let k = chunk[1] * factor;
        let i = chunk[2] * factor;
        let d = chunk[3] * factor;
        writeln!(
            buf,
            "{leaf_distance} {i} {k} {d}"
        ).unwrap();
    }
    
    
}

pub fn quenched_chain_crit_scan(opt: DemandVelocityCritOpt, out: &str)
{
    let mut current_chain_len = opt.chain_start.get();
    let cleaner = Cleaner::new();

    let header = [
        "chain_len",
        "a",
        "b",
        "critical_root_demand"
    ];

    let mut crit_buf = create_buf_with_command_and_version_and_header(out, header);

    loop {
        let i_name = current_chain_len.to_string();
        let start = i_name.len();
        let zeros = &ZEROS[start..];
        let name = format!("TMP_{zeros}{i_name}{out}.dat");

        let mut m_opt = opt.opts.clone();
        m_opt.chain_length = NonZeroUsize::new(current_chain_len).unwrap();

        quenched_chain_calc_demand_velocity(m_opt, &name);
        let gp_name = format!("{name}.gp");

        let mut gp_writer = create_gnuplot_buf(&gp_name);
        let png = format!("{name}.png");
        writeln!(gp_writer, "set t pngcairo").unwrap();
        writeln!(gp_writer, "set output '{png}'").unwrap();
        writeln!(gp_writer, "set title 'N={current_chain_len}'").unwrap();
        writeln!(gp_writer, "set ylabel 'v'").unwrap();
        writeln!(gp_writer, "set xlabel 'r'").unwrap();
        writeln!(gp_writer, "set fit quiet").unwrap();
        writeln!(gp_writer, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
        writeln!(gp_writer, "f(x)=a*x+b").unwrap();
        writeln!(gp_writer, "fit f(x) '{name}' u 1:2:(t($2)) yerr via a,b").unwrap();
        
        if let Some(range) = &opt.y_range{
            writeln!(gp_writer, "set yrange [{}:{}]", range.start(), range.end()).unwrap();
        }
        writeln!(gp_writer, "p '{name}' t '', f(x)").unwrap();
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
            
            cleaner.add_multi([name, gp_name, png]);

            writeln!(
                crit_buf,
                "{} {} {} {}",
                current_chain_len,
                a,
                b,
                crit
            ).unwrap();
        }

        current_chain_len += opt.chain_step.get();
        if current_chain_len > opt.chain_end.get(){
            break;
        }
    }
    create_video(
        "TMP_*.png", 
        out, 
        15,
        true
    );

    cleaner.clean();
}


pub fn chain_crit_scan(opt: DemandVelocityCritOpt, out: &str, skip_video: bool)
{
    let mut current_chain_len = opt.chain_start.get();
    let cleaner = Cleaner::new();

    let header = [
        "chain_len",
        "N",
        "a",
        "b",
        "critical_root_demand"
    ];

    let mut crit_buf = create_buf_with_command_and_version_and_header(out, header);

    loop {
        let i_name = current_chain_len.to_string();
        let start = i_name.len();
        let zeros = &ZEROS[start..];
        let name = format!("TMP_{zeros}{i_name}{out}.dat");

        let mut m_opt = opt.opts.clone();
        m_opt.chain_length = NonZeroUsize::new(current_chain_len).unwrap();

        let n = chain_calc_demand_velocity(m_opt, &name);
        let gp_name = format!("{name}.gp");

        let mut gp_writer = create_gnuplot_buf(&gp_name);
        let png = format!("{name}.png");
        writeln!(gp_writer, "set t pngcairo").unwrap();
        writeln!(gp_writer, "set output '{png}'").unwrap();
        writeln!(gp_writer, "set title 'N={current_chain_len}'").unwrap();
        writeln!(gp_writer, "set ylabel 'v'").unwrap();
        writeln!(gp_writer, "set xlabel 'r'").unwrap();
        writeln!(gp_writer, "set fit quiet").unwrap();
        writeln!(gp_writer, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
        writeln!(gp_writer, "f(x)=a*x+b").unwrap();
        writeln!(gp_writer, "fit f(x) '{name}' u 1:2:(t($2)) yerr via a,b").unwrap();
        
        if let Some(range) = &opt.y_range{
            writeln!(gp_writer, "set yrange [{}:{}]", range.start(), range.end()).unwrap();
        }
        writeln!(gp_writer, "p '{name}' t '', f(x)").unwrap();
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
            
            cleaner.add_multi([name, gp_name, png]);

            writeln!(
                crit_buf,
                "{} {n} {} {} {}",
                current_chain_len,
                a,
                b,
                crit
            ).unwrap();
        }

        current_chain_len += opt.chain_step.get();
        if current_chain_len > opt.chain_end.get(){
            break;
        }
    }
    if !skip_video{
        create_video(
            "TMP_*.png", 
            out, 
            15,
            true
        );
    }

    cleaner.clean();
}

pub fn tree_calc_demand_velocity<P>(opt: TreeDemandVelocityOpt, out: P) -> usize
where P: AsRef<Path>
{
    if let Some(t) = opt.threads{
        rayon::ThreadPoolBuilder::new().num_threads(t.get()).build_global().unwrap();
    }
    let mut rng = Pcg64::seed_from_u64(opt.seed);
    let ratio = RatioIter::from_float(
        opt.root_demand_rate_min, 
        opt.root_demand_rate_max, 
        opt.root_demand_samples
    );
    let ratios = ratio.float_iter()
        .map(
            |ratio|
            {
                let rng = Pcg64::from_rng(&mut rng).unwrap();
                Model::create_tree(
                    opt.num_children,
                    opt.tree_depth, 
                    rng, 
                    ratio,
                    opt.max_stock
                )
            }
        )
        .collect_vec();
    let n = ratios[0].nodes.len();

    let velocities: Vec<_> = ratios.into_par_iter()
        .map(
            |mut model|
            {
                let mut sum = 0.0;
                for _ in 0..opt.samples.get(){
                    model.reset_delays();
                    for _ in 0..opt.time.get(){
                        model.update_demand();
                        model.update_production();
                    }
                    sum += model.current_demand[0] / opt.time.get() as f64;
                }
                sum /= opt.samples.get() as f64;
                sum
            }
        ).collect();

    let header = [
        "root_demand",
        "velocity"
    ];

    let mut buf = create_buf_with_command_and_version_and_header(out, header);
    for (root_demand, velocity) in ratio.float_iter().zip(velocities.iter())
    {
        writeln!(
            buf,
            "{root_demand} {velocity}"
        ).unwrap();
    }
    n
}

pub fn tree_crit_scan(opt: TreeDemandVelocityCritOpt, out: Utf8PathBuf)
{
    
    let mut current_tree_depth = opt.tree_depth_start;
    let cleaner = Cleaner::new();

    let header = [
        "tree_depth",
        "a",
        "b",
        "critical_root_demand",
        "N"
    ];

    let mut crit_buf = create_buf_with_command_and_version_and_header(out.as_path(), header);

    loop {
        let i_name = current_tree_depth.to_string();
        let start = i_name.len();
        let zeros = &ZEROS[start..];
        let name = format!("TMP_{zeros}{i_name}{}.dat", out.as_str());

        let mut m_opt = opt.opts.clone();
        m_opt.tree_depth = current_tree_depth;

        let n = tree_calc_demand_velocity(m_opt, &name);
        let gp_name = format!("{name}.gp");

        let mut gp_writer = create_gnuplot_buf(&gp_name);
        let png = format!("{name}.png");
        writeln!(gp_writer, "set t pngcairo").unwrap();
        writeln!(gp_writer, "set output '{png}'").unwrap();
        writeln!(gp_writer, "set title 'DEPTH={current_tree_depth}'").unwrap();
        writeln!(gp_writer, "set ylabel 'v'").unwrap();
        writeln!(gp_writer, "set xlabel 'r'").unwrap();
        writeln!(gp_writer, "set fit quiet").unwrap();
        writeln!(gp_writer, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
        writeln!(gp_writer, "f(x)=a*x+b").unwrap();
        writeln!(gp_writer, "fit f(x) '{name}' u 1:2:(t($2)) yerr via a,b").unwrap();
        
        if let Some(range) = &opt.y_range{
            writeln!(gp_writer, "set yrange [{}:{}]", range.start(), range.end()).unwrap();
        }
        writeln!(gp_writer, "p '{name}' t '', f(x)").unwrap();
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
            
            cleaner.add_multi([name, gp_name, png]);

            writeln!(
                crit_buf,
                "{} {} {} {} {n}",
                current_tree_depth,
                a,
                b,
                crit
            ).unwrap();
        }

        current_tree_depth += 1;
        if current_tree_depth > opt.tree_depth_end{
            break;
        }
    }
    create_video(
        "TMP_*.png", 
        out.as_str(), 
        15,
        true
    );

    cleaner.clean();
}

pub struct RandTreeSample{
    pub filename: String,
    pub num_nodes: usize,
    pub max_depth_reached: usize,
    pub leaf_count: usize
}

pub fn rand_tree_calc_demand_velocity_samples<P>(
    opt: RandTreeDemandVelocityOpt, 
    out: P
) -> Vec<RandTreeSample>
where P: AsRef<Path>
{
    if let Some(t) = opt.threads{
        rayon::ThreadPoolBuilder::new().num_threads(t.get()).build_global().unwrap();
    }
    let mut rng = Pcg64::seed_from_u64(opt.seed);

    let mut trees = (0..opt.samples.get())
        .map(
            |_|
            {
                let m_rng = Pcg64::from_rng(&mut rng).unwrap();
                Model::create_rand_tree(
                    opt.max_depth,
                    m_rng, 
                    opt.root_demand_rate_min, 
                    opt.max_stock,
                    opt.distr.get_distr()
                )
            }
        ).collect_vec();
    

    let ratios = RatioIter::from_float(
        opt.root_demand_rate_min, 
        opt.root_demand_rate_max, 
        opt.root_demand_samples
    );

    let header = [
        "Demand",
        "Velocity"
    ];

    let out_str = out.as_ref().as_os_str().to_str().unwrap();
    trees.par_iter_mut()
        .enumerate()
        .map(
            |(idx, (tree, max_depth_reached))|
            {
                let out_name = format!("Tree{idx}_{out_str}");
                let path: &Path = out_name.as_ref();
                let mut buf = create_buf_with_command_and_version_and_header(
                    path, 
                    header
                );
                for demand in ratios.float_iter(){
                    tree.demand_at_root = demand;
                    let mut sum = 0.0;
                    for _ in 0..opt.samples_per_tree.get(){
                        tree.reset_delays();
                        for _ in 0..opt.time.get() {
                            tree.update_demand();
                            tree.update_production();
                        }
                        sum += tree.current_demand[0];
                    }
                    sum /= (opt.time.get() * opt.samples_per_tree.get()) as f64;
                    let velocity = sum;
                    writeln!(
                        buf,
                        "{demand} {velocity}"
                    ).unwrap();
                }

                let leaf_count = tree.nodes
                    .iter()
                    .filter(|node| node.children.is_empty())
                    .count();

                RandTreeSample{
                    filename: out_name,
                    num_nodes: tree.nodes.len(),
                    max_depth_reached: *max_depth_reached,
                    leaf_count
                }
            }
        ).collect()
}

pub fn rand_tree_crit_scan(opt: RandTreeDemandVelocityCritOpt, out: Utf8PathBuf)
{
    
    let mut current_tree_depth = opt.tree_depth_start;
    let cleaner = Cleaner::new();

    let header = [
        "Max_tree_depth",
        "average_crit",
        "variance_crit"
    ];

    let mut result_buffer = create_buf_with_command_and_version_and_header(
        &out, 
        header
    );


    let header = [
        "max_allowed_tree_depth",
        "crit_sample",
        "num_nodes",
        "max_tree_depth_reached",
        "leaf_count"
    ];


    struct Sample{
        crit: f64,
        num_nodes: usize,
        max_tree_depth_reached: usize,
        leaf_count: usize
    }


    loop {
        let i_name = current_tree_depth.to_string();
        let all_samples_name = format!(
            "all_depth{i_name}_{}",
            out.as_str()
        );
        let mut result_samples = create_buf_with_command_and_version_and_header(
            all_samples_name, 
            header
        );
        let start = i_name.len();
        let zeros = &ZEROS[start..];
        let name = format!("TMP_{zeros}{i_name}{}.dat", out.as_str());
        let mut samples = Vec::new();

        let mut m_opt = opt.opts.clone();
        m_opt.max_depth = current_tree_depth;

        let tree_samples = rand_tree_calc_demand_velocity_samples(m_opt, &name);

        let hist_name = format!("Depth{i_name}{}.hist", out.as_str());
        let mut hist = HistF64::new(
            0.0, 
            1.0, 
            opt.hist_bins.get()
        ).unwrap();

        tree_samples.par_iter()
            .map(
                |sample|
                {   
                    let file = &sample.filename;
                    let gp_name = format!("{file}.gp");
                    let png = format!("{file}.png");
                    let mut gp = create_gnuplot_buf(&gp_name);
        
                    writeln!(gp, "set t pngcairo").unwrap();
                    writeln!(gp, "set output '{png}'").unwrap();
                    writeln!(gp, "set title 'max tree depth = {current_tree_depth}'").unwrap();
                    writeln!(gp, "set ylabel 'v'").unwrap();
                    writeln!(gp, "set xlabel 'r'").unwrap();
                    writeln!(gp, "set fit quiet").unwrap();
                    writeln!(gp, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
                    writeln!(gp, "f(x)=a*x+b").unwrap();
                    writeln!(gp, "fit f(x) '{file}' u 1:2:(t($2)) yerr via a,b").unwrap();
                    
                    if let Some(range) = &opt.y_range{
                        writeln!(gp, "set yrange [{}:{}]", range.start(), range.end()).unwrap();
                    }
                    writeln!(gp, "p '{file}' t '', f(x)").unwrap();
                    writeln!(gp, "print(b)").unwrap();
                    writeln!(gp, "print(a)").unwrap();
                    writeln!(gp, "set output").unwrap();
        
                    drop(gp);
                    let out = call_gnuplot(&gp_name);
                    if out.status.success(){
                        let s = String::from_utf8(out.stderr)
                            .unwrap();
                    
                        let mut iter = s.lines();
        
                        let b: f64 = iter.next().unwrap().parse().unwrap();
                        let a: f64 = iter.next().unwrap().parse().unwrap();
                        let crit = -b/a;
        
                        if !opt.dont_delete_tmps{
                            cleaner.add_multi([gp_name, png]);
                        }
                        Sample{
                            crit,
                            num_nodes: sample.num_nodes,
                            max_tree_depth_reached: sample.max_depth_reached,
                            leaf_count: sample.leaf_count
                        }
                    } else {
                        eprintln!("WARNING: INVALID CRIT ENCOUNTERED!");
                        Sample{
                            crit: f64::NAN,
                            num_nodes: sample.num_nodes,
                            max_tree_depth_reached: sample.max_depth_reached,
                            leaf_count: sample.leaf_count
                        }
                    }
                }
            ).collect_into_vec(&mut samples);

        if opt.ffmpeg{
            let globbing = format!("Tree*_{name}.png");
            let out = format!("Trees_Depth{current_tree_depth}");
            crate::misc::create_video(
                &globbing,
                &out,
                15,
                false
            );
        }


        if !opt.dont_delete_tmps{
            cleaner.add_multi(
                tree_samples.into_iter()
                    .map(|sample| sample.filename)
            );
        }

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        writeln!(
            result_samples,
            "# depth change"
        ).unwrap();
        for sample in samples.iter()
        {
            let crit = sample.crit;
            sum += crit;
            sum_sq += crit * crit;
            writeln!(
                result_samples,
                "{current_tree_depth} {crit} {} {} {}",
                sample.num_nodes,
                sample.max_tree_depth_reached,
                sample.leaf_count
            ).unwrap();
            hist.increment_quiet(crit);
        }
        let average = sum / samples.len() as f64;
        let var = sum_sq / samples.len() as f64 - average * average;

        writeln!(
            result_buffer,
            "{current_tree_depth} {average} {var}"
        ).unwrap();

        let header = [
            "bin_left",
            "bin_right",
            "hits"
        ];

        let mut hist_buf = create_buf_with_command_and_version_and_header(hist_name, header);

        for (bin, hits) in hist.bin_hits_iter(){
            writeln!(
                hist_buf,
                "{} {} {}",
                bin[0],
                bin[1],
                hits
            ).unwrap();
        }


        current_tree_depth += 1;
        if current_tree_depth > opt.tree_depth_end {
            break;
        }
    }
    
    cleaner.clean();
}

/// scans through chain length
pub fn closed_multi_chain_crit_scan(opt: ClosedMultiChainCritOpts, out: Utf8PathBuf)
{
    assert!(
        opt.chain_len_start <= opt.chain_len_end,
        "Chain len start needs to be smaller than chain len end"
    );
    let mut current_chain_len = opt.chain_len_start.get();
    let cleaner = Cleaner::new();

    let header = [
        "chain_len",
        "N",
        "a",
        "b",
        "critical_root_demand"
    ];

    let mut crit_buf = create_buf_with_command_and_version_and_header(out.as_path(), header);

    loop {
        let i_name = current_chain_len.to_string();
        let start = i_name.len();
        let zeros = &ZEROS[start..];
        let name = format!("TMP_{zeros}{i_name}{}.dat", out.as_str());

        let mut m_opt = opt.opts.clone();
        m_opt.other_chain_len = NonZeroUsize::new(current_chain_len).unwrap();

        let n = closed_multi_chain_velocity_scan(m_opt, &name);
        let gp_name = format!("{name}.gp");

        let mut gp_writer = create_gnuplot_buf(&gp_name);
        let png = format!("{name}.png");
        writeln!(gp_writer, "set t pngcairo").unwrap();
        writeln!(gp_writer, "set output '{png}'").unwrap();
        writeln!(gp_writer, "set title 'chain len = {current_chain_len}'").unwrap();
        writeln!(gp_writer, "set ylabel 'v'").unwrap();
        writeln!(gp_writer, "set xlabel 'r'").unwrap();
        writeln!(gp_writer, "set fit quiet").unwrap();
        writeln!(gp_writer, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
        writeln!(gp_writer, "f(x)=a*x+b").unwrap();
        writeln!(gp_writer, "fit f(x) '{name}' u 1:2:(t($2)) yerr via a,b").unwrap();
        
        if let Some(range) = &opt.y_range{
            writeln!(gp_writer, "set yrange [{}:{}]", range.start(), range.end()).unwrap();
        }
        writeln!(gp_writer, "p '{name}' t '', f(x)").unwrap();
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
            
            cleaner.add_multi([name, gp_name, png]);

            writeln!(
                crit_buf,
                "{} {n} {} {} {}",
                current_chain_len,
                a,
                b,
                crit
            ).unwrap();
        }

        current_chain_len += 1;
        if current_chain_len > opt.chain_len_end.get(){
            break;
        }
    }
    create_video(
        "TMP_*.png", 
        out.as_str(), 
        15,
        true
    );

    cleaner.clean();
}

/// Scans through num_chains
pub fn closed_multi_chain_crit_scan2(opt: ClosedMultiChainCritOpts2, out: Utf8PathBuf)
{
    assert!(
        opt.num_chains_end >= opt.num_chains_start,
        "num chain start needs to be smaller than num chain end"
    );
    let mut current_num_chains = opt.num_chains_start.get();
    let cleaner = Cleaner::new();

    let header = [
        "num_chains",
        "N",
        "a",
        "b",
        "critical_root_demand"
    ];

    let mut crit_buf = create_buf_with_command_and_version_and_header(out.as_path(), header);

    loop {
        let i_name = current_num_chains.to_string();
        let start = i_name.len();
        let zeros = &ZEROS[start..];
        let name = format!("TMP_{zeros}{i_name}{}.dat", out.as_str());

        let mut m_opt = opt.opts.clone();
        m_opt.num_chains = NonZeroUsize::new(current_num_chains).unwrap();

        let n = closed_multi_chain_velocity_scan(m_opt, &name);
        let gp_name = format!("{name}.gp");

        let mut gp_writer = create_gnuplot_buf(&gp_name);
        let png = format!("{name}.png");
        writeln!(gp_writer, "set t pngcairo").unwrap();
        writeln!(gp_writer, "set output '{png}'").unwrap();
        writeln!(gp_writer, "set title 'num chains = {current_num_chains}'").unwrap();
        writeln!(gp_writer, "set ylabel 'v'").unwrap();
        writeln!(gp_writer, "set xlabel 'r'").unwrap();
        writeln!(gp_writer, "set fit quiet").unwrap();
        writeln!(gp_writer, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
        writeln!(gp_writer, "f(x)=a*x+b").unwrap();
        writeln!(gp_writer, "fit f(x) '{name}' u 1:2:(t($2)) yerr via a,b").unwrap();
        
        if let Some(range) = &opt.y_range{
            writeln!(gp_writer, "set yrange [{}:{}]", range.start(), range.end()).unwrap();
        }
        writeln!(gp_writer, "p '{name}' t '', f(x)").unwrap();
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
            
            cleaner.add_multi([name, gp_name, png]);

            writeln!(
                crit_buf,
                "{} {n} {} {} {}",
                current_num_chains,
                a,
                b,
                crit
            ).unwrap();
        }

        current_num_chains += 1;
        if current_num_chains > opt.num_chains_end.get(){
            break;
        }
    }
    create_video(
        "TMP_*.png", 
        out.as_str(), 
        15,
        true
    );

    cleaner.clean();
}

/// Returns number of nodes in chain
pub fn chain_calc_demand_velocity<P>(opt: DemandVelocityOpt, out: P) -> usize
where P: AsRef<Path>
{
    if let Some(t) = opt.threads{
        rayon::ThreadPoolBuilder::new().num_threads(t.get()).build_global().unwrap();
    }
    let mut rng = Pcg64::seed_from_u64(opt.seed);
    let ratio = RatioIter::from_float(
        opt.root_demand_rate_min, 
        opt.root_demand_rate_max, 
        opt.root_demand_samples
    );
    let model = Model::new_multi_chain_from_rng(
        opt.num_chains,
        opt.chain_length.get() - 1, 
        Pcg64::from_rng(&mut rng).unwrap(), 
        1.0,
        opt.max_stock
    );

    let ratios: Vec<_> = ratio.float_iter()
        .map(
            |ratio|
            {
                let rng = Pcg64::from_rng(&mut rng).unwrap();
                let mut m = model.clone();
                m.rng = rng;
                m.demand_at_root = ratio;
                m
            }
        )
        .collect();

    // iterate the model a few times to later be 
    // able to initiate the model with a better 
    // stock pattern
    let mut tmp = model;
    for _ in 0..opt.time.get()*10{
        tmp.update_demand();
        tmp.update_production();
    }
    let stocks = tmp.stock_avail;

    let n = ratios[0].nodes.len();

    let velocities: Vec<_> = ratios.into_par_iter()
        .map(
            |mut model|
            {
                let mut sum = 0.0;

                for _ in 0..opt.samples.get(){
                    model.reset_delays();
                    model.stock_avail.clone_from(&stocks);
                    for _ in 0..opt.time.get(){
                        model.update_demand();
                        model.update_production();
                    }
                    sum += model.current_demand[0] / opt.time.get() as f64;
                }
                sum /= opt.samples.get() as f64;
                sum
            }
        ).collect();

    let header = [
        "root_demand",
        "velocity"
    ];

    let mut buf = create_buf_with_command_and_version_and_header(out, header);
    for (root_demand, velocity) in ratio.float_iter().zip(velocities.iter())
    {
        writeln!(
            buf,
            "{root_demand} {velocity}"
        ).unwrap();
    }
    n
}



pub fn quenched_chain_calc_demand_velocity<P>(opt: DemandVelocityOpt, out: P)
where P: AsRef<Path>
{
    if let Some(t) = opt.threads{
        rayon::ThreadPoolBuilder::new().num_threads(t.get()).build_global().unwrap();
    }
    let mut rng = Pcg64::seed_from_u64(opt.seed);
    let ratio = RatioIter::from_float(
        opt.root_demand_rate_min, 
        opt.root_demand_rate_max, 
        opt.root_demand_samples
    );
    let ratios: Vec<_> = ratio.float_iter()
        .map(
            |ratio|
            {
                let model_rng = Pcg64::from_rng(&mut rng).unwrap();

                Model::new_multi_chain_from_rng(
                    opt.num_chains,
                    opt.chain_length.get() - 1, 
                    model_rng, 
                    ratio,
                    opt.max_stock
                )
            }
        )
        .collect();

    let velocities: Vec<_> = ratios.into_par_iter()
        .map(
            |mut model|
            {
                let mut sum = 0.0;
                for _ in 0..opt.samples.get(){
                    model.reset_delays();
                    let quenched_production = Uniform::new_inclusive(0.0, 1.0)
                        .sample_iter(&mut model.rng)
                        .take(model.nodes.len())
                        .collect_vec();
                    for _ in 0..opt.time.get(){
                        model.update_demand();
                        model.update_production_quenched(&quenched_production);
                    }
                    sum += model.current_demand[0] / opt.time.get() as f64;
                }
                sum /= opt.samples.get() as f64;
                sum
            }
        ).collect();

    let header = [
        "root_demand",
        "velocity"
    ];

    let mut buf = create_buf_with_command_and_version_and_header(out, header);
    for (root_demand, velocity) in ratio.float_iter().zip(velocities.iter())
    {
        writeln!(
            buf,
            "{root_demand} {velocity}"
        ).unwrap();
    }
}

/// returns N
pub fn closed_multi_chain_velocity_scan<P>(opt: ClosedMultiChainVelocityOpts, out: P) -> usize
where P: AsRef<Path>
{
    if let Some(t) = opt.threads{
        rayon::ThreadPoolBuilder::new().num_threads(t.get()).build_global().unwrap();
    }
    let mut rng = Pcg64::seed_from_u64(opt.seed);
    let ratio = RatioIter::from_float(
        opt.root_demand_rate_min, 
        opt.root_demand_rate_max, 
        opt.root_demand_samples
    );
    let ratios = ratio.float_iter()
        .map(
            |ratio|
            {
                let rng = Pcg64::from_rng(&mut rng).unwrap();
                Model::new_closed_multi_chain_from_rng(
                    opt.num_chains,
                    opt.other_chain_len, 
                    rng, 
                    ratio,
                    opt.max_stock
                )
            }
        )
        .collect_vec();
    let n = ratios[0].nodes.len();

    let velocities: Vec<_> = ratios.into_par_iter()
        .map(
            |mut model|
            {
                let mut sum = 0.0;
                for _ in 0..opt.samples.get(){
                    model.reset_delays();
                    for _ in 0..opt.time.get(){
                        model.update_demand();
                        model.update_production();
                    }
                    sum += model.current_demand[0] / opt.time.get() as f64;
                }
                sum /= opt.samples.get() as f64;
                sum
            }
        ).collect();

    let header = [
        "root_demand",
        "velocity"
    ];

    let mut buf = create_buf_with_command_and_version_and_header(out, header);
    for (root_demand, velocity) in ratio.float_iter().zip(velocities.iter())
    {
        writeln!(
            buf,
            "{root_demand} {velocity}"
        ).unwrap();
    }
    n
}


