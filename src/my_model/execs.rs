use std::{io::Write, num::*, path::Path};
use camino::Utf8PathBuf;
use itertools::Itertools;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
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


    let mut buf = create_buf_with_command_and_version_and_header(out, header);
    let factor = 1.0 / opt.average_over_samples.get() as f64;
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


pub fn chain_crit_scan(opt: DemandVelocityCritOpt, out: &str)
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

        chain_calc_demand_velocity(m_opt, &name);
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

pub fn tree_crit_scan(opt: TreeDemandVelocityCritOpt, out: Utf8PathBuf)
{
    
    let mut current_tree_depth = opt.tree_depth_start;
    let cleaner = Cleaner::new();

    let header = [
        "tree_depth",
        "a",
        "b",
        "critical_root_demand"
    ];

    let mut crit_buf = create_buf_with_command_and_version_and_header(out.as_path(), header);

    loop {
        let i_name = current_tree_depth.to_string();
        let start = i_name.len();
        let zeros = &ZEROS[start..];
        let name = format!("TMP_{zeros}{i_name}{}.dat", out.as_str());

        let mut m_opt = opt.opts.clone();
        m_opt.tree_depth = current_tree_depth;

        tree_calc_demand_velocity(m_opt, &name);
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
                "{} {} {} {}",
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



pub fn chain_calc_demand_velocity<P>(opt: DemandVelocityOpt, out: P)
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
                let rng = Pcg64::from_rng(&mut rng).unwrap();
                Model::new_multi_chain_from_rng(
                    opt.num_chains,
                    opt.chain_length.get() - 1, 
                    rng, 
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
}


pub fn tree_calc_demand_velocity<P>(opt: TreeDemandVelocityOpt, out: P)
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
}