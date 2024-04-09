use std::{io::Write, num::*, path::Path};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use super::{opts::*, model::*};
use crate::misc::*;

pub fn test_profile()
{
    let demand = 0.5;
    let size = 12;
    let mut model = Model::new_chain(
        size, 
        NonZeroUsize::new(1).unwrap(), 
        234073, 
        demand
    );
    let name = "test_profile.dat";
    let mut header = vec!["time_step".to_string()];
    for i in 0..size - 1{
        header.push(format!("d_{i}"));
        header.push(format!("I_{i}"));
        header.push(format!("k_{i}"));
        header.push(format!("a_{i}"));
    }
    header.push(format!("d_{}", size-1));
    header.push(format!("I_{}", size-1));
    let mut buf = create_buf_with_command_and_version_and_header(name, header);
    for t in 1..200{

        model.update_demand();
        model.update_production();

        write!(buf, "{t} ").unwrap();
        for i in 0..size - 1{
            write!(
                buf,
                "{} {} {} {} ",
                model.current_demand[i],
                model.currently_produced[i],
                model.stock_avail[i][0].stock,
                model.stock_avail[i][0].currently_avail,
            ).unwrap();
        }
        writeln!(
            buf,
            "{} {}",
            model.current_demand[size-1],
            model.currently_produced[size-1]
        ).unwrap();
        
    }
}


pub fn chain_crit_scan(opt: DemandVelocityCritOpt, out: &str)
{
    let zeros = "000000000";
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
        let zeros = &zeros[start..];
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
                    ratio
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