use std::{
    num::NonZeroUsize,
    io::Write
};
use indicatif::*;
use clap::*;
use rayon::prelude::*;

use crate::{create_buf_with_command_and_version, open_as_unwrapped_lines_filter_comments, write_slice_head};


pub fn remove_mean(slice: &mut [f64])
{
    let mut mean: f64 = slice.iter().sum();
    mean /= slice.len() as f64;
    slice.iter_mut()
        .for_each(|v| *v -= mean);
}



pub fn pearson_correlation_coefficient<I>(iterator: I) -> f64
where I: IntoIterator<Item = (f64, f64)>
{
    let mut product_sum = 0.0;
    let mut x_sum = 0.0;
    let mut x_sq_sum = 0.0;
    let mut y_sum = 0.0;
    let mut y_sq_sum = 0.0;
    let mut counter = 0_u64;

    for (x, y) in iterator
    {
        product_sum = x.mul_add(y, product_sum);
        x_sq_sum = x.mul_add(x, x_sq_sum);
        y_sq_sum = y.mul_add(y, y_sq_sum);
        x_sum += x;
        y_sum += y;
        counter += 1;
    }

    let factor = (counter as f64).recip();
    let average_x = x_sum * factor;
    let average_y = y_sum * factor;
    let average_product = product_sum * factor;

    let covariance = average_product - average_x * average_y;
    let variance_x = x_sq_sum * factor - average_x * average_x;
    let variance_y = y_sq_sum * factor - average_y * average_y;
    let std_product = (variance_x * variance_y).sqrt();

    covariance / std_product
}


#[derive(Parser, Debug)]
pub struct CorOpts{
    #[arg(long, short)]
    /// Read in time series
    glob: String,

    #[arg(long, short, default_value_t=0)]
    /// Skip number of samples in the beginning
    skip: usize,

    #[arg(long, short, default_value_t=NonZeroUsize::new(1).unwrap())]
    /// Only calculate the autocorrelation every…
    every: NonZeroUsize,

    #[arg(long, short)]
    /// Path of result
    out: String,

    #[arg(short, long, default_value_t)]
    /// Which column shall we calculate the autocorrelation for?
    col: usize,

    /// How many threads?
    #[arg(short)]
    j: Option<NonZeroUsize>,

    /// Optional max time step
    #[arg(short, long)]
    pub max_time: Option<NonZeroUsize>
}

pub fn calc_correlations(opt: CorOpts)
{
    if let Some(j) = opt.j {
        rayon::ThreadPoolBuilder::new()
            .num_threads(j.get())
            .build_global()
            .unwrap();
    }
    let all: Vec<_> = glob::glob(&opt.glob)
        .unwrap()
        .map(Result::unwrap)
        .collect();

    

    let p_vecs: Vec<_> = all
        .par_iter()
        .progress()
        .map(
            |p|
            {
                let mut data: Vec<_> = open_as_unwrapped_lines_filter_comments(p)
                    .skip(opt.skip)
                    .map(
                        |line|
                        {
                            let mut iter = line.split_ascii_whitespace();
                            let val: f64 = iter.nth(opt.col)
                                .unwrap().parse().unwrap();
                            val
                        }
                    ).collect();
                remove_mean(&mut data);
                
                cross_correlation(&data, &data, opt.every, opt.max_time)
            }
        ).collect();
    
    if p_vecs.len() == 1 {
        let head = ["Delay", "Autocorrelation"];
        println!("Creating {}", opt.out);
        let mut buf = create_buf_with_command_and_version(opt.out);
        
        write_slice_head(&mut buf, head).unwrap();
        let iter = (0_u64..)
            .step_by(opt.every.get())
            .zip(&p_vecs[0]);
        for (delay, pear) in iter
        {
            writeln!(buf, "{delay} {pear}").unwrap();
        }
    } else {        
        let mut median = Vec::new();
        let mut write_all = |name: &str, vecs: &[Vec<f64>]|
        {
            let mut buf = create_buf_with_command_and_version(name);
            let mut head = vec!["Delay".to_string()];
            for p in all.iter(){
                head.push(p.as_os_str().to_str().unwrap().to_owned());
            }
            head.push("Mean".to_owned());
            head.push("Median".to_owned());
            write_slice_head(&mut buf, head).unwrap();

            let factor = (vecs.len() as f64).recip();
            
            for i in 0..vecs[0].len()
            {
                write!(buf, "{} ", i * opt.every.get()).unwrap();
                median.clear();
                let mut sum = 0.0;
                for v in vecs.iter(){
                    sum += v[i];
                    median.push(v[i]);
                    write!(buf, "{} ", v[i]).unwrap();
                }
                let average = sum * factor;
                median.sort_unstable_by(f64::total_cmp);
                let med = median[median.len() / 2];
                writeln!(buf, "{average} {med}").unwrap();
            }
        };
        let p_name = format!("{}_p.dat", opt.out);
        write_all(&p_name, &p_vecs);
        
    }
}

pub fn cross_correlation(
    a: &[f64], 
    b: &[f64], 
    every: NonZeroUsize,
    max_time: Option<NonZeroUsize>
) -> Vec<f64>
{
    assert_eq!(a.len(), b.len());
    let min = match max_time {
        Some(val) => val.get().min(a.len()),
        None => a.len()
    };
    (0..min)
        .step_by(every.get())
        .map(
            |lag|
            {
                let a_slice = &a[lag..];
                let iter = a_slice.iter()//.chain(a[..lag].iter())
                    .zip(
                        b.iter()//.chain(b[..lag].iter())
                    )
                    .map(|(a,b)| (*a, *b));
                pearson_correlation_coefficient(iter)
            }
        ).collect()
}

pub fn cross_correlation_unnormalized_alternative(
    a: &[f64], 
    b: &[f64], 
    every: NonZeroUsize,
    max_time: Option<NonZeroUsize>
) -> Vec<f64>
{
    assert_eq!(a.len(), b.len());
    let min = match max_time {
        Some(val) => val.get().min(a.len()),
        None => a.len()
    };
    (0..min)
        .step_by(every.get())
        .map(
            |lag|
            {
                let a_slice = &a[lag..];
                let a_slice_mean = average(a_slice.iter().copied());
                let b_slice = &b[..a_slice.len()];
                let b_slice_mean = average(b_slice.iter().copied());
                let product_iter = a_slice.iter().zip(b_slice.iter()).map(|(&a, &b)| a*b);
                let product_mean = average(product_iter);
                product_mean - a_slice_mean * b_slice_mean
            }
        ).collect()
}

fn average<I>(iter: I) -> f64
where I: Iterator<Item=f64> + ExactSizeIterator
{
    let mut sum = 0.0;
    let len = iter.len();
    iter.for_each(|v| sum += v);
    sum / len as f64
}