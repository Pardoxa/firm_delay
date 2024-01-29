use std::{
    borrow::Borrow, 
    num::NonZeroUsize,
    io::Write
};
use indicatif::*;
use clap::*;
use rayon::prelude::*;

use crate::{create_buf_with_command_and_version, open_as_unwrapped_lines_filter_comments};


pub fn remove_mean(slice: &mut [f64])
{
    let mut mean: f64 = slice.iter().sum();
    mean /= slice.len() as f64;
    slice.iter_mut()
        .for_each(|v| *v -= mean);
}

pub fn cross_correlation(a: &[f64], b: &[f64]) -> Vec<f64>
{
    assert_eq!(a.len(), b.len());
    (0..a.len())
        .progress()
        .map(
            |lag|
            {
                a[lag..].iter()//.chain(a[..lag].iter())
                    .zip(
                        b.iter()//.chain(b[..lag].iter())
                    ).map(|(&this, &other)| this * other)
                    .sum()
            }
        ).collect()
}

pub fn cross_correlation_alt(a: &[f64], b: &[f64]) -> Vec<f64>
{
    assert_eq!(a.len(), b.len());
    (0..a.len())
        .progress()
        .map(
            |lag|
            {
                let iter = a[lag..].iter()//.chain(a[..lag].iter())
                    .zip(
                        b.iter()//.chain(b[..lag].iter())
                    ).map(|(a,b)| (*a, *b));
                pearson_correlation_coefficient(iter)
            }
        ).collect()
}

pub fn pearson_correlation_coefficient<I, F>(iterator: I) -> f64
where I: IntoIterator<Item = (F, F)>,
    F: Borrow<f64>
{
    let mut product_sum = 0.0;
    let mut x_sum = 0.0;
    let mut x_sq_sum = 0.0;
    let mut y_sum = 0.0;
    let mut y_sq_sum = 0.0;
    let mut counter = 0_u64;

    for (x, y) in iterator
    {
        let x = *x.borrow();
        let y = *y.borrow();
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

    let mut res_vecs = Vec::new();
    let mut p_vecs = Vec::new();

    all
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
                            let val: f64 = iter.nth(1).unwrap().parse().unwrap();
                            val
                        }
                    ).collect();
                remove_mean(&mut data);
                let res = cross_correlation_test(&data, &data, opt.every, opt.max_time);
                let p = cross_correlation_test2(&data, &data, opt.every, opt.max_time);
                (res, p)
            }
        ).unzip_into_vecs(&mut res_vecs, &mut p_vecs);
    
    if res_vecs.len() == 1 {
        println!("Creating {}", opt.out);
        let mut buf = create_buf_with_command_and_version(opt.out);
        let iter = (0_u64..)
            .step_by(opt.every.get())
            .zip(&res_vecs[0])
            .zip(&p_vecs[0]);
        for ((delay, auto), pear) in iter
        {
            writeln!(buf, "{delay} {auto} {pear}").unwrap();
        }
    } else {        
        let mut median = Vec::new();
        let mut write_all = |name: &str, vecs: &[Vec<f64>]|
        {
            let mut buf = create_buf_with_command_and_version(name);
    
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
        let r_name = format!("{}_res.dat", opt.out);
        write_all(&r_name, &res_vecs);
        
    }
}

pub fn cross_correlation_test(
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
                let p_sum: f64 = a_slice.iter()//.chain(a[..lag].iter())
                    .zip(
                        b.iter()//.chain(b[..lag].iter())
                    ).map(|(&this, &other)| this * other)
                    .sum();
                p_sum / a_slice.len() as f64
            }
        ).collect()
}

pub fn cross_correlation_test2(
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