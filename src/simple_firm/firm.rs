use {
    rand_pcg::Pcg64,
    rand::{
        SeedableRng,
        distributions::Distribution
    },
    std::io::Write,
    super::*,
    rayon::prelude::*,
    crate::any_dist::*
};


#[derive(Clone, Copy)]
struct Firm{
    current_delay: f64,
    buffer: f64
}

pub struct FocusFirmInner {
    focus: Firm,
    others: Vec<Firm>,
    iteration: usize
}


pub struct FocusFirmOuter{
    pub focus_firm: FocusFirmInner,
    rng: Pcg64,
    delay_dist: AnyDist
}

impl FocusFirmOuter{
    pub fn new_const_buf(k: usize, focus_buffer: f64, other_buffer: f64, delay_dist: AnyDist, rng: Pcg64) -> Self
    {
        let focus_firm = FocusFirmInner::new_const_buf(k, focus_buffer, other_buffer);
        Self{
            rng,
            focus_firm,
            delay_dist
        }
    }

    pub fn new_const_min_buf(k: usize, focus_buffer: f64, other_buffer: BufferConstMin, delay_dist: AnyDist, rng: Pcg64) -> Self
    {
        let focus_firm = FocusFirmInner::new_const_min_buf(k, focus_buffer, other_buffer);
        Self{
            rng,
            focus_firm,
            delay_dist
        }
    }

    pub fn new_const_min_frac_buf(k: usize, focus_buffer: f64, other_buffer: BufferConstMinFrac, delay_dist: AnyDist, rng: Pcg64) -> Self
    {
        let focus_firm = FocusFirmInner::new_const_min_frac_buf(k, focus_buffer, other_buffer);
        Self{
            rng,
            focus_firm,
            delay_dist
        }
    }

    pub fn new<D>(k: usize, focus_buffer: f64, delay_dist: AnyDist, mut rng: Pcg64, buffer_dist: D) -> Self
    where D: Distribution<f64>
    {
        let focus_firm = FocusFirmInner::new(k, focus_buffer, buffer_dist, &mut rng);
        Self{
            focus_firm,
            rng,
            delay_dist
        }
    }

    pub fn new_dist_max<D>(k: usize, focus_buffer: f64, delay_dist: AnyDist, mut rng: Pcg64, buffer_dist: D, max_buf: f64) -> Self
    where D: Distribution<f64>
    {
        let focus_firm = FocusFirmInner::new_dist_max(k, focus_buffer, buffer_dist, max_buf, &mut rng);
        Self{
            focus_firm,
            rng,
            delay_dist
        }
    }

    // The only reason I programmed this so complicated was: I had fun doing so
    pub fn get_firm_creator(buf_dist: AnyBufDist, delay_dist: AnyDist) -> Box<dyn Fn (usize, f64, Pcg64) -> FocusFirmOuter + std::marker::Sync>
    {
        match buf_dist{
            AnyBufDist::Constant(val) => {
                let fun = move |k, focus_buffer, rng| {
                    FocusFirmOuter::new_const_buf(k, focus_buffer, val, delay_dist.clone(), rng)
                };

                Box::new(fun)
            },
            AnyBufDist::Any(any) => {
                let dist: AnyDist = any.into();

                match dist{
                    AnyDist::Exponential(exp) => {
                        let fun = move |k, focus_buffer, rng| {
                            let buffer_dist = exp.create_dist();
                            FocusFirmOuter::new(k, focus_buffer, delay_dist.clone(), rng, buffer_dist)
                        };
                        Box::new(fun)
                    },
                    AnyDist::Uniform(uni) => {
                        let uniform = uni.create_dist();
                        let fun = move |k, focus_buffer, rng| {
                            FocusFirmOuter::new(k, focus_buffer, delay_dist.clone(), rng, uniform)
                        };
                        Box::new(fun)
                    }
                }
            },
            AnyBufDist::ConstMin(cm) => {
                let fun = move |k, focus_buffer, rng| {
                    FocusFirmOuter::new_const_min_buf(k, focus_buffer, cm.clone(), delay_dist.clone(), rng)
                };
                Box::new(fun)
            },
            AnyBufDist::ConstMinFrac(cmf) => {
                let fun = move |k, focus_buffer, rng| {
                    FocusFirmOuter::new_const_min_frac_buf(k, focus_buffer, cmf.clone(), delay_dist.clone(), rng)
                };
                Box::new(fun)
            },
            AnyBufDist::UniformMax(u_max) => {
                let uniform: UniformDistCreator = u_max.uniform.into();
                let dist = uniform.create_dist();
                let fun = move |k, focus_buffer, rng| {
                    FocusFirmOuter::new_dist_max(k, focus_buffer, delay_dist.clone(), rng, dist, u_max.buf_max)
                };
                Box::new(fun)
            }
        }
    }

    pub fn iterate(&mut self)
    {
        match self.delay_dist {
            AnyDist::Exponential(exp) => {
                let dist = exp.create_dist();
                let iter = dist.sample_iter(&mut self.rng);
                self.focus_firm.iterate(iter);
            },
            AnyDist::Uniform(uni) => {
                let dist = uni.create_dist();
                let iter = dist.sample_iter(&mut self.rng);
                self.focus_firm.iterate(iter);
            }
        }
    }
}

impl FocusFirmInner {
    pub fn new_const_buf(k: usize, focus_buffer: f64, other_buffer: f64) -> Self
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer: focus_buffer
        };
        let others = (0..k)
            .map(
                |_| 
                {
                    Firm{
                        current_delay: 0.0,
                        buffer: other_buffer
                    }
                }
            ).collect();
        Self{
            focus,
            others,
            iteration: 0
        }
    }

    pub fn new_const_min_buf(k: usize, focus_buffer: f64, other_buffer: BufferConstMin) -> Self
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer: focus_buffer
        };
        let mut others = if k == 0 {
            Vec::new()
        } else {
            vec![Firm{current_delay: 0.0, buffer: other_buffer.buf_min}]
        };
        if k > 1 {
            others.extend(
                (0..k-1)
                    .map(
                        |_| 
                        {
                            Firm{
                                current_delay: 0.0,
                                buffer: other_buffer.buf_const
                            }
                        }
                    )
                );
        }

        Self{
            focus,
            others,
            iteration: 0
        }
    }

    pub fn new_const_min_frac_buf(k: usize, focus_buffer: f64, other_buffer: BufferConstMinFrac) -> Self
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer: focus_buffer
        };
        let at_min = ((k as f64) * other_buffer.min_frac).floor() as usize;
        let mut others = if k == 0 {
            Vec::new()
        } else {
            vec![Firm{current_delay: 0.0, buffer: other_buffer.buf_min}; at_min]
        };
        if k > 1 {
            others.extend(
                (0..k-at_min)
                    .map(
                        |_| 
                        {
                            Firm{
                                current_delay: 0.0,
                                buffer: other_buffer.buf_const
                            }
                        }
                    )
                );
        }

        Self{
            focus,
            others,
            iteration: 0
        }
    }

    pub fn new_dist_max<D>(k: usize, focus_buffer: f64, buffer_dist: D, max_buf: f64, rng: &mut Pcg64) -> Self
    where D: Distribution<f64>
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer: focus_buffer
        };
        let others = buffer_dist.sample_iter(rng)
            .take(k)
            .map(
                |buffer|
                    {
                        let buf = buffer.min(max_buf);
                        Firm{
                            current_delay: 0.0,
                            buffer: buf
                        }
                    }
            ).collect();
        Self{
            focus,
            others,
            iteration: 0
        }
    }

    pub fn new<D>(k: usize, focus_buffer: f64, buffer_dist: D, rng: &mut Pcg64) -> Self
    where D: Distribution<f64>
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer: focus_buffer
        };
        let others = buffer_dist.sample_iter(rng)
            .take(k)
            .map(
                |buffer|
                    {
                        Firm{
                            current_delay: 0.0,
                            buffer
                        }
                    }
            ).collect();
        Self{
            focus,
            others,
            iteration: 0
        }
    }

    #[inline]
    pub fn iterate<I>(&mut self, mut delta_iter: I)
    where I: Iterator<Item=f64>
    {
        let mut maximum = 0.0;
        for firm in self.others.iter_mut()
        {
            maximum = firm.current_delay.max(maximum);
            firm.current_delay =
                (firm.current_delay  - firm.buffer)
                    .max(0.0)
                + unsafe{delta_iter.next().unwrap_unchecked()};
        }

        self.focus.current_delay =
            (self.focus.current_delay - self.focus.buffer + maximum)
                .max(0.0)
            + unsafe{delta_iter.next().unwrap_unchecked()};
        self.iteration += 1;
    }
}

pub fn different_k(option: &SimpleFirmDifferentKOpts)
{
    let dist: AnyDist = option.delay_dist.clone().try_into()
        .expect("Const Distribution not allowed for this");
    let mut buf = option.get_buf();
    write!(buf, "#k=").unwrap();
    for k in option.k.iter()
    {
        write!(buf, "{k} ").unwrap();
    }
    writeln!(buf).unwrap();
    let mut rng = Pcg64::seed_from_u64(option.seed);

    let buf_dist = option.buf_dist.clone();
    let firm_creator = FocusFirmOuter::get_firm_creator(buf_dist, dist);

    let mut firms: Vec<_> = option.k.iter()
        .map(
            |&k| 
            {
                let f_rng = Pcg64::from_rng(&mut rng).unwrap();
                firm_creator(k, option.focus_buffer, f_rng)
            }
        )
        .collect();

    for i in 0..option.iter_limit.get()
    {
        write!(buf, "{i}").unwrap();
        for firm in firms.iter_mut(){
            write!(buf, " {}", firm.focus_firm.focus.current_delay).unwrap();
            firm.iterate();
        }
        writeln!(buf).unwrap();
    }

}


pub fn measure_phase(opt: &SimpleFirmPhase)
{
    let len = opt.focus_buffer.len();
    let time = opt.iter_limit.get();
    let t = time as f64;

    let delay_dist = opt.delay_dist.clone();
    let buf_dist = opt.buffer_dist.clone();

    
    let new_firms = FocusFirmOuter::get_firm_creator(buf_dist, delay_dist);


    (0..len)
        .into_par_iter()
        .for_each(
            |i|
            {
                let mut buf = opt.get_buf(i);
                writeln!(buf, "#k lastDelay av av_dt variance variance_dt").unwrap();
                let mut rng = Pcg64::seed_from_u64(opt.seed);
                let buffer = opt.focus_buffer[i];

                for k in (opt.k_start.get()..opt.k_end.get()).step_by(opt.k_step_by.get()){
                    
                    let f_rng = Pcg64::from_rng(&mut rng).unwrap();
                    
                    let mut firms = new_firms(k, buffer, f_rng);
                    let mut sum = 0.0;
                    let mut sum_sq = 0.0;

                    for _ in 0..time{
                        firms.iterate();
                        let current_delay = firms.focus_firm.focus.current_delay;
                        sum += current_delay;
                        sum_sq += current_delay * current_delay;
                    }

                    let av = sum / t;
                    let av_dt = av / t;
                    let var = sum_sq / t - av * av;
                    let var_dt = var / t;
                    writeln!(buf, "{k} {} {av} {av_dt} {var} {var_dt}", firms.focus_firm.focus.current_delay).unwrap();
                } 
            }
        );    
}