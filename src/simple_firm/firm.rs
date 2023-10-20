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
    pub fn new_const_buf(k: usize, buffer: f64, delay_dist: AnyDist, rng: Pcg64) -> Self
    {
        let focus_firm = FocusFirmInner::new_const_buf(k, buffer);
        Self{
            rng,
            focus_firm,
            delay_dist
        }
    }

    pub fn new(k: usize, focus_buffer: f64, delay_dist: AnyDist, mut rng: Pcg64, buffer_dist: AnyDist) -> Self
    {
        let focus_firm = match buffer_dist{
            AnyDist::Exponential(exp) => {
                let dist = exp.create_dist();
                FocusFirmInner::new(k, focus_buffer, dist, &mut rng)
            },
            AnyDist::Uniform(uni) => {
                let dist = uni.create_dist();
                FocusFirmInner::new(k, focus_buffer, dist, &mut rng)
            }
        };
        Self{
            focus_firm,
            rng,
            delay_dist
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
    pub fn new_const_buf(k: usize, buffer: f64) -> Self
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer
        };
        let others = (0..k-1)
            .map(
                |_| 
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

    pub fn new<D>(k: usize, focus_buffer: f64, buffer_dist: D, rng: &mut Pcg64) -> Self
    where D: Distribution<f64>
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer: focus_buffer
        };
        let others = buffer_dist.sample_iter(rng)
            .take(k-1)
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
    let dist: AnyDist = option.delay_dist.clone().into();
    let mut buf = option.get_buf();
    write!(buf, "#k=").unwrap();
    for k in option.k.iter()
    {
        write!(buf, "{k} ").unwrap();
    }
    writeln!(buf).unwrap();
    let mut rng = Pcg64::seed_from_u64(option.seed);

    let mut firms: Vec<_> = option.k.iter()
        .map(
            |k| 
            {
                let f_rng = Pcg64::from_rng(&mut rng).unwrap();
                FocusFirmOuter::new_const_buf(k.get(), option.buffer, dist.clone(), f_rng)
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
    
    let buffer_dist: AnyDist = opt.buffer_dist.clone().into();

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
                    let mut firms = FocusFirmOuter::new_const_buf(k, buffer, buffer_dist.clone(), f_rng);
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