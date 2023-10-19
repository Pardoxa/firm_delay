use {
    rand_pcg::Pcg64,
    rand::{SeedableRng, 
        distributions::{Uniform, Distribution}
    },
    std::io::Write,
    super::*,
    rayon::prelude::*
};


struct Firm{
    current_delay: f64,
    buffer: f64
}

struct FocusFirm{
    focus: Firm,
    others: Vec<Firm>,
    iteration: usize,
    delta: f64,
    rng: Pcg64
}

impl FocusFirm{
    pub fn new_const_buf(k: usize, delta: f64, buffer: f64, rng: Pcg64) -> Self
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
            iteration: 0,
            delta,
            rng
        }
    }

    pub fn new<D>(k: usize, delta: f64, focus_buffer: f64, buffer_dist: D, mut rng: Pcg64) -> Self
    where D: Distribution<f64>
    {
        let focus = Firm{
            current_delay: 0.0,
            buffer: focus_buffer
        };
        let others = buffer_dist.sample_iter(&mut rng)
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
            iteration: 0,
            delta,
            rng
        }
    }

    #[inline]
    pub fn iterate(&mut self)
    {
        let uniform = Uniform::new_inclusive(0.0, self.delta);
        let mut iter = uniform.sample_iter(&mut self.rng);

        let mut maximum = 0.0;
        for firm in self.others.iter_mut()
        {
            maximum = firm.current_delay.max(maximum);
            firm.current_delay =
                (firm.current_delay  - firm.buffer)
                    .max(0.0)
                + unsafe{iter.next().unwrap_unchecked()};
        }

        self.focus.current_delay =
            (self.focus.current_delay - self.focus.buffer + maximum)
                .max(0.0)
            + unsafe{iter.next().unwrap_unchecked()};
        self.iteration += 1;
    }
}

pub fn different_k(option: &SimpleFirmDifferentKOpts)
{

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
                FocusFirm::new_const_buf(k.get(), option.delta, option.buffer, f_rng)
            }
        )
        .collect();

    for i in 0..option.iter_limit.get()
    {
        write!(buf, "{i}").unwrap();
        for firm in firms.iter_mut(){
            write!(buf, " {}", firm.focus.current_delay).unwrap();
            firm.iterate();
        }
        writeln!(buf).unwrap();
    }

}

pub fn measure_phase(opt: &SimpleFirmPhase)
{
    let len = opt.buffer.len();
    let time = opt.iter_limit.get();
    let t = time as f64;

    (0..len)
        .into_par_iter()
        .for_each(
            |i|
            {
                let mut buf = opt.get_buf(i);
                writeln!(buf, "#k lastDelay av av_dt variance variance_dt").unwrap();
                let mut rng = Pcg64::seed_from_u64(opt.seed);
                let buffer = opt.buffer[i];

                for k in (opt.k_start.get()..opt.k_end.get()).step_by(opt.k_step_by.get()){
                    
                    let f_rng = Pcg64::from_rng(&mut rng).unwrap();
                    let mut firms = FocusFirm::new_const_buf(k, opt.delta, buffer, f_rng);
                    let mut sum = 0.0;
                    let mut sum_sq = 0.0;

                    for _ in 0..time{
                        firms.iterate();
                        sum += firms.focus.current_delay;
                        sum_sq += firms.focus.current_delay * firms.focus.current_delay;
                    }

                    let av = sum / t;
                    let av_dt = av / t;
                    let var = sum_sq / t - av * av;
                    let var_dt = var / t;
                    writeln!(buf, "{k} {} {av} {av_dt} {var} {var_dt}", firms.focus.current_delay).unwrap();
                } 
            }
        );    
}