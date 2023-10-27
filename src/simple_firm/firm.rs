use {
    rand_pcg::Pcg64,
    rand::{
        SeedableRng,
        distributions::Distribution,
        Rng
    },
    std::{
        io::Write,
        sync::Mutex,
        ops::DerefMut,
        num::NonZeroU64,
        sync::atomic::{AtomicU64, Ordering::Relaxed}
    },
    super::*,
    rayon::prelude::*,
    crate::any_dist::*,
    sampling::AtomicHistF64,
    indicatif::*
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

    pub fn average_other_delay(&self) -> f64
    {
        let all: f64 = self.focus_firm
            .others
            .iter()
            .map(|f| f.current_delay)
            .sum();
        all / self.focus_firm.others.len() as f64
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

    /// Get Firm builder function according to the correct distributions
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
                self.focus_firm.iterate(dist, &mut self.rng);
            },
            AnyDist::Uniform(uni) => {
                let dist = uni.create_dist();
                self.focus_firm.iterate(dist, &mut self.rng);
            }
        }
    }

    pub fn iterate_moran(&mut self, tmp: &mut [f64], n: usize, ids: &mut Vec<usize>, scratch: &mut Vec<usize>)
    {
        match self.delay_dist {
            AnyDist::Exponential(exp) => {
                let dist = exp.create_dist();
                self.focus_firm.iterate_moron(dist, &mut self.rng, tmp, n, ids, scratch);
            },
            AnyDist::Uniform(uni) => {
                let dist = uni.create_dist();
                self.focus_firm.iterate_moron(dist, &mut self.rng, tmp, n, ids, scratch);
            }
        }
    }

    pub fn iterate_multiple_steps(&mut self, steps: NonZeroU64)
    {
        match self.delay_dist {
            AnyDist::Exponential(exp) => {
                let dist = exp.create_dist();
                for _ in 0..steps.get(){
                    self.focus_firm.iterate(dist, &mut self.rng);
                }
            },
            AnyDist::Uniform(uni) => {
                let dist = uni.create_dist();
                for _ in 0..steps.get(){
                    self.focus_firm.iterate(dist, &mut self.rng);
                }
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
    pub fn iterate<I>(&mut self, dist: I, rng: &mut Pcg64)
    where I: Distribution<f64>
    {
        let mut delta_iter = dist.sample_iter(rng);
        let mut maximum = self.focus.current_delay;
        for firm in self.others.iter_mut()
        {
            maximum = firm.current_delay.max(maximum);
            firm.current_delay =
                (firm.current_delay  - firm.buffer)
                    .max(0.0)
                + unsafe{delta_iter.next().unwrap_unchecked()};
        }

        self.focus.current_delay =
            (maximum - self.focus.buffer)
                .max(0.0)
            + unsafe{delta_iter.next().unwrap_unchecked()};
        self.iteration += 1;
    }

    #[inline]
    pub fn iterate_moron<I>(&mut self, dist: I, rng: &mut Pcg64, tmp: &mut [f64], n: usize, ids: &mut Vec<usize>, scratch: &mut Vec<usize>)
    where I: Distribution<f64>
    {

        for temp in tmp.iter_mut()
        {
            for _ in 0..n{
                let idx = rng.gen_range(0..ids.len());
                let val = ids.swap_remove(idx);
                scratch.push(val);
            }
            *temp = 0.0;
            for &i in scratch.iter(){
                let delay = self.others[i].current_delay;
                if delay > *temp {
                    *temp = delay;
                }
            }
            ids.append(scratch);
        }
        let mut delta_iter = dist.sample_iter(rng);
        for (firm, max) in self.others.iter_mut().zip(tmp.iter())
        {
            
            firm.current_delay =
                (max  - firm.buffer)
                    .max(0.0)
                + unsafe{delta_iter.next().unwrap_unchecked()};
        }

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

pub fn different_k_with_max(option: &SimpleFirmDifferentKOpts)
{
    let dist: AnyDist = option.delay_dist.clone().into();
    let mut buf = option.get_buf();
    let mut buf2 = option.get_buf2();
    write!(buf, "#k=").unwrap();
    write!(buf2, "#k=").unwrap();
    for k in option.k.iter()
    {
        write!(buf, "{k} ").unwrap();
        write!(buf2, "{k} ").unwrap();
    }
    writeln!(buf).unwrap();
    writeln!(buf2).unwrap();
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
    
    let every = option.write_every.unwrap_or(1);
    for i in 0..option.iter_limit.get()
    {
        let w = (i % every) == 0;
        if w{
            write!(buf, "{i}").unwrap();
            write!(buf2, "{i}").unwrap();
        }
        for firm in firms.iter_mut(){
            if w {
                write!(buf, " {}", firm.focus_firm.focus.current_delay).unwrap();
            }
            let mut max = f64::NEG_INFINITY;
            let mut buffer = 0.0;
            for f in firm.focus_firm.others.iter()
            {
                if f.current_delay > max {
                    max = f.current_delay;
                    buffer = f.buffer;
                }
            }
            if w{
                write!(buf2, " {max} {buffer}").unwrap();
            }
            firm.iterate();
        }
        if w{
            writeln!(buf).unwrap();
            writeln!(buf2).unwrap();
        }
    }

}

pub fn average_delay_measurement(option: &SimpleFirmAverageAfter)
{
    let dist: AnyDist = option.delay_dist.clone().into();
    let mut buf = option.get_buf(false);
    
    let rng = Pcg64::seed_from_u64(option.seed);
    let rng = Mutex::new(rng);

    let delta = (option.other_buffer_max-option.other_buffer_min)/(option.other_buffer_steps-1) as f64;

    let samples_per_thread = option.average_samples / option.threads;
    let actual_samples = samples_per_thread * option.threads;

    let bar = crate::misc::indication_bar(option.other_buffer_steps as u64);

    for i in (0..option.other_buffer_steps).progress_with(bar){
        let buffer = option.other_buffer_min + delta * i as f64;
        let global_sum = Mutex::new(0.0);
        (0..option.threads)
            .into_par_iter()
            .for_each(
                |_|
                {
                    let mut guard = rng.lock().unwrap();
                    let mut thread_rng = Pcg64::from_rng(guard.deref_mut()).unwrap();
                    drop(guard);
                    let mut sum = 0.0;
                    for _ in 0..samples_per_thread{
                        let f_rng = Pcg64::from_rng(&mut thread_rng).unwrap();
                        let mut firms = FocusFirmOuter::new_const_buf(
                            option.k, 
                            option.focus_buffer, 
                            buffer, 
                            dist.clone(), 
                            f_rng
                        );

                        firms.iterate_multiple_steps(option.time);

                        sum += firms.focus_firm.focus.current_delay;
                    }
                    let mut guard = global_sum.lock().unwrap();
                    *(guard.deref_mut()) += sum;
                    drop(guard);
                }
            );
        let sum = global_sum.into_inner().unwrap();
        let average = sum / actual_samples as f64;
        writeln!(buf, "{} {}", buffer, average).unwrap();
    }
    

}

pub fn average_delay_order_measurement(option: &SimpleFirmAverageAfter)
{
    rayon::ThreadPoolBuilder::new().num_threads(option.threads).build_global().unwrap();
    let dist: AnyDist = option.delay_dist.clone().into();
    let mut buf = option.get_buf(false);
    writeln!(buf, "#B average_mid average_end order av_time_above slope").unwrap();
    let rng = Pcg64::seed_from_u64(option.seed);
    let rng = Mutex::new(rng);

    let delta = (option.other_buffer_max-option.other_buffer_min)/(option.other_buffer_steps-1) as f64;

    let samples_per_thread = option.average_samples / option.threads;
    let actual_samples = samples_per_thread * option.threads;
    dbg!(actual_samples);
    dbg!(samples_per_thread);

    let bar = crate::misc::indication_bar(option.other_buffer_steps as u64);

    let time_half = NonZeroU64::new(option.time.get() / 2).unwrap();

    let scan_buf_dist = option.get_scan_buf_dist();

    for i in (0..option.other_buffer_steps).progress_with(bar){
        let buffer = option.other_buffer_min + delta * i as f64;
        let global_sum_mid = Mutex::new(0.0);
        let global_sum_end = Mutex::new(0.0);
        let glob_time_above = AtomicU64::new(0);

        let bd = match scan_buf_dist{
            ScanBufDist::Const => {
                AnyBufDist::Constant(buffer)
            },
            ScanBufDist::Uniform(hw) => {
                let any_uniform = UniformDistCreator2{mean: buffer, half_width: hw.half_width};
                if !any_uniform.is_valid(){
                    continue;
                }
                let uniform = any_uniform.into();
                AnyBufDist::Any(AnyDistCreator::Uniform(uniform))
            },
            ScanBufDist::MinScan(ms) => {
                let c = ms.other_consts;
                AnyBufDist::ConstMin(BufferConstMin { buf_const: c, buf_min: buffer })
            }
        };
        let firm_creator = FocusFirmOuter::get_firm_creator(bd, dist.clone());

        (0..option.threads)
            .into_par_iter()
            .for_each(
                |_|
                {
                    let mut guard = rng.lock().unwrap();
                    let mut thread_rng = Pcg64::from_rng(guard.deref_mut()).unwrap();
                    drop(guard);
                    let mut sum_mid = 0.0;
                    let mut sum_end = 0.0;
                    
                    let mut time_above = 0;
                    for _ in 0..samples_per_thread{
                        let f_rng = Pcg64::from_rng(&mut thread_rng).unwrap();
                        let mut firms = firm_creator(option.k, option.focus_buffer, f_rng);
                        
                        for _ in 0..time_half.get(){
                            firms.iterate();
                            if firms.focus_firm.focus.current_delay > firms.focus_firm.focus.buffer {
                                time_above += 1;
                            }
                        }
                        sum_mid += firms.focus_firm.focus.current_delay;
                        for _ in 0..time_half.get(){
                            firms.iterate();
                            if firms.focus_firm.focus.current_delay > firms.focus_firm.focus.buffer {
                                time_above += 1;
                            }
                        }
                        sum_end += firms.focus_firm.focus.current_delay;
                    }

                    glob_time_above.fetch_add(time_above, std::sync::atomic::Ordering::Relaxed);
                    
                    let mut guard = global_sum_mid.lock().unwrap();
                    *(guard.deref_mut()) += sum_mid;
                    drop(guard);

                    let mut guard: std::sync::MutexGuard<'_, f64> = global_sum_end.lock().unwrap();
                    *(guard.deref_mut()) += sum_end;
                    drop(guard);
                }
            );
        let sum_mid: f64 = global_sum_mid.into_inner().unwrap();
        
        let average_mid = sum_mid / actual_samples as f64;

        let sum_end = global_sum_end.into_inner().unwrap();
        let average_end = sum_end / actual_samples as f64;


        let slope = (average_end-average_mid)/ (time_half.get() as f64);

        let order = sum_end/sum_mid;

        let av_time_above = (glob_time_above.into_inner() as f64 / actual_samples as f64) 
            / (time_half.get() * 2) as f64;

        writeln!(buf, "{} {} {} {} {av_time_above} {slope}", buffer, average_mid, average_end, order).unwrap();
    }
    

}

pub fn recreate_moran(option: &SimpleFirmAverageAfter)
{
    rayon::ThreadPoolBuilder::new().num_threads(option.threads).build_global().unwrap();
    let dist: AnyDist = option.delay_dist.clone().into();
    let mut buf = option.get_buf(true);
    writeln!(buf, "#B average_mid average_end order av_time_mean_above slope").unwrap();
    let rng = Pcg64::seed_from_u64(option.seed);
    let rng = Mutex::new(rng);

    let delta = (option.other_buffer_max-option.other_buffer_min)/(option.other_buffer_steps-1) as f64;

    let samples_per_thread = option.average_samples / option.threads;
    let actual_samples = samples_per_thread * option.threads;
    dbg!(actual_samples);
    dbg!(samples_per_thread);

    let bar = crate::misc::indication_bar(option.other_buffer_steps as u64);

    let time_half = NonZeroU64::new(option.time.get() / 2).unwrap();

    let scan_buf_dist = option.get_scan_buf_dist();

    for i in (0..option.other_buffer_steps).progress_with(bar){
        let buffer = option.other_buffer_min + delta * i as f64;
        let global_sum_mid = Mutex::new(0.0);
        let global_sum_end = Mutex::new(0.0);
        let glob_time_above = AtomicU64::new(0);

        let bd = match scan_buf_dist{
            ScanBufDist::Const => {
                AnyBufDist::Constant(buffer)
            },
            ScanBufDist::Uniform(hw) => {
                let any_uniform = UniformDistCreator2{mean: buffer, half_width: hw.half_width};
                if !any_uniform.is_valid(){
                    continue;
                }
                let uniform = any_uniform.into();
                AnyBufDist::Any(AnyDistCreator::Uniform(uniform))
            },
            ScanBufDist::MinScan(ms) => {
                let c = ms.other_consts;
                AnyBufDist::ConstMin(BufferConstMin { buf_const: c, buf_min: buffer })
            }
        };
        let firm_creator = FocusFirmOuter::get_firm_creator(bd, dist.clone());

        (0..option.threads)
            .into_par_iter()
            .for_each(
                |_|
                {
                    let mut guard = rng.lock().unwrap();
                    let mut thread_rng = Pcg64::from_rng(guard.deref_mut()).unwrap();
                    drop(guard);
                    let mut sum_mid = 0.0;
                    let mut sum_end = 0.0;
                    
                    let mut time_above = 0;

                    let mut ids: Vec<_> = (0..option.k).collect();
                    let mut scratch = Vec::new();
                    let mut tmp = vec![0.0; ids.len()];
                    for _ in 0..samples_per_thread{
                        let f_rng = Pcg64::from_rng(&mut thread_rng).unwrap();
                        let mut firms = firm_creator(option.k, option.focus_buffer, f_rng);
                        
                        for _ in 0..time_half.get(){
                            firms.iterate_moran(&mut tmp, 5, &mut ids, &mut scratch);
                            let delay = firms.average_other_delay();
                            if delay > buffer {
                                time_above += 1;
                            }
                        }
                        sum_mid += firms.average_other_delay();
                        for _ in 0..time_half.get(){
                            firms.iterate_moran(&mut tmp, 5, &mut ids, &mut scratch);
                            let delay = firms.average_other_delay();
                            if delay > buffer {
                                time_above += 1;
                            }
                        }
                        sum_end += firms.average_other_delay();
                    }

                    glob_time_above.fetch_add(time_above, std::sync::atomic::Ordering::Relaxed);
                    
                    let mut guard = global_sum_mid.lock().unwrap();
                    *(guard.deref_mut()) += sum_mid;
                    drop(guard);

                    let mut guard: std::sync::MutexGuard<'_, f64> = global_sum_end.lock().unwrap();
                    *(guard.deref_mut()) += sum_end;
                    drop(guard);
                }
            );
        let sum_mid: f64 = global_sum_mid.into_inner().unwrap();
        
        let average_mid = sum_mid / actual_samples as f64;

        let sum_end = global_sum_end.into_inner().unwrap();
        let average_end = sum_end / actual_samples as f64;


        let slope = (average_end-average_mid)/ (time_half.get() as f64);

        let order = sum_end/sum_mid;

        let av_time_above = (glob_time_above.into_inner() as f64 / actual_samples as f64) 
            / (time_half.get() * 2) as f64;

        writeln!(buf, "{} {} {} {} {av_time_above} {slope}", buffer, average_mid, average_end, order).unwrap();
    }
    

}

pub fn recreate_moran_avalanch(option: &SimpleFirmAverageAfter)
{
    rayon::ThreadPoolBuilder::new().num_threads(option.threads).build_global().unwrap();
    let dist: AnyDist = option.delay_dist.clone().into();
    let mut buf = option.get_buf(true);
    writeln!(buf, "#B average_mid average_end order av_time_mean_above slope average_avalanch_duration average_avalanch_count average_avalanch_size").unwrap();
    let rng = Pcg64::seed_from_u64(option.seed);
    let rng = Mutex::new(rng);

    let delta = (option.other_buffer_max-option.other_buffer_min)/(option.other_buffer_steps-1) as f64;

    let samples_per_thread = option.average_samples / option.threads;
    let actual_samples = samples_per_thread * option.threads;
    dbg!(actual_samples);
    dbg!(samples_per_thread);

    let bar = crate::misc::indication_bar(option.other_buffer_steps as u64);

    let time_half = NonZeroU64::new(option.time.get() / 2).unwrap();

    let scan_buf_dist = option.get_scan_buf_dist();

    for i in (0..option.other_buffer_steps).progress_with(bar){
        let buffer = option.other_buffer_min + delta * i as f64;
        let global_sum_mid = Mutex::new(0.0);
        let global_sum_end = Mutex::new(0.0);
        let glob_time_above = AtomicU64::new(0);
        let glob_avalanch_len_sum = AtomicU64::new(0);
        let glob_avalanch_counter = AtomicU64::new(0);
        let glob_avalanch_size_sum = Mutex::new(0.0);

        let bd = match scan_buf_dist{
            ScanBufDist::Const => {
                AnyBufDist::Constant(buffer)
            },
            ScanBufDist::Uniform(hw) => {
                let any_uniform = UniformDistCreator2{mean: buffer, half_width: hw.half_width};
                if !any_uniform.is_valid(){
                    continue;
                }
                let uniform = any_uniform.into();
                AnyBufDist::Any(AnyDistCreator::Uniform(uniform))
            },
            ScanBufDist::MinScan(ms) => {
                let c = ms.other_consts;
                AnyBufDist::ConstMin(BufferConstMin { buf_const: c, buf_min: buffer })
            }
        };
        let firm_creator = FocusFirmOuter::get_firm_creator(bd, dist.clone());

        (0..option.threads)
            .into_par_iter()
            .for_each(
                |_|
                {
                    let mut guard = rng.lock().unwrap();
                    let mut thread_rng = Pcg64::from_rng(guard.deref_mut()).unwrap();
                    drop(guard);
                    let mut sum_mid = 0.0;
                    let mut sum_end = 0.0;
                    
                    let mut time_above = 0;

                    let mut ids: Vec<_> = (0..option.k).collect();
                    let mut scratch = Vec::new();
                    let mut tmp = vec![0.0; ids.len()];
                    let mut avalanch_counter = 0_u64;
                    let mut avalanch_len_sum = 0_u64;
                    let mut avalanch_len = 0;
                    let mut avalanch_size = 0.0;
                    let mut avalanch_size_sum = 0.0;
                    for _ in 0..samples_per_thread{
                        let f_rng = Pcg64::from_rng(&mut thread_rng).unwrap();
                        let mut firms = firm_creator(option.k, option.focus_buffer, f_rng);
                        
                        for _ in 0..time_half.get(){
                            firms.iterate_moran(&mut tmp, 5, &mut ids, &mut scratch);
                            let delay = firms.average_other_delay();
                            if delay > buffer {
                                time_above += 1;
                                avalanch_len += 1;
                                avalanch_size += delay - buffer;
                            } else if avalanch_len > 0 {
                                avalanch_len_sum += avalanch_len;
                                avalanch_len = 0;
                                avalanch_counter += 1;
                                avalanch_size_sum += avalanch_size;
                                avalanch_size = 0.0;
                            }
                        }
                        sum_mid += firms.average_other_delay();
                        for _ in 0..time_half.get(){
                            firms.iterate_moran(&mut tmp, 5, &mut ids, &mut scratch);
                            let delay = firms.average_other_delay();
                            if delay > buffer {
                                time_above += 1;
                                avalanch_len += 1;
                                avalanch_size += delay - buffer;
                            } else if avalanch_len > 0 {
                                avalanch_len_sum += avalanch_len;
                                avalanch_len = 0;
                                avalanch_counter += 1;
                                avalanch_size_sum += avalanch_size;
                                avalanch_size = 0.0
                            }
                        }
                        if avalanch_len > 0 {
                            avalanch_len_sum += avalanch_len;
                            avalanch_counter += 1;
                            avalanch_size_sum += avalanch_size;
                        }
                        avalanch_len = 0;
                        avalanch_size = 0.0;
                        sum_end += firms.average_other_delay();
                    }

                    glob_avalanch_counter.fetch_add(avalanch_counter, Relaxed);
                    glob_avalanch_len_sum.fetch_add(avalanch_len_sum, Relaxed);
                    glob_time_above.fetch_add(time_above, Relaxed);
                    
                    let mut guard = global_sum_mid.lock().unwrap();
                    *(guard.deref_mut()) += sum_mid;
                    drop(guard);

                    let mut guard = global_sum_end.lock().unwrap();
                    *(guard.deref_mut()) += sum_end;
                    drop(guard);

                    let mut guard = glob_avalanch_size_sum.lock().unwrap();
                    *(guard.deref_mut()) += avalanch_size_sum;
                    drop(guard);
                }
            );
        let sum_mid: f64 = global_sum_mid.into_inner().unwrap();
        
        let average_mid = sum_mid / actual_samples as f64;

        let sum_end = global_sum_end.into_inner().unwrap();
        let average_end = sum_end / actual_samples as f64;

        let avalanch_count = glob_avalanch_counter.into_inner() as f64;
        let avalanch_len_sum = glob_avalanch_len_sum.into_inner();
        let average_avalanch_len = avalanch_len_sum as f64 / avalanch_count;
        let average_avalanch_count = avalanch_count / actual_samples as f64;


        let slope = (average_end-average_mid)/ (time_half.get() as f64);

        let order = sum_end/sum_mid;

        let av_time_above = (glob_time_above.into_inner() as f64 / actual_samples as f64) 
            / (time_half.get() * 2) as f64;

        let average_avalanch_size = glob_avalanch_size_sum.into_inner().unwrap() / actual_samples as f64;

        writeln!(buf, "{buffer} {average_mid} {average_end} {order} {av_time_above} {slope} {average_avalanch_len} {average_avalanch_count} {average_avalanch_size}").unwrap();
    }
    

}

pub fn sample_simple_firm_buffer_hist(option: &SimpleFirmBufferHistogram)
{
    let dist: AnyDist = option.delay_dist.clone().into();

    let buf_dist: UniformDistCreator = option.buf_dist.into();
    let buf_min = buf_dist.min;
    let buf_max = buf_dist.max;
    let buf_dist = AnyBufDist::Any(AnyDistCreator::Uniform(buf_dist));
    let firm_creator = FocusFirmOuter::get_firm_creator(buf_dist, dist);

    let rng = Pcg64::seed_from_u64(option.seed);
    let rng_arc = Mutex::new(rng);

    let hist = AtomicHistF64::new(buf_min, buf_max, option.bins)
        .expect("uable to create hist");

    let spt = option.samples / option.threads;
    let actual_samples = spt * option.threads;

    (0..option.threads)
        .into_par_iter()
        .for_each(
            |_|
            {
                let mut thread_rng = rng_arc.lock().unwrap();
                let mut rng = Pcg64::from_rng(thread_rng.deref_mut())
                    .unwrap();
                drop(thread_rng);
                for _ in 0..spt{
                    let firm_rng = Pcg64::from_rng(&mut rng)
                        .unwrap();
                    let mut firms = firm_creator(option.k, option.focus_buffer, firm_rng);
                    for _ in 0..option.iter_limit.get(){
                        firms.iterate();
                    }
                    let others = firms.focus_firm.others.as_slice();
                    let mut max = others[0].current_delay;
                    let mut buf = others[0].buffer;
                    for f in &others[1..]
                    {
                        if f.current_delay > max {
                            max = f.current_delay;
                            buf = f.buffer;
                        }
                    }
                    hist.increment_quiet(buf);
                }
            }
        );

    let mut buf = option.get_buf();

    for ([left, right], bin_hits) in hist.bin_hits_iter()
    {
        let hits = bin_hits as f64 / actual_samples as f64;
        writeln!(buf, "{} {} {}", left, right, hits).unwrap();
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