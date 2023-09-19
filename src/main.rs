
use rand_pcg::Pcg64;
use rand::distributions::Uniform;
use rand::distributions::Distribution;
use rand::SeedableRng;
use std::io::Write;

use std::fs::File;
use std::io::BufWriter;

fn main() {
    
    measure_end()
    
    
}

fn measure_end()
{
    let file = File::create("measureb0.48L.dat").unwrap();
    let mut buf = BufWriter::new(file);
    for k in (1..10000).step_by(10){
        

        let mut firms = FocusFirm::new(k, 0.5, 0.48);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let time = 2000;
        for _ in 0..2000{
            firms.iterate();
            sum += firms.focus.current_delay;
            sum_sq += firms.focus.current_delay * firms.focus.current_delay;
        }
        let av = sum / (time as f64);
        let var = sum_sq/(time as f64) - av*av;
        let cv = var / av; 
        writeln!(buf, "{} {} {var} {cv}", k, firms.focus.current_delay).unwrap();
    }  
}

fn different_k()
{
    for k in [1, 10, 100, 1000, 10000]{
        
        let file = File::create(format!("k{k}.dat")).unwrap();
        let mut buf = BufWriter::new(file);

        let mut firms = FocusFirm::new(k, 0.5, 0.4);
        for i in 0..1000{
            write!(buf, "{} {}", i, firms.focus.current_delay).unwrap();
            //for f in firms.others.iter()
            //{
            //    write!(buf, " {}", f.current_delay).unwrap();
            //}
            writeln!(buf).unwrap();
            firms.iterate();
        }
    }
}


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
    pub fn new(k: usize, delta: f64, buffer: f64) -> Self
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
            rng: Pcg64::seed_from_u64(892356)
        }
    }

    pub fn iterate(&mut self)
    {
        let uniform = Uniform::new_inclusive(0.0, self.delta);
        let mut iter = uniform.sample_iter(&mut self.rng);

        let mut sum = 0.0;
        for firm in self.others.iter_mut()
        {
            sum += firm.current_delay;

            firm.current_delay = (firm.current_delay + iter.next().unwrap() - firm.buffer).max(0.0);
        }

        self.focus.current_delay = (self.focus.current_delay + iter.next().unwrap() - self.focus.buffer + sum).max(0.0);
        self.iteration += 1;
    }
}