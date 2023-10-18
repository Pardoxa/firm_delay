
use clap::Parser;
use global_opts::CmdChooser;
use misc::parse;
use simple_firm::SimpleFirmDifferentKOpts;

pub mod misc;
mod global_opts;
mod simple_firm;

fn main() {
    
   // measure_end()
    let command = CmdChooser::parse();

    match command{
        CmdChooser::SimpleFirmDifK(opt) => {
            let o: SimpleFirmDifferentKOpts = parse(opt.json);
            o.exec();
        }
    }
    
}

/*fn measure_end()
{
    let file = File::create("measureb0.48L.dat").unwrap();
    let mut buf = BufWriter::new(file);

    writeln!(buf, "#k currentDelay variance_time var_div_av_time").unwrap();
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
}*/


