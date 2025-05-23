use {
    flate2::bufread::GzDecoder, 
    fs_err::File, 
    indicatif::{ProgressBar, ProgressStyle}, 
    serde::{de::DeserializeOwned, Deserialize, Serialize}, 
    serde_json::Value, 
    std::{
        fmt::Display, 
        io::{stdout, BufRead, BufReader, BufWriter, Write}, 
        num::*, 
        path::Path, 
        process::{exit, Command, Output}, 
        sync::RwLock
    }
};
mod cleaner;
pub use cleaner::*;

mod ratio_iters;
pub use ratio_iters::*;
mod sync_queue;
pub use sync_queue::*;

pub static GLOBAL_ADDITIONS: RwLock<Option<String>> = RwLock::new(None);
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn open_bufreader<P>(path: P) -> BufReader<File>
where P: AsRef<Path>
{
    let p = path.as_ref();
    let file = match File::open(p){
        Err(e) => panic!("Unable to open {p:?} - encountered {e:?}"),
        Ok(file) =>  file
    };
    BufReader::new(file)
}

pub fn open_gz_bufreader<P>(path: P) -> impl BufRead
where P: AsRef<Path>
{
    let buf = open_bufreader(path);
    let decoder = GzDecoder::new(buf);
    BufReader::new(decoder)
}

pub fn open_as_unwrapped_lines<P>(path: P) -> impl Iterator<Item = String>
where P: AsRef<Path>
{
    open_bufreader(path)
        .lines()
        .map(Result::unwrap)
}

pub fn open_gz_as_unwrapped_lines<P>(path: P) -> impl Iterator<Item = String>
where P: AsRef<Path>
{
    open_gz_bufreader(path)
        .lines()
        .map(Result::unwrap)
}

pub fn open_as_unwrapped_lines_filter_comments<P>(path: P) -> impl Iterator<Item = String>
where P: AsRef<Path>
{
    open_as_unwrapped_lines(path)
        .filter(|line| !line.starts_with('#'))
}

pub fn open_gz_as_unwrapped_lines_filter_comments<P>(path: P) -> impl Iterator<Item = String>
where P: AsRef<Path>
{
    open_gz_as_unwrapped_lines(path)
        .filter(|line| !line.starts_with('#'))
}

pub fn write_json<W: Write>(mut writer: W, json: &Value)
{
    write!(writer, "#").unwrap();
    serde_json::to_writer(&mut writer, json).unwrap();
    writeln!(writer).unwrap();
}

pub fn indication_bar(len: u64) -> ProgressBar
{
        // for indication on when it is finished
        let bar = ProgressBar::new(len);
        bar.set_style(ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise} - {eta_precise}] {wide_bar}")
            .unwrap()
        );
        bar
}
pub fn create_buf<P>(path: P) -> BufWriter<File>
where P: AsRef<Path>
{
    let file = File::create(path.as_ref())
        .expect("Unable to create file");
    BufWriter::new(file)
}

pub fn write_commands<W: Write>(mut w: W) -> std::io::Result<()>
{
    write!(w, "#")?;
    for arg in std::env::args()
    {
        write!(w, " {arg}")?;
    }
    writeln!(w)
}

pub fn write_commands_and_version<W: Write>(mut w: W) -> std::io::Result<()>
{
    writeln!(w, "# {VERSION}")?;
    writeln!(w, "# Git Hash: {} Compile-time: {}", env!("GIT_HASH"), env!("BUILD_TIME_CHRONO"))?;
    let l = GLOBAL_ADDITIONS.read().unwrap();
    if let Some(add) = l.as_deref(){
        writeln!(w, "# {add}")?;
    }
    drop(l);
    write_commands(w)
}

#[must_use]
pub fn create_buf_with_command_and_version<P>(path: P) -> BufWriter<File>
where P: AsRef<Path>
{
    let mut buf = create_buf(path);
    write_commands_and_version(&mut buf)
        .expect("Unable to write Version and Command in newly created file");
    buf
}

pub fn create_buf_with_command_and_version_and_header<P, S, D>(path: P, header: S) -> BufWriter<File>
where P: AsRef<Path>,
    S: IntoIterator<Item=D>,
    D: Display
{
    let mut buf = create_buf_with_command_and_version(path);
    write_slice_head(&mut buf, header)
        .expect("unable to write header");
    buf
}

pub fn write_slice_head<W, S, D>(mut w: W, slice: S) -> std::io::Result<()>
where W: std::io::Write,
    S: IntoIterator<Item=D>,
    D: Display
{
    write!(w, "#")?;
    for (s, i) in slice.into_iter().zip(1_u16..){
        write!(w, " {s}_{i}")?;
    }
    writeln!(w)
}

pub fn create_gnuplot_buf<P>(path: P) -> BufWriter<File>
where P: AsRef<Path>
{
    let mut buf = create_buf_with_command_and_version(path);
    writeln!(buf, "reset session").unwrap();
    writeln!(buf, "set encoding utf8").unwrap();
    buf
}

pub fn  call_gnuplot(gp_file_name: &str) -> Output
{
    let out = Command::new("gnuplot")
        .arg(gp_file_name)
        .output()
        .unwrap();
    if !out.status.success(){
        eprintln!("FAILED: {gp_file_name}!");
        dbg!(&out);
    }
    out
}

pub fn create_video(
    glob: &str, 
    out_stub: &str, 
    framerate: u8, 
    also_convert: bool
)
{
    let program = "ffmpeg";
    let out = format!("{out_stub}.mp4");

    let _ = std::fs::remove_file(&out);

    let video_out = Command::new(program)
        .arg("-f")
        .arg("image2")
        .arg("-r")
        .arg(framerate.to_string())
        .arg("-pattern_type")
        .arg("glob")
        .arg("-i")
        .arg(glob)
        .arg("-vcodec")
        .arg("libx264")
        .arg("-crf")
        .arg("22")
        .arg(&out)
        .output()
        .unwrap();

    assert!(video_out.status.success());

    if also_convert{
        let mut c = Command::new(program);
        let new_name = format!("{out_stub}_conv.mp4");
        let _ = std::fs::remove_file(&new_name);
        let args = [
            "-i",
            &out,
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
            &new_name
        ];
        c
            .args(args);
        let output = c.output().unwrap();
        assert!(output.status.success());
    }

}

pub fn parse_and_add_to_global<P, T>(file: Option<P>) -> T
where P: AsRef<Path>,
    T: Default + Serialize + DeserializeOwned
{
    match file
    {
        None => {
            let example = T::default();
            serde_json::to_writer_pretty(
                std::io::stdout(),
                &example
            ).expect("Unable to reach stdout");
            exit(0)
        }, 
        Some(file) => {
            let f = File::open(file.as_ref())
                .expect("Unable to open file");
            let buf = BufReader::new(f);

            let json_val: Value = match serde_json::from_reader(buf)
            {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("json parsing error!");
                    dbg!(e);
                    exit(1);
                }
            };

            let opt: T = match serde_json::from_value(json_val.clone())
            {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("json parsing error!");
                    dbg!(e);
                    exit(1);
                }
            };
            let s = serde_json::to_string(&opt).unwrap();
            let mut w = GLOBAL_ADDITIONS.write().unwrap();
            *w = Some(s);
            drop(w);

            opt  
        }
    }
}

pub fn parse_without_add_to_global<P, T>(file: Option<P>) -> T
where P: AsRef<Path>,
    T: Default + Serialize + DeserializeOwned
{
    match file
    {
        None => {
            let example = T::default();
            serde_json::to_writer_pretty(
                std::io::stdout(),
                &example
            ).expect("Unable to reach stdout");
            exit(0)
        }, 
        Some(file) => {
            let f = File::open(file.as_ref())
                .expect("Unable to open file");
            let buf = BufReader::new(f);

            let json_val: Value = match serde_json::from_reader(buf)
            {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("json parsing error!");
                    dbg!(e);
                    exit(1);
                }
            };

            let opt: T = match serde_json::from_value(json_val.clone())
            {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("json parsing error!");
                    dbg!(e);
                    exit(1);
                }
            };

            opt  
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct RequestedTime
{
    pub seconds: Option<NonZeroU64>,
    pub minutes: Option<NonZeroU64>,
    pub hours: Option<NonZeroU64>,
    pub days: Option<NonZeroU64>
}

impl RequestedTime
{
    #[allow(dead_code)]
    pub fn in_seconds(&self) -> u64
    {
        let mut time = self.seconds.map_or(0, NonZeroU64::get);
        if let Some(min) = self.minutes
        {
            time += min.get() * 60;
        }

        if let Some(h) = self.hours
        {
            time += h.get() * (60*60);
        }

        if let Some(d) = self.days
        {
            time += d.get() * (24*60*60);
        }

        if time == 0 {
            eprintln!("Time is zero! That is invalid! Fatal Error");
            panic!("No Time")
        }

        time
    }
}

pub trait PrintAlternatives{
    fn print_alternatives(layer: u8);
}

pub(crate) fn print_spaces(layer: u8){
    for _ in 0..layer{
        print!(" ");
    }
}

#[derive(Debug, Clone, Serialize, Default, Deserialize)]
pub struct SampleRangeF64
{
    pub start: f64,
    pub end: f64,
    pub samples: usize
}

impl SampleRangeF64{
    pub fn get_iter(&'_ self) -> impl Iterator<Item=f64> + '_
    {
        let delta = (self.end - self.start) / (self.samples - 1) as f64;
        (0..self.samples-1)
            .map(
                move |i|
                {
                    self.start + delta * i as f64
                }
            ).chain(std::iter::once(self.end))
    }
}


#[derive(Clone, Copy, Debug)]
pub struct Stats{
    pub average: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}

impl Stats{
    pub fn get_std_dev(&self) -> f64
    {
        self.variance.sqrt()
    }

    pub fn get_cv(&self) -> f64
    {
        self.get_std_dev() / self.average
    }

    pub fn from_ref_iter<'a, T: IntoIterator<Item = &'a f64>>(iter: T) -> Self {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut counter = 0_u64;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;

        iter.into_iter()
            .for_each(
                |v| 
                {
                    sum += v;
                    sum_sq = v.mul_add(*v, sum_sq);
                    counter += 1;
                    min = min.min(*v);
                    max = max.max(*v);
                }
            );



        let factor = (counter as f64).recip();
        let average = sum * factor;
        let variance = sum_sq * factor - average * average;
        Self { average, variance, min, max }
    }
}


pub fn print_alternatives_helper<'a, E, I>(iter: I, layer: u8, name: &'a str)
where I: IntoIterator<Item = &'a E>,
    E: Serialize + 'a
{
    let mut stdout = stdout();
    let msg = "Serialization issue";
    for (idx, dist) in iter.into_iter().enumerate(){
        print_spaces(layer);
        println!("{name} {idx})");
        serde_json::to_writer_pretty(&mut stdout, dist)
            .expect(msg);
        println!();
    }
}