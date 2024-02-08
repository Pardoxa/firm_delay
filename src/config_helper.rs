use std::io::{stdout, Write};

use camino::Utf8PathBuf;
use clap::Parser;
use glob::glob;

use crate::{
    complexer_firms::AutoOpts, 
    create_buf, 
    global_opts::SubAutoBufSwaper, 
    open_as_unwrapped_lines_filter_comments,
    parse_without_add_to_global
};



pub fn exec_auto_sub_buf_swapper(opt: SubAutoBufSwaper)
{
    let opts: AutoOpts = parse_without_add_to_global(Some(opt.example_file));

    open_as_unwrapped_lines_filter_comments(opt.buffer_file)
        .filter(|line| !line.trim_start().is_empty())
        .for_each(
            |line|
            {
                let buf = line.trim();
                let float: f64 = buf.parse().unwrap();
                let name = format!("{buf}.json");
                let writer = create_buf(name);
                let mut o = opts.clone();
                o.mean_opts.buffer = float;
                if let Some(sub) = opt.change_sub_prob
                {
                    o.mean_opts.substitution_prob = sub;
                }
                serde_json::to_writer_pretty(writer, &o)
                    .unwrap();
            }
        )
}

#[derive(Parser, Debug)]
pub struct AutoBufSubJobOpts
{
    /// Glob to all the jsons
    pub glob: String,

    /// create file instead of printing to stdout
    pub out: Option<Utf8PathBuf>
}

fn exec_auto_buf_sub_job_creator_helper<W>(opt: AutoBufSubJobOpts, mut writer: W)
where W: Write
{
    let files = glob(&opt.glob)
        .unwrap()
        .map(Result::unwrap);
    let mut first = true;

    for file in files{
        if first {
            first = false;
        } else {
            writeln!(writer, "§").unwrap();
        }
        let utf8_path: Utf8PathBuf = file.try_into().unwrap();
        let opt: AutoOpts = parse_without_add_to_global(Some(&utf8_path));
        let filestem = utf8_path.file_stem().unwrap();
        let file_name = utf8_path.file_name().unwrap();
        let wanted_stub = format!("{filestem}_{}", opt.self_links.str());
        let j = opt.num_seeds;
        let command = format!(
            "time firm_delay sub auto -o {wanted_stub} -j {} --json {}",
            j,
            file_name
        );
        if let Some(parent) = utf8_path.parent()
        {
            writeln!(
                writer,
                "cd {parent:?}"
            ).unwrap();
        }
        
        writeln!(
            writer,
            "{command}"
        ).unwrap();
        writeln!(
            writer,
            "gzip {wanted_stub}*.time"
        ).unwrap();
        writeln!(
            writer,
            "firm_delay auto -g '{wanted_stub}*.time.gz' -s 10 -e 20 -j {j} -o cor_{wanted_stub} -m 60000 -c 0"
        ).unwrap();
    }
}

pub fn exec_auto_buf_sub_job_creator(opt: AutoBufSubJobOpts)
{
    match opt.out.as_deref()
    {
        Some(path) => {
            let buf = create_buf(path);
            exec_auto_buf_sub_job_creator_helper(opt, buf)
        },
        None => {
            exec_auto_buf_sub_job_creator_helper(opt, stdout())
        }
    }
}