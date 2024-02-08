use crate::{complexer_firms::AutoOpts, create_buf, global_opts::SubAutoBufSwaper, open_as_unwrapped_lines_filter_comments, parse_and_add_to_global};



pub fn exec_auto_sub_buf_swapper(opt: SubAutoBufSwaper)
{
    let opts: AutoOpts = parse_and_add_to_global(Some(opt.example_file));

    open_as_unwrapped_lines_filter_comments(opt.buffer_file)
        .filter(|line| !line.trim().is_empty())
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