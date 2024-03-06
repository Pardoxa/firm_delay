use std::io::Write;

pub fn write_digraph<'a, W, I>(mut writer: W, network: &'a [I])
where W: Write,
    &'a I: IntoIterator<Item = &'a usize>
{
    writeln!(
        writer,
        "digraph {{"
    ).unwrap();
    for idx in 0..network.len()
    {
        writeln!(writer, "{idx}").unwrap();
    }
    for (idx, adj) in network.iter().enumerate()
    {
        for j in adj {
            writeln!(
                writer,
                "{idx} -> {j}"
            ).unwrap();
        }
    }
    writeln!(
        writer,
        "}}"
    ).unwrap();
}