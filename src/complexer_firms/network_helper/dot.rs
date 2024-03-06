use std::io::Write;

pub fn write_digraph<W>(mut writer: W, network: &[Vec<usize>])
where W: Write
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