use std::io::Write;

use crate::my_model::model::Node;

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

pub fn write_my_digraph<W>(mut writer: W, nodes: &[Node])
where W: Write
{
    writeln!(
        writer,
        "digraph {{"
    ).unwrap();
    for idx in 0..nodes.len(){
        writeln!(writer, "{idx}").unwrap();
    }
    for (idx, node) in nodes.iter().enumerate(){
        for child in node.children.iter(){
            writeln!(
                writer,
                "{idx} -> {child}"
            ).unwrap();
        }
    }
    writeln!(
        writer,
        "}}"
    ).unwrap();
}

pub fn write_my_digraph_dir_parent<W>(mut writer: W, nodes: &[Node])
where W: Write
{
    writeln!(
        writer,
        "digraph {{"
    ).unwrap();
    for idx in 0..nodes.len(){
        writeln!(writer, "{idx}").unwrap();
    }
    for (idx, node) in nodes.iter().enumerate(){
        for parent in node.parents.iter(){
            let parent = parent.node_idx;
            writeln!(
                writer,
                "{idx} -> {parent}"
            ).unwrap();
        }
    }
    writeln!(
        writer,
        "}}"
    ).unwrap();
}