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

pub fn write_my_digraph<W>(mut writer: W, nodes: &[Node], direction_parent: bool)
where W: Write
{
    let fun = if direction_parent{
        write_dir_parent
    } else {
        write_dir_child
    };
    writeln!(
        writer,
        "digraph {{"
    ).unwrap();
    for idx in 0..nodes.len(){
        writeln!(writer, "{idx}").unwrap();
    }
    for (idx, node) in nodes.iter().enumerate(){
        fun(idx, node, &mut writer);
    }
    writeln!(
        writer,
        "}}"
    ).unwrap();
}

fn write_dir_child<W>(idx: usize, node: &Node, writer: &mut W)
where W: Write
{
    for child in node.children.iter(){
        writeln!(
            writer,
            "{idx} -> {child}"
        ).unwrap();
    }
}

fn write_dir_parent<W>(idx: usize, node: &Node, writer: &mut W)
where W: Write
{
    for parent in node.parents.iter(){
        let parent = parent.node_idx;
        writeln!(
            writer,
            "{idx} -> {parent}"
        ).unwrap();
    }
}