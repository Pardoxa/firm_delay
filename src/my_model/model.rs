use std::{io::Write, num::*, path::Path};
use itertools::Itertools;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use super::opts::*;

use crate::misc::*;

use crate::Cleaner;
const STOCK_CAPACITY: f64 = 1.0;

#[derive(Debug, Clone, Copy)]
pub struct IndexHelper{
    node_idx: usize,
    // Idx of stock and available product
    internal_idx: usize
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StockAvailItem{
    stock: f64,
    currently_avail: f64,
    demand_passed_on: f64
}

#[derive(Debug, Clone)]
pub struct Node{
    children: Vec<usize>,
    parents: Vec<IndexHelper>
}



#[inline]
fn set_const(slice: &mut [f64], c: f64)
{
    slice.iter_mut()
        .for_each(|val| *val = c);
}

fn calc_leaf_order(slice: &[Node]) -> Vec<usize>
{
    // add leafs to stack
    let mut stack = slice.iter()
        .enumerate()
        .filter_map(
            |(idx, node)|
            {
                node.children
                    .is_empty()
                    .then_some(idx)
            }
        ).collect_vec();
    let mut counter = vec![0; slice.len()];

    let mut order = Vec::new();
    while let Some(node_idx) = stack.pop()
    {
        order.push(node_idx);
        for parent in &slice[node_idx].parents{
            counter[parent.node_idx] += 1;
            if slice[parent.node_idx].children.len() == counter[parent.node_idx]
            {
                stack.push(parent.node_idx);
            }
        }
    }
    order
}

fn calc_root_order(slice: &[Node]) -> Vec<usize>
{
    let mut counter = vec![0; slice.len()];
    let mut stack = vec![0];
    let mut order = Vec::new();
    while let Some(node_idx) = stack.pop()
    {
        order.push(node_idx);
        for &child in slice[node_idx].children.iter()
        {
            counter[child] += 1;
            let node = &slice[child];
            if counter[child] == node.parents.len(){
                stack.push(child);
            }
        }
    }
    assert_eq!(slice.len(), order.len());
    order
}

pub fn test_profile()
{
    let demand = 0.5;
    let size = 12;
    let mut model = Model::new_chain(
        size, 
        NonZeroUsize::new(1).unwrap(), 
        234073, 
        demand
    );
    let name = "test_profile.dat";
    let mut header = vec!["time_step".to_string()];
    for i in 0..size - 1{
        header.push(format!("d_{i}"));
        header.push(format!("I_{i}"));
        header.push(format!("k_{i}"));
        header.push(format!("a_{i}"));
    }
    header.push(format!("d_{}", size-1));
    header.push(format!("I_{}", size-1));
    let mut buf = create_buf_with_command_and_version_and_header(name, header);
    for t in 1..200{

        model.update_demand();
        model.update_production();

        write!(buf, "{t} ").unwrap();
        for i in 0..size - 1{
            write!(
                buf,
                "{} {} {} {} ",
                model.current_demand[i],
                model.currently_produced[i],
                model.stock_avail[i][0].stock,
                model.stock_avail[i][0].currently_avail,
            ).unwrap();
        }
        writeln!(
            buf,
            "{} {}",
            model.current_demand[size-1],
            model.currently_produced[size-1]
        ).unwrap();
        
    }
}


pub fn chain_crit_scan(opt: DemandVelocityCritOpt, out: &str)
{
    let zeros = "000000000";
    let mut current_chain_len = opt.chain_start.get();
    let cleaner = Cleaner::new();

    let header = [
        "chain_len",
        "a",
        "b",
        "critical_root_demand"
    ];

    let mut crit_buf = create_buf_with_command_and_version_and_header(out, header);

    loop {
        let i_name = current_chain_len.to_string();
        let start = i_name.len();
        let zeros = &zeros[start..];
        let name = format!("TMP_{zeros}{i_name}{out}.dat");

        let mut m_opt = opt.opts.clone();
        m_opt.chain_length = NonZeroUsize::new(current_chain_len).unwrap();

        chain_calc_demand_velocity(m_opt, &name);
        let gp_name = format!("{name}.gp");

        let mut gp_writer = create_gnuplot_buf(&gp_name);
        let png = format!("{name}.png");
        writeln!(gp_writer, "set t pngcairo").unwrap();
        writeln!(gp_writer, "set output '{png}'").unwrap();
        writeln!(gp_writer, "set title 'N={current_chain_len}'").unwrap();
        writeln!(gp_writer, "set ylabel 'v'").unwrap();
        writeln!(gp_writer, "set xlabel 'r'").unwrap();
        writeln!(gp_writer, "set fit quiet").unwrap();
        writeln!(gp_writer, "t(x)=x>0.01?0.00000000001:10000000").unwrap();
        writeln!(gp_writer, "f(x)=a*x+b").unwrap();
        writeln!(gp_writer, "fit f(x) '{name}' u 1:2:(t($2)) yerr via a,b").unwrap();
        
        if let Some(range) = &opt.y_range{
            writeln!(gp_writer, "set yrange [{}:{}]", range.start(), range.end()).unwrap();
        }
        writeln!(gp_writer, "p '{name}' t '', f(x)").unwrap();
        writeln!(gp_writer, "print(b)").unwrap();
        writeln!(gp_writer, "print(a)").unwrap();
        writeln!(gp_writer, "set output").unwrap();
        drop(gp_writer);
        let out = call_gnuplot(&gp_name);
        if out.status.success(){
            let s = String::from_utf8(out.stderr)
                .unwrap();
        
            let mut iter = s.lines();
                
            let b: f64 = iter.next().unwrap().parse().unwrap();
            let a: f64 = iter.next().unwrap().parse().unwrap();
            let crit = -b/a;
            
            cleaner.add_multi([name, gp_name, png]);

            writeln!(
                crit_buf,
                "{} {} {} {}",
                current_chain_len,
                a,
                b,
                crit
            ).unwrap();
        }

        current_chain_len += opt.chain_step.get();
        if current_chain_len > opt.chain_end.get(){
            break;
        }
    }
    create_video(
        "TMP_*.png", 
        out, 
        15,
        true
    );

    cleaner.clean();
}



pub fn chain_calc_demand_velocity<P>(opt: DemandVelocityOpt, out: P)
where P: AsRef<Path>
{
    if let Some(t) = opt.threads{
        rayon::ThreadPoolBuilder::new().num_threads(t.get()).build_global().unwrap();
    }
    let mut rng = Pcg64::seed_from_u64(opt.seed);
    let ratio = RatioIter::from_float(
        opt.root_demand_rate_min, 
        opt.root_demand_rate_max, 
        opt.root_demand_samples
    );
    let ratios: Vec<_> = ratio.float_iter()
        .map(
            |ratio|
            {
                let rng = Pcg64::from_rng(&mut rng).unwrap();
                Model::new_multi_chain_from_rng(
                    opt.num_chains,
                    opt.chain_length.get() - 1, 
                    rng, 
                    ratio
                )
            }
        )
        .collect();

    let velocities: Vec<_> = ratios.into_par_iter()
        .map(
            |mut model|
            {
                let mut sum = 0.0;
                for _ in 0..opt.samples.get(){
                    model.reset_delays();
                    for _ in 0..opt.time.get(){
                        model.update_demand();
                        model.update_production();
                    }
                    sum += model.current_demand[0] / opt.time.get() as f64;
                }
                sum /= opt.samples.get() as f64;
                sum
            }
        ).collect();

    let header = [
        "root_demand",
        "velocity"
    ];

    let mut buf = create_buf_with_command_and_version_and_header(out, header);
    for (root_demand, velocity) in ratio.float_iter().zip(velocities.iter())
    {
        writeln!(
            buf,
            "{root_demand} {velocity}"
        ).unwrap();
    }
}

#[allow(non_snake_case)]
pub struct Model{
    nodes: Vec<Node>,
    currently_produced: Vec<f64>,
    current_demand: Vec<f64>,
    root_order: Vec<usize>,
    leaf_order: Vec<usize>,
    rng: Pcg64,
    stock_avail: Vec<Vec<StockAvailItem>>,
    demand_at_root: f64
}

impl Model{
    pub fn reset_delays(&mut self) 
    {
        set_const(&mut self.current_demand, 0.0);
        set_const(&mut self.currently_produced, 0.0);
        self.stock_avail
            .iter_mut()
            .flat_map(|lists| lists.iter_mut())
            .for_each(
                |item|
                {
                    item.currently_avail = 0.0;
                    item.demand_passed_on = 0.0;
                    item.stock = 0.0;
                }
            );
    }

    fn new_multi_chain_from_rng(
        num_chains: NonZeroUsize,
        other_chain_len: usize, 
        rng: Pcg64, 
        demand_at_root: f64
    ) -> Self{
        if other_chain_len == 0 {
            return Self{
                rng,
                nodes: vec![Node{children: Vec::new(), parents: Vec::new()}],
                current_demand: vec![0.0],
                currently_produced: vec![0.0],
                root_order: vec![0],
                leaf_order: vec![0],
                stock_avail: vec![Vec::new()],
                demand_at_root
            };
        }
        let total_node_count = 1 + other_chain_len * num_chains.get();
        let mut nodes = Vec::with_capacity(total_node_count);
        let first = Node{
            children: Vec::new(),
            parents: Vec::new()
        };
        nodes.push(first);
        let mut stock_avail = vec![Vec::new(); total_node_count];
        for _ in 0..num_chains.get(){
            
            let idx = nodes[0].children.len();
            let chain_first = Node{
                children: vec![],
                parents: vec![IndexHelper{internal_idx: idx, node_idx: 0}]
            };
            let c_idx = nodes.len();
            nodes.push(chain_first);
            nodes[0].children.push(c_idx);
            stock_avail[0].push(StockAvailItem::default());
            for _ in 0..other_chain_len-1{
                let last_idx = nodes.len() - 1;
                let this_idx = nodes.len();
                let node = Node{
                    children: Vec::new(),
                    parents: vec![IndexHelper{internal_idx: 0, node_idx: last_idx}]
                };
                nodes.push(node);
                nodes[last_idx].children.push(this_idx);
                stock_avail[last_idx].push(StockAvailItem::default());
            }
        }
        let root_order = calc_root_order(&nodes);
        let leaf_order = calc_leaf_order(&nodes);

        Self{
            rng,
            nodes,
            current_demand: vec![0.0; total_node_count],
            currently_produced: vec![0.0; total_node_count],
            root_order,
            leaf_order,
            stock_avail,
            demand_at_root
        }
    }

    fn new_chain(
        size: usize, 
        num_chains: NonZeroUsize,
        seed: u64, 
        demand_at_root: f64,
    ) -> Self{
        Self::new_multi_chain_from_rng(
            num_chains,
            size - 1,
            Pcg64::seed_from_u64(seed), 
            demand_at_root
        )
    }


    fn update_demand(&mut self)
    {
        
        self.current_demand[0] += self.demand_at_root;
        set_const(&mut self.current_demand[1..], 0.0);
        for &idx in self.root_order.iter()
        {
            let demand = self.current_demand[idx];
            for (&child, stock_item) in self.nodes[idx].children.iter().zip(self.stock_avail[idx].iter_mut()){
                stock_item.demand_passed_on = (demand - stock_item.stock).max(0.0);
                self.current_demand[child] += stock_item.demand_passed_on;
            }
        }
    }

    fn update_production(&mut self)
    {
 
        self.stock_avail
            .iter_mut()
            .for_each(
                |avail|
                {
                    // only for now as a sanity check:
                    avail.iter_mut()
                        .for_each(|item| item.currently_avail = f64::NAN);
                }
            );
        set_const(&mut self.currently_produced, 0.0);
        let uniform = Uniform::new_inclusive(0.0, 1.0);

        for (&idx, rand) in self.leaf_order.iter().zip(uniform.sample_iter(&mut self.rng)){
            // firstly calculate actual production
            let production = &mut self.currently_produced[idx];
            let this_demand = self.current_demand[idx];
            *production = this_demand.min(rand);
            let iter = self.stock_avail[idx].iter();
            for StockAvailItem{currently_avail: avail, stock, ..} in iter {
                *production = production.min(avail + stock);
            }
            // next calculate new stocks
            let iter = self.stock_avail[idx]
                .iter_mut();
            for item in iter {
                item.stock = STOCK_CAPACITY.min(item.currently_avail + item.stock - *production);
            }

            if this_demand <= 0.0 {
                for parent in self.nodes[idx].parents.iter(){
                    let stock = &mut self.stock_avail[parent.node_idx][parent.internal_idx];
                    stock.currently_avail = 0.0;
                }   
            } else {
                for parent in self.nodes[idx].parents.iter(){
                    let stock = &mut self.stock_avail[parent.node_idx][parent.internal_idx];
                    stock.currently_avail = *production * stock.demand_passed_on / this_demand;
                }
            }
            
        }
        self.current_demand[0] = 0.0_f64.max(self.current_demand[0] - self.currently_produced[0]);
    }
}
