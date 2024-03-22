use std::io::Write;
use itertools::Itertools;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64;

use crate::create_buf_with_command_and_version;


#[derive(Debug, Clone, Copy)]
pub struct IndexHelper{
    node_idx: usize,
    // Idx of stock and available product
    internal_idx: usize
}

#[derive(Clone, Copy, Debug)]
pub struct StockAvailItem{
    stock: f64,
    currently_avail: f64,
}

#[derive(Debug, Clone)]
pub struct Node{
    children: Vec<usize>,
    parents: Vec<IndexHelper>
}

#[allow(non_snake_case)]
pub struct Model{
    nodes: Vec<Node>,
    currently_produced: Vec<f64>,
    current_demand: Vec<f64>,
    root_order: Vec<usize>,
    leaf_order: Vec<usize>,
    current_limit: Vec<f64>,
    rng: Pcg64,
    stock_avail: Vec<Vec<StockAvailItem>>,
    demand_at_root: f64
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

pub fn test_demand()
{
    for demand in [0.2, 0.25, 0.3, 0.35, 0.4]{
        let mut model = Model::new_chain(50, 203984579, demand);
        let mut buf = create_buf_with_command_and_version(
            format!("test_demand{demand}.dat")
        );
        for i in 1..10000{
            model.update_demand();
            model.update_production();
            writeln!(
                buf,
                "{i} {}",
                model.current_demand[0]
            ).unwrap();
        }
    }
    
}

impl Model{
    fn new_chain(size: usize, seed: u64, demand_at_root: f64) -> Self{
        let first = Node{
            children: vec![1],
            parents: Vec::new()
        };
        let mut nodes = vec![first];
        let mut stock_avail = vec![vec![StockAvailItem{currently_avail: 0.0, stock: 0.0}]];
        for i in 1..size-1{
            let node = Node{
                children: vec![i + 1],
                parents: vec![IndexHelper{internal_idx: 0, node_idx: i - 1}]
            };
            stock_avail.push(vec![StockAvailItem{currently_avail: 0.0, stock: 0.0}]);
            nodes.push(node);
        }
        let last = Node{
            children: Vec::new(),
            parents: vec![IndexHelper{internal_idx: 0, node_idx: size - 2}]
        };
        nodes.push(last);
        stock_avail.push(Vec::new());
        let root_order = calc_root_order(&nodes);
        let leaf_order = calc_leaf_order(&nodes);

        let rng = Pcg64::seed_from_u64(seed);

        Self{
            rng,
            nodes,
            current_demand: vec![0.0; size],
            current_limit: vec![0.0; size],
            currently_produced: vec![0.0; size],
            root_order,
            leaf_order,
            stock_avail,
            demand_at_root
        }
    }


    fn update_demand(&mut self)
    {
        
        self.current_demand[0] += self.demand_at_root;
        set_const(&mut self.current_demand[1..], 0.0);

        for &idx in self.root_order.iter()
        {
            let demand = self.current_demand[idx];
            for &child in self.nodes[idx].children.iter(){
                self.current_demand[child] += demand;
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
        self.current_limit
            .iter_mut()
            .zip(uniform.sample_iter(&mut self.rng))
            .for_each(
                |(limit, rand_val)|
                {
                    *limit = rand_val
                }
            );
        for &idx in self.leaf_order.iter(){
            // firstly calculate actual production
            let production = &mut self.currently_produced[idx];
            *production = self.current_limit[idx];
            let iter = self.stock_avail[idx].iter();
            for StockAvailItem{currently_avail: avail, stock} in iter {
                *production = production.min(avail + stock);
            }
            // next calculate new stocks
            let iter = self.stock_avail[idx]
                .iter_mut();
            for item in iter {
                item.stock = 1.0_f64.min(item.currently_avail + item.stock - *production);
            }

            for parent in self.nodes[idx].parents.iter(){
                let passed_on = *production * self.current_demand[parent.node_idx] / self.current_demand[idx];
                self.stock_avail[parent.node_idx][parent.internal_idx].currently_avail = passed_on;
            }
        }
        self.current_demand[0] = 0.0_f64.max(self.current_demand[0] - self.currently_produced[0]);
    }
}
