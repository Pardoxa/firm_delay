use std::num::*;
use camino::Utf8Path;
use itertools::Itertools;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64;

use crate::{
    complexer_firms::network_helper::write_my_digraph, 
    create_buf, MyDistr
};

#[allow(non_snake_case)]
pub struct Model{
    pub nodes: Vec<Node>,
    pub currently_produced: Vec<f64>,
    pub current_demand: Vec<f64>,
    pub root_order: Vec<usize>,
    pub leaf_order: Vec<usize>,
    pub rng: Pcg64,
    pub stock_avail: Vec<Vec<StockAvailItem>>,
    pub demand_at_root: f64,
    pub max_stock: f64
}

pub fn write_tree_dot(
    child_count: NonZeroUsize, 
    depth: usize, 
    dot_name: &Utf8Path, 
    parent_direction: bool
)
{
    let model = Model::create_tree(
        child_count, 
        depth, 
        Pcg64::seed_from_u64(0), 
        0.0, 
        0.0
    );
    let file = create_buf(dot_name);
    write_my_digraph(file, &model.nodes, parent_direction);
}

pub fn write_rand_tree_dot(
    max_depth: usize, 
    dot_name: &Utf8Path, 
    parent_direction: bool,
    seed: u64,
    distr: Box<dyn MyDistr>
)
{
    let (model, _) = Model::create_rand_tree(
        max_depth,
        Pcg64::seed_from_u64(seed),
        0.0,
        0.0,
        distr
    );
    let file = create_buf(dot_name);
    write_my_digraph(file, &model.nodes, parent_direction);
}

pub fn write_closed_multi_chain(
    other_chain_len: NonZeroUsize,
    num_chains: NonZeroUsize,
    dot_name: &Utf8Path
){
    let model = Model::new_closed_multi_chain_from_rng(
        num_chains, 
        other_chain_len, 
        Pcg64::new(0, 0), 
        0.0, 
        0.0
    );
    let writer = create_buf(dot_name);
    write_my_digraph(
        writer, 
        &model.nodes, 
        false
    )
}

// Used internaly in the model impl
// for a stack 
struct HelpInfos{
    level: usize,
    id: usize
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

    /// Returns random tree and its depth
    pub fn create_rand_tree(
        max_depth: usize,
        mut rng: Pcg64,
        demand_at_root: f64,
        max_stock: f64,
        distr: Box<dyn MyDistr>
    ) -> (Self, usize)
    {
        let mut nodes = vec![Node::default()];
        let mut stack = Vec::new();
        if max_depth > 0 {
            stack.push(
                HelpInfos{
                    level: 0,
                    id: 0
                }
            );
        }
        let mut stock_avail = vec![Vec::new()];
        let mut max_depth_reached = 0;
        while let Some(infos) = stack.pop(){
            max_depth_reached = max_depth_reached.max(infos.level);
            let next_level = infos.level + 1;
            let num_children = distr.rand_amount(&mut rng);
            for i in 0..num_children{
                let node = Node{
                    parents: vec![IndexHelper{node_idx: infos.id, internal_idx: i}], 
                    children: Vec::new()
                };
                let this_id = nodes.len();
                nodes.push(node);
                nodes[infos.id].children.push(this_id);
                stock_avail[infos.id].push(StockAvailItem::default());
                stock_avail.push(Vec::new());
                if next_level < max_depth{
                    stack.push(
                        HelpInfos{
                            level: next_level,
                            id: this_id
                        }
                    );
                }
            }
        }

        let total_node_count = nodes.len();
        let root_order = calc_root_order(&nodes);
        let leaf_order = calc_leaf_order(&nodes);

        (
            Self{
                rng,
                nodes,
                current_demand: vec![0.0; total_node_count],
                currently_produced: vec![0.0; total_node_count],
                root_order,
                leaf_order,
                stock_avail,
                demand_at_root,
                max_stock
            },
            max_depth_reached
        )
        
    }

    pub fn create_tree(
        num_children: NonZeroUsize,
        depth: usize,
        rng: Pcg64,
        demand_at_root: f64,
        max_stock: f64
    ) -> Self
    {
        let mut nodes = vec![Node::default()];
        
        let mut stack = Vec::new();
        if depth > 0 {
            stack.push(
                HelpInfos{
                    level: 0,
                    id: 0
                }    
            );
        } 
        let mut stock_avail = vec![Vec::new()];
        while let Some(infos) = stack.pop() {
            let next_level = infos.level + 1;
            for i in 0..num_children.get(){
                let node = Node{
                    parents: vec![IndexHelper{node_idx: infos.id, internal_idx: i}], 
                    children: Vec::new()
                };
                let this_id = nodes.len();
                nodes.push(node);
                nodes[infos.id].children.push(this_id);
                stock_avail[infos.id].push(StockAvailItem::default());
                stock_avail.push(Vec::new());
                if next_level < depth{
                    stack.push(
                        HelpInfos{
                            level: next_level,
                            id: this_id
                        }
                    );
                }
            }
        }

        let total_node_count = nodes.len();
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
            demand_at_root,
            max_stock
        }

    }

    pub fn new_multi_chain_from_rng(
        num_chains: NonZeroUsize,
        other_chain_len: usize, 
        rng: Pcg64, 
        demand_at_root: f64,
        max_stock: f64
    ) -> Self{
        if other_chain_len == 0 {
            return Self{
                rng,
                nodes: vec![Node::default()],
                current_demand: vec![0.0],
                currently_produced: vec![0.0],
                root_order: vec![0],
                leaf_order: vec![0],
                stock_avail: vec![Vec::new()],
                demand_at_root,
                max_stock
            };
        }
        let total_node_count = 1 + other_chain_len * num_chains.get();
        let mut nodes = Vec::with_capacity(total_node_count);
        let first = Node::default();
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
            demand_at_root,
            max_stock
        }
    }

    pub fn new_closed_multi_chain_from_rng(
        num_chains: NonZeroUsize,
        other_chain_len: NonZeroUsize, 
        rng: Pcg64, 
        demand_at_root: f64,
        max_stock: f64
    ) -> Self{
        let total_node_count = 2 + other_chain_len.get() * num_chains.get();
        let mut nodes = Vec::with_capacity(total_node_count);
        let first = Node::default();
        nodes.push(first);
        let last = Node::default();
        nodes.push(last);
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
            for _ in 0..other_chain_len.get()-1{
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

            // now connect to last node!
            let last_nodes_idx = nodes.len() - 1;
            nodes[1].parents.push(
                IndexHelper { 
                    node_idx: last_nodes_idx, 
                    internal_idx: 0 
                }
            );
            nodes[last_nodes_idx].children.push(1);
            stock_avail[last_nodes_idx].push(StockAvailItem::default());
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
            demand_at_root,
            max_stock
        }
    }

    pub fn update_demand(&mut self)
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

    pub fn update_production(&mut self)
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
                item.stock = self.max_stock.min(item.currently_avail + item.stock - *production);
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

    pub fn update_production_quenched(&mut self, production_quenched_rand: &[f64])
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

        for (&idx, &rand) in self.leaf_order.iter().zip(production_quenched_rand){
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
                item.stock = self.max_stock.min(item.currently_avail + item.stock - *production);
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

#[derive(Debug, Clone, Copy)]
pub struct IndexHelper{
    pub node_idx: usize,
    // Idx of stock and available product
    pub internal_idx: usize
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StockAvailItem{
    pub stock: f64,
    pub currently_avail: f64,
    pub demand_passed_on: f64
}

#[derive(Debug, Clone, Default)]
pub struct Node{
    pub children: Vec<usize>,
    pub parents: Vec<IndexHelper>
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