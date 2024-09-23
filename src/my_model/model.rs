use std::{collections::VecDeque, num::*, ops::DerefMut, sync::Mutex};
use camino::{Utf8Path, Utf8PathBuf};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use sampling::{HistF64, Histogram};
//use std::process::Command;
use crate::misc::*;
use itertools::Itertools;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rand_pcg::Pcg64;
use indicatif::ProgressIterator;
use std::io::Write;

use crate::{
    complexer_firms::network_helper::write_my_digraph, create_buf, Cleaner, MyDistr
};

use super::{GnuplotFit, InitialStock, RandTreeCompareOpts};

pub fn regular_vs_random_tree(opts: RandTreeCompareOpts, out: Utf8PathBuf)
{
    let header = [
        "Demand",
        "Velocity"
    ];
    let name_stub = format!("{out}_{}", opts.initial_stock.to_str());
    let out = name_stub.as_str();

    let ratios = RatioIter::from_float(
        0.0, 
        1.0, 
        opts.demand_samples
    );
    let mut ratio_list = ratios.float_iter().collect_vec();
    ratio_list.reverse();

    let cleaner = Cleaner::new();
    let mut rng = Pcg64::seed_from_u64(opts.seed);
    let rngs = opts.s.iter()
        .map(
            |_|
            Pcg64::from_rng(&mut rng).unwrap()
        ).collect_vec();


    let time_steps_as_float = opts.time_steps.get() as f64;

    let N = Mutex::new(None);

    let regular_tree_results: Vec<_> = opts.s
        .par_iter()
        .zip(rngs)
        .map(
            |(&s, model_rng)|
            {
                let mut tree = Model::create_tree(
                    opts.regular_tree_z, 
                    opts.regular_tree_depth.get() - 1, 
                    model_rng, 
                    10.0, 
                    s
                );

                if let Ok(mut N_guard) = N.try_lock(){
                    if N_guard.is_none(){
                        let N = tree.nodes.len();
                        *N_guard = Some(N);
                        println!("N={N}");
                    }
                    drop(N_guard);
                }

                let out_name = format!("RegularTree{s}.dat");
                let gp_name = format!("RegularTree{s}.gp");
                let png_name = format!("RegularTree{s}.png");
                
                let mut buf = create_buf_with_command_and_version_and_header(
                    &out_name, 
                    header
                );
                for &demand in ratio_list.iter(){
                    tree.demand_at_root = demand;
                    // Currently warmup does nothing!
                    //tree.warmup(opts.warmup_samples);
                    tree.reset_delays();
                    tree.set_avail_stock_to_initial(opts.initial_stock);
                    for _ in 0..opts.time_steps.get() {
                        tree.update_demand();
                        tree.update_production();
                    }
                    let velocity = tree.current_demand[0] / time_steps_as_float;
                    writeln!(
                        buf,
                        "{demand} {velocity}"
                    ).unwrap();
                    if velocity <= super::execs::FIT_ZERO_THRESHOLD {
                        break;
                    }
                }
                drop(buf);
    
                let fit = GnuplotFit{
                    data_file: out_name,
                    gp_name,
                    png_name,
                    title: None,
                    y_range: None
                };
    
                let fit_result = fit.create_fit(&cleaner).expect("Error in fit!");
                fit_result.crit
            }
        ).collect();

    
    let base_hist = HistF64::new(-1.0, 1.0, opts.hist_bins.get())
        .unwrap();

    let hists = opts.s
        .iter()
        .map(|_| Mutex::new(base_hist.clone()))
        .collect_vec();



    let N = N.into_inner().unwrap().unwrap();
    
    let rng = Mutex::new(rng);

    (0..opts.tree_samples.get())
        .into_par_iter()
        .for_each(
            |sample_idx|
            {
                let dist = opts.rand_tree_distr.get_distr();
                let mut rng_guard = rng.lock().unwrap();
                let model_rng = Pcg64::from_rng(rng_guard.deref_mut()).unwrap();
                drop(rng_guard);
                let (mut model, _) = Model::create_rand_tree_with_N(
                    usize::MAX,
                    model_rng,
                    10.0,
                    0.0,
                    dist,
                    N
                );
        
                for ((s, hist), regular_tree_crit) in opts.s.iter().zip(hists.iter()).zip(regular_tree_results.iter()){
                    model.reset_delays();
                    model.max_stock = *s;
                    model.set_avail_stock_to_initial(opts.initial_stock);
        
                    let out_name = format!("Tree{sample_idx}_{s}.dat");
                    let gp_name = format!("Tree{sample_idx}_{s}.gp");
                    let png_name = format!("Tree{sample_idx}_{s}.png");
                    
                    let mut buf = create_buf_with_command_and_version_and_header(
                        &out_name, 
                        header
                    );
                    for &demand in ratio_list.iter(){
                        model.demand_at_root = demand;
                        // Currently warmup does nothing!
                        //model.warmup(opts.warmup_samples);
                        model.reset_delays();
                        for _ in 0..opts.time_steps.get() {
                            model.update_demand();
                            model.update_production();
                        }
                        let velocity = model.current_demand[0] / time_steps_as_float;
                        writeln!(
                            buf,
                            "{demand} {velocity}"
                        ).unwrap();
                        if velocity <= super::execs::FIT_ZERO_THRESHOLD {
                            break;
                        }
                    }
                    drop(buf);
        
                    let fit = GnuplotFit{
                        data_file: out_name,
                        gp_name,
                        png_name,
                        title: None,
                        y_range: None
                    };
        
                    let fit_result = fit.create_fit(&cleaner).expect("Error in fit!");
                    let diff = regular_tree_crit - fit_result.crit;
                    let mut guard = hist.lock().unwrap();
                    guard.increment(diff)
                        .expect("Hist error");
                    drop(guard);
                }
            }
        );

    cleaner.clean();

    let header = [
        "bin_mid",
        "normed",
        "hits"
    ];

    let iter = hists.into_iter()
        .zip(opts.s.iter())
        .zip(regular_tree_results.iter());

    for ((hist, s), regular_tree_crit) in iter
    {
        let name = format!("{out}_N{}_{s}.hist", N);
        let mut buf = create_buf_with_command_and_version_and_header(
            name, 
            header
        );
        writeln!(
            buf,
            "# regular tree crit: {regular_tree_crit}"
        ).unwrap();
        let hist = hist.into_inner().unwrap();
        let borders = hist.borders();
        let bin_size = borders[1] - borders[0];
        let all: usize = hist.hist().iter().sum();
        let norm_factor = (all as f64 * bin_size).recip();
        let mut sum = 0.0;
        for (bin, hits) in hist.bin_hits_iter(){
            let mid = (bin[0] + bin[1]) * 0.5;

            let normed = hits as f64 * norm_factor;
            sum += normed * bin_size;
            writeln!(
                buf,
                "{mid} {normed} {hits}"
            ).unwrap();
        }
        println!("SUM {sum} bin_size {bin_size}");
    }

    let reg_name = format!("{out}_regular_tree.dat");
    let header = [
        "s",
        "regular_tree_crit"
    ];
    let mut buf = create_buf_with_command_and_version_and_header(
        reg_name, 
        header
    );

    for (s, crit) in opts.s.iter().zip(regular_tree_results.iter())
    {
        writeln!(
            buf,
            "{s} {crit}"
        ).unwrap();
    }

}

#[allow(non_snake_case)]
#[derive(Clone)]
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
    appendix: usize,
    dot_name: &Utf8Path
){
    let model = Model::new_closed_multi_chain_from_rng(
        num_chains, 
        other_chain_len, 
        Pcg64::new(0, 0), 
        0.0, 
        0.0,
        appendix
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
            let next_level = infos.level + 1;
            let num_children = distr.rand_amount(&mut rng);
            if num_children > 0 {
                max_depth_reached = max_depth_reached.max(next_level);
            }
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


    /// Returns random tree and its depth
    pub fn create_rand_tree_with_N(
        max_depth: usize,
        mut rng: Pcg64,
        demand_at_root: f64,
        max_stock: f64,
        distr: Box<dyn MyDistr>,
        N: usize
    ) -> (Self, usize)
    {
        let mut nodes = vec![Node::default()];
        let mut queue = VecDeque::new();
        if max_depth > 0 {
            queue.push_back(
                HelpInfos{
                    level: 0,
                    id: 0
                }
            );
        }
        let mut stock_avail = vec![Vec::new()];
        let mut max_depth_reached = 0;
        while let Some(infos) = queue.pop_front(){
            let next_level = infos.level + 1;
            let mut num_children = distr.rand_amount(&mut rng);
            if num_children > 0 {
                max_depth_reached = max_depth_reached.max(next_level);
            }
            let stop_loop = nodes.len() + num_children >= N;
            if stop_loop {
                num_children = N - nodes.len();
            }
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
                    queue.push_back(
                        HelpInfos{
                            level: next_level,
                            id: this_id
                        }
                    );
                }
            }
            if stop_loop{
                break;
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
        max_stock: f64,
        appendix_nodes: usize
    ) -> Self{
        let total_node_count = 2 + other_chain_len.get() * num_chains.get() + appendix_nodes;
        let mut nodes = Vec::with_capacity(total_node_count);
        let first = Node::default();
        nodes.push(first);
        let connection_node = Node::default();
        nodes.push(connection_node);
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

            // now connect to connection node!
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

        let mut idx_parent = 1;
        for _ in 0..appendix_nodes{
            let mut node = Node::default();
            node.parents.push(
                IndexHelper{
                    internal_idx: 0,
                    node_idx: idx_parent
                }
            );
            let child_id = nodes.len();
            nodes[idx_parent].children.push(child_id);
            stock_avail[idx_parent].push(StockAvailItem::default());
            idx_parent = child_id;
            nodes.push(node);
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

    pub fn set_avail_stock(&mut self, stock: f64)
    {
        self.stock_avail.iter_mut()
            .for_each(
                |stocks|
                {
                    stocks.iter_mut()
                        .for_each(
                            |entry|
                            {
                                entry.stock = stock;
                            }
                        )
                }
            );
    }

    pub fn set_avail_stock_to_initial(&mut self, initial: InitialStock)
    {
        match initial{
            InitialStock::Empty => {
                self.set_avail_stock(0.0);
            },
            InitialStock::Full => {
                self.set_avail_stock(self.max_stock);
            },
            InitialStock::Iter => {
                unimplemented!()
            }
        }
    }

    pub fn warmup(&mut self, warmup_samples: u64)
    {
        let bar = crate::misc::indication_bar(warmup_samples);
        for _ in (0..warmup_samples).progress_with(bar){
            self.update_demand();
            self.update_production();
        }
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