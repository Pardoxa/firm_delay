use itertools::*;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use sampling::traits::MarkovChain;
pub struct ImpactNetworkHelper{
    //k: usize, // currently not needed, remove comment if that changes
    network: Vec<Vec<usize>>,
    // reverse network contains all links except self links
    reverse_network: Vec<Vec<usize>>,
    rng: Pcg64Mcg
}

pub struct Step{
    i: usize,
    old_j: usize,
    new_j: usize
}

impl MarkovChain<Step, ()> for ImpactNetworkHelper{
    fn m_step(&mut self) -> Step {
        let which_i = self.rng.gen_range(0..self.network.len());
        let which_a = self.rng.gen_range(0..self.network[which_i].len());

        let removed = self.network[which_i].swap_remove(which_a);
        let pos = self.reverse_network[removed]
            .iter().position(|&val| val == which_i)
            .unwrap();
        self.reverse_network[removed].swap_remove(pos);
        loop{
            let new = self.rng.gen_range(0..self.network.len());
            if !self.network[which_i].contains(&new)
            {
                self.network[which_i].push(new);
                self.reverse_network[new].push(which_i);
                return Step{
                    i: which_i,
                    old_j: removed,
                    new_j: new
                };
            }
        }
    }

    fn undo_step(&mut self, step: &Step) {
        self.undo_step_quiet(step)
    }

    fn undo_step_quiet(&mut self, step: &Step) {
        let pos = self.network[step.i]
            .iter().position(|&val| val == step.new_j)
            .unwrap();
        let _new_j = self.network[step.i].swap_remove(pos);
        let pos = self.reverse_network[step.new_j]
            .iter().position(|&val| val == step.i)
            .unwrap();
        self.reverse_network[step.new_j].swap_remove(pos);
        self.reverse_network[step.old_j].push(step.i);
        self.network[step.i].push(step.old_j);

    }
}

#[allow(clippy::upper_case_acronyms)]
struct BFS<'a>{
    visited: Vec<bool>,
    reverse_network: &'a [Vec<usize>],
    current_queue: Vec<usize>,
    next_queue: Vec<usize>,
    level: u32
}

impl<'a> Iterator for BFS<'a>
{
    type Item = (u32, usize);
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.current_queue.pop()?;
        self.reverse_network[next]
            .iter()
            .for_each(
                |&idx|
                {
                    if !self.visited[idx]{
                        self.visited[idx] = true;
                        self.next_queue.push(idx);
                    }
                }
            );
        if self.current_queue.is_empty(){
            let level = self.level;
            self.level += 1;
            std::mem::swap(&mut self.next_queue, &mut self.current_queue);
            Some((level, next))
        } else {
            Some((self.level, next))
        }
    }
}

impl<'a> BFS<'a>
{
    pub fn new(reverse_network: &'a[Vec<usize>], start: usize) -> Self
    {
        let n = reverse_network.len();
        let mut visited = vec![false; n];
        visited[start] = true;
        let current_queue = vec![start];
        Self{
            visited,
            current_queue,
            next_queue: Vec::new(),
            reverse_network,
            level: 0
        }
    }

    pub fn recycle(self, network: &[Vec<usize>], start: usize) -> BFS
    {
        let mut visited = self.visited;
        visited.iter_mut()
            .for_each(|item| *item = false);
        let mut c_queue = self.current_queue;
        c_queue.clear();
        c_queue.push(start);
        let mut next_queue = self.next_queue;
        next_queue.clear();
        BFS { 
            visited, 
            reverse_network: network, 
            current_queue: c_queue, 
            next_queue, 
            level: 0 
        }
    }

    pub fn recycle_self(&mut self, start: usize)
    {
        self.visited.iter_mut()
            .for_each(|item| *item = false);
        self.current_queue.clear();
        self.current_queue.push(start);
        self.next_queue.clear();
        self.level = 0;
    }
}

impl ImpactNetworkHelper{
    pub fn new(k: usize, n: usize) -> Self{
        let mut this = Self::new_empty(k, n);
        this.self_links();
        this.ring_links();
        let mut rng = Pcg64Mcg::seed_from_u64(814932);
        let mut order = (0..this.reverse_network.len()).collect_vec();
        for _ in 2..k {
            order.shuffle(&mut rng);
            this.neighbor_adding_sweep(&order);
        }
        this
    }

    pub fn new_both_dirs(k: usize, n: usize) -> Self{
        let mut this = Self::new_empty(k, n);
        //this.self_links();
        //this.ring_links();
        for _ in 0..k {
            this.neighbor_adding_sweep_both_directions();
        }
        this
    }

    pub fn rebuild_both_dirs(&mut self, iterations: usize)
    {
        let len = self.network[0].len();
        for _ in 0..len * iterations{
            self.rebuild_sweep_both_directions();
        }
    }

    pub fn markov_greed(&mut self, steps: usize, sz: usize)
    {
        let mut current_energy = self.pseudo_energy();
        let mut step_vec = Vec::new();
        for _ in 0..steps{
            self.m_steps(sz, &mut step_vec);
            let new_energy = self.pseudo_energy();
            if new_energy > current_energy{
                self.undo_steps_quiet(&step_vec);
            } else {
                current_energy = new_energy;
            }
        }
    }

    fn pseudo_energy(&self) -> usize
    {
        let mut bfs_rn = BFS::new(&self.reverse_network, 0);
        let mut counter = 1;
        let len = self.network.len();
        let mut sum = 0;
        let mut all = vec![u32::MAX; len];
        loop{
            for (level, idx) in &mut bfs_rn
            {
                all[idx] = level;
            }
            sum += all.iter().sum::<u32>() as usize;
            if counter == len{
                break;
            }
            bfs_rn.recycle_self(counter);
            all.iter_mut().for_each(|i| *i = u32::MAX);
            counter += 1;
        }
        sum
    }

    pub fn into_inner_network(self) -> Vec<Vec<usize>>
    {
        self.network
    }

    fn new_empty(k: usize, n: usize) -> Self
    {
        Self{
            //k,
            network: vec![Vec::with_capacity(k); n],
            reverse_network: vec![Vec::new(); n],
            rng: Pcg64Mcg::seed_from_u64(823947)
        }
        
    }

    fn self_links(&mut self)
    {
        self.network
            .iter_mut()
            .enumerate()
            .for_each(
                |(idx, adj)|
                {
                    adj.push(idx);
                }
            );
    }

    fn ring_links(&mut self)
    {
        let iter = (0..self.network.len())
            .circular_tuple_windows();
        for (i, j) in iter {
            self.network[i].push(j);
            self.reverse_network[j].push(i);
        }
    }

    fn neighbor_adding_sweep(&mut self, order: &[usize])
    {
       
        // maybe I want to change the order in which this is iterated…
        for &i in order
        {
            let bfs = BFS::new(&self.reverse_network, i);
            if let Some((_, idx)) = bfs.last()
            {
                self.network[i].push(idx);
                self.reverse_network[idx].push(i);
            }

        }
    }

    fn neighbor_adding_sweep_both_directions(&mut self)
    {
        let mut front = true;
        let iter = (0..self.network.len())
            .batching(
                |it|
                {
                    let next = if front{
                        it.next()
                    } else {
                        it.next_back()
                    };
                    front = !front;
                    next
                }
            );
        for i in iter
        {
            let mut count = vec![u32::MAX; self.reverse_network.len()];
            let mut bfs = BFS::new(&self.reverse_network, i);
            for (level, idx) in &mut bfs {
                count[idx] = count[idx].min(level);
            }
            let bfs = bfs.recycle(&self.network, i);
            for (level, idx) in bfs {
                count[idx] = count[idx].min(level);
            }

            let mut maximum = count[0];
            let mut idx = 0;
            for (&level, index) in count[1..].iter().zip(1..)
            {
                if level >= maximum{
                    maximum = level;
                    idx = index;
                }
            }
            self.network[i].push(idx);
            self.reverse_network[idx].push(i);
        }
    }

    fn rebuild_sweep_both_directions(&mut self)
    {
        for i in 0..self.network.len()
        {
            // first delete links
            let removed = self.network[i].remove(0);
            let pos = self.reverse_network[removed]
                .iter().position(|&val| val == i)
                .unwrap();
            self.reverse_network[removed].swap_remove(pos);
            // then add new ones
            let mut count = vec![u32::MAX; self.reverse_network.len()];
            let mut bfs = BFS::new(&self.reverse_network, i);
            for (level, idx) in &mut bfs {
                count[idx] = count[idx].min(level);
            }
            let bfs = bfs.recycle(&self.network, i);
            for (level, idx) in bfs {
                count[idx] = count[idx].min(level);
            }

            let mut maximum = count[0];
            let mut idx = 0;
            for (&level, index) in count[1..].iter().zip(1..)
            {
                if level >= maximum{
                    maximum = level;
                    idx = index;
                }
            }
            self.network[i].push(idx);
            self.reverse_network[idx].push(i);
            
        }
    }
}

// if this does not work:
// Maybe I should try two BFS's and add the levels and take the minimum
// also: check if the shuffling is really doing anything