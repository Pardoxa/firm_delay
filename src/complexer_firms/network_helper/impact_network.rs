use itertools::*;
use rand::{seq::SliceRandom, SeedableRng};
use rand_pcg::Pcg64Mcg;
pub struct ImpactNetworkHelper{
    //k: usize, // currently not needed, remove comment if that changes
    network: Vec<Vec<usize>>,
    // reverse network contains all links except self links
    reverse_network: Vec<Vec<usize>>
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

    pub fn into_inner_network(self) -> Vec<Vec<usize>>
    {
        self.network
    }

    fn new_empty(k: usize, n: usize) -> Self
    {
        Self{
            //k,
            network: vec![Vec::with_capacity(k); n],
            reverse_network: vec![Vec::new(); n]
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
}

// if this does not work:
// Maybe I should try two BFS's and add the levels and take the minimum
// also: check if the shuffling is really doing anything