use rand_distr::Exp;
use rand_pcg::Pcg64;



pub struct ModelSettings{
    const_demand: f64,
    lambda: f64
}

pub struct Model{
    in_network: Vec<Vec<usize>>,
    out_network: Vec<Vec<usize>>,
    in_order: Vec<usize>,
    // out order is reverse of in order
    out_order: Vec<usize>,
    stack: Vec<usize>,
    roots: Vec<usize>,
    ends: Vec<usize>,
    last_demand: Vec<f64>,
    current_demand: Vec<f64>,
    settings: ModelSettings,
    rng: Pcg64,
    avail: Vec<Option<f64>>
}

impl Model{
    fn iterate_once(&mut self)
    {
        self.calc_demand();
        self.do_work();
    }

    fn calc_demand(&mut self){
        std::mem::swap(&mut self.current_demand, &mut self.last_demand);
        self.current_demand
            .iter_mut()
            .for_each(|d| *d = 0.0);
        for &i in self.roots.iter(){
            self.current_demand[i] = self.last_demand[i] + self.settings.const_demand;
        }
        for &i in self.out_order.iter(){
            for &j in self.out_network[i].iter()
            {
                self.current_demand[j] += self.current_demand[i];
            }
        } 
    }

    fn do_work(&mut self)
    {
        self.avail
            .iter_mut()
            .for_each(|val| *val = None);
        let dist = Exp::new(self.settings.lambda).unwrap();
        for i in self.in_order.iter()
        {
            
        }
    }
}