use serde_json::Value;

use {
    serde::{Serialize, Deserialize},
    std::{
        num::*,
        io::{Write, BufWriter},
        fs::File
    },
    crate::misc::*
};

#[derive(Serialize, Deserialize, Clone)]
pub struct SimpleFirmDifferentKOpts{
    /// vector containing all k values
    pub k: Vec<NonZeroUsize>,

    /// Buffer of the firm
    pub buffer: f64,

    pub delta: f64,

    /// How many time steps to iterate
    pub iter_limit: NonZeroU64,
}

impl Default for SimpleFirmDifferentKOpts{
    fn default() -> Self {
        Self { 
            k: vec![NonZeroUsize::new(1).unwrap(), NonZeroUsize::new(10).unwrap(), NonZeroUsize::new(100).unwrap()], 
            buffer: 0.5, 
            delta: 0.49, 
            iter_limit: NonZeroU64::new(1000).unwrap()
        }
    }
}

impl SimpleFirmDifferentKOpts{
    pub fn get_name(&self) -> String
    {
        let version = crate::misc::VERSION;

        let ks = if self.k.is_empty(){
            panic!("Invalid! empty k");
        } else if self.k.len() <= 10 {
            let mut s = format!("{}", self.k[0]);
            for k in self.k.iter().skip(1)
            {
                s = format!("{s}_{k}");
            }
            s
        } else {
            let len = self.k.len();
            let start = self.k[0];
            let end = self.k.last().unwrap();
            format!("{start}_l{len}_{end}")
        };

        format!(
            "v{version}_b{}_d{}_k{ks}_it{}.dat",
            self.buffer,
            self.delta,
            self.iter_limit
        )
    }

    pub fn get_buf(&self) -> BufWriter<File>
    {
        let name = self.get_name();
        let file = File::create(name)
            .unwrap();
        let mut buf = BufWriter::new(file);
        writeln!(buf, "#Version {VERSION}").unwrap();
        write_commands(&mut buf)
            .expect("write error");
        let val: Value = serde_json::to_value(self.clone())
            .expect("serialization error");
        write_json(&mut buf, &val);
        buf
    }

    pub fn exec(self) 
    {
        super::different_k(&self)
    }
}