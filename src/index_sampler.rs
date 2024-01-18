use rand::Rng;
use core::panic;
use std::collections::*;
use rand_distr::{Uniform, Distribution};
use serde::{Serialize, Deserialize};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSampler
{
    pub len: usize,
    pub amount: usize,
    pub indices: Vec<u32>,
    pub cash: HashSet<u32>,
    pub how: SampleType
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub enum SampleType
{
    Reject,
    Inplace
}

impl IndexSampler
{

    pub fn measure_which<R>(len: usize, amount: usize, rng: &mut R) -> Self
    where R:  Rng + ?Sized
    {
        //println!("WHICH index sample method to use? len {len} amount {amount}");
        let mut inplace = Self::new_inplace(len, amount);

        let mut samples_inplace = Vec::new();
        for _ in 0..15 {
            let time = Instant::now();
            inplace.sample_inplace(rng);
            samples_inplace.push(time.elapsed());
        }
        
        let mut reject = Self::new_reject(len, amount);
        let mut samples_reject = Vec::new();
        for _ in 0..15 {
            let time = Instant::now();
            reject.sample_rejection(rng);
            samples_reject.push(time.elapsed());
        }
        
        samples_inplace.sort_unstable();
        samples_reject.sort_unstable();
        
        //for (i, r) in samples_inplace.iter().zip(samples_reject.iter())
        //{
        //    println!("INPLACE {}; Reject {}", humantime::format_duration(*i), humantime::format_duration(*r))
        //}

        // comparing median to see what's better
        if samples_inplace[7] < samples_reject[7]
        {
            //println!("Choosing inplace method!");
            inplace
        } else {
            //println!("Choosing reject method!");
            reject
        }
    
    }

    pub fn new_inplace(len: usize, amount: usize) -> Self{

        if amount > std::u32::MAX as usize {
            panic!("Abbort! amount cant be cast as u32")
        }

        if len > std::u32::MAX as usize {
            panic!("Abbort! len cant be cast as u32")
        }

        if amount >= len {
            panic!("Amount to large!")
        }

        let mut indices = Vec::with_capacity(len);
        indices.extend(
            0..len as u32
        );

        Self{
            how: SampleType::Inplace,
            len,
            amount,
            cash: HashSet::new(),
            indices
        }
        
    }

    #[allow(dead_code)]
    pub fn new_reject(len: usize, amount: usize) -> Self{

        if amount > std::u32::MAX as usize {
            panic!("Abbort! amount cant be cast as u32")
        }

        if len > std::u32::MAX as usize {
            panic!("Abbort! len cant be cast as u32")
        }

        if amount >= len {
            panic!("Amount to large!")
        }

        Self{
            how: SampleType::Reject,
            len,
            amount,
            cash: HashSet::new(),
            indices: Vec::with_capacity(amount)
        }
        
    }

    pub fn sample_indices<'a, R>(&'a mut self, rng: &mut R) -> &'a [u32]
    where
        R: Rng + ?Sized
    {
        match self.how{
            SampleType::Reject =>  self.sample_rejection(rng),
            SampleType::Inplace => self.sample_inplace(rng)
        }
    }

    pub fn sample_indices_without<'a, R>(&'a mut self, rng: &mut R, to_ignore: u32) -> &'a [u32]
    where
        R: Rng + ?Sized
    {
        match self.how{
            SampleType::Reject =>  self.sample_rejection_without(rng, to_ignore),
            SampleType::Inplace => self.sample_inplace_without(rng, to_ignore)
        }
    }

    /// Randomly sample exactly `amount` indices from `0..length`, using rejection
    /// sampling.
    ///
    /// Since `amount <<< length` there is a low chance of a random sample in
    /// `0..length` being a duplicate. We test for duplicates and resample where
    /// necessary. The algorithm is `O(amount)` time and memory.
    ///
    /// This function  is generic over X primarily so that results are value-stable
    /// over 32-bit and 64-bit platforms.
    fn sample_rejection<'a, R>(&'a mut self, rng: &mut R) -> &'a [u32]
    where
        R: Rng + ?Sized
    {
        
        let distr = Uniform::new(0, self.len as u32);
        self.cash.clear();
        self.indices.clear();
        
        self.indices
            .extend(
                (0..self.amount)
                    .map(
                        |_|
                        {
                            let mut pos = distr.sample(rng);
                            while !self.cash.insert(pos) {
                                pos = distr.sample(rng);
                            }
                            pos
                        }
                    )
            );

        self.indices.as_slice()
    }

    /// Randomly sample exactly `amount` indices from `0..length`, using rejection
    /// sampling.
    ///
    /// Since `amount <<< length` there is a low chance of a random sample in
    /// `0..length` being a duplicate. We test for duplicates and resample where
    /// necessary. The algorithm is `O(amount)` time and memory.
    ///
    /// This function  is generic over X primarily so that results are value-stable
    /// over 32-bit and 64-bit platforms.
    fn sample_rejection_without<'a, R>(&'a mut self, rng: &mut R, to_ignore: u32) -> &'a [u32]
    where
        R: Rng + ?Sized
    {
        
        let distr = Uniform::new(0, self.len as u32);
        self.cash.clear();
        self.indices.clear();

        self.indices
            .extend(
                (0..self.amount)
                    .map(
                        |_|
                        {
                            let mut pos = distr.sample(rng);
                            while pos == to_ignore || !self.cash.insert(pos) {
                                pos = distr.sample(rng);
                            }
                            pos
                        }
                    )
            );

        self.indices.as_slice()
    }

    /// Randomly sample exactly `amount` indices from `0..length`, using an inplace
    /// partial Fisher-Yates method.
    /// Sample an amount of indices using an inplace partial fisher yates method.
    ///
    /// This  randomizes only the first `amount`.
    /// It returns the corresponding slice
    ///
    /// This method is not appropriate for large `length` and potentially uses a lot
    /// of memory; because of this we only implement for `u32` index (which improves
    /// performance in all cases).
    ///
    /// shuffling is `O(amount)` time.
    pub fn sample_inplace_amount<'a, R>(&'a mut self, rng: &mut R, amount: usize) -> &'a [u32]
    where R: Rng + ?Sized {
        
        let len = self.len as u32;
        for i in 0..amount as u32 {
            let j: u32 = rng.gen_range(i..len);
            self.indices.swap(i as usize, j as usize);
        }
        
        &self.indices[0..amount]
    }

    /// Randomly sample exactly `amount` indices from `0..length`, using an inplace
    /// partial Fisher-Yates method.
    /// Sample an amount of indices using an inplace partial fisher yates method.
    ///
    /// This  randomizes only the first `amount`.
    /// It returns the corresponding slice
    ///
    /// This method is not appropriate for large `length` and potentially uses a lot
    /// of memory; because of this we only implement for `u32` index (which improves
    /// performance in all cases).
    ///
    /// shuffling is `O(amount)` time.
    pub fn sample_inplace<'a, R>(&'a mut self, rng: &mut R) -> &'a [u32]
    where R: Rng + ?Sized {
        self.sample_inplace_amount(rng, self.amount)
    }

    /// Randomly sample exactly `amount` indices from `0..length`, using an inplace
    /// partial Fisher-Yates method.
    /// Sample an amount of indices using an inplace partial fisher yates method.
    ///
    /// This  randomizes only the first `amount`.
    /// It returns the corresponding slice
    ///
    /// This method is not appropriate for large `length` and potentially uses a lot
    /// of memory; because of this we only implement for `u32` index (which improves
    /// performance in all cases).
    ///
    /// shuffling is `O(amount)` time.
    fn sample_inplace_without<'a, R>(&'a mut self, rng: &mut R, to_ignore: u32) -> &'a [u32]
    where R: Rng + ?Sized {
        
        let len = self.len as u32;
        for i in 0..self.amount as u32 {
            let j: u32 = rng.gen_range(i..len);
            self.indices.swap(i as usize, j as usize);
        }
        for i in 0..self.amount{
            if self.indices[i] == to_ignore {
                let j = rng.gen_range(self.amount..self.indices.len());
                self.indices.swap(i, j);
                break;
            }
        }
        
        &self.indices[0..self.amount]
    }
}