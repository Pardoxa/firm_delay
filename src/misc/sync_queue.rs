use std::{collections::VecDeque, num::NonZeroUsize, sync::Mutex};



pub struct SyncQueue<T>{
    queue: Mutex<VecDeque<T>>
}

impl<T> SyncQueue<T>
{
    #[allow(dead_code)]
    pub fn new(queue: VecDeque<T>) -> Self
    {
        Self { queue: Mutex::new(queue) }
    }

    pub fn pop(&self) -> Option<T>
    {
        let mut lock = self.queue
            .lock()
            .unwrap();
        let item = lock.pop_front();
        drop(lock);
        item
    }

    pub fn push(&self, item: T)
    {
        let mut lock = self.queue
            .lock()
            .unwrap();
        lock.push_back(item);
        drop(lock);
    }

    pub fn map<F, K>(self, fun: F) -> SyncQueue<K>
    where F: FnMut (T) -> K
    {
        let  queue = self.queue.into_inner().unwrap();
        let queue = queue.into_iter()
            .map(fun)
            .collect();
        SyncQueue{queue: Mutex::new(queue)}
    }
}

impl SyncQueue<usize>
{
    pub fn create_work_queue(samples: usize, desired_package_amount: NonZeroUsize) -> Self
    {
        let mut remaining = samples;
        let r_step = (samples / desired_package_amount).max(1);

        let mut queue = VecDeque::new();
        loop{
            let amount = if remaining > r_step {
                remaining -= r_step;
                r_step
            } else {
                let tmp = remaining;
                remaining = 0;
                tmp
            };
            queue.push_back(amount);
            if remaining == 0 {
                break;
            }
        }
        assert_eq!(samples, queue.iter().sum::<usize>());
        Self{queue: Mutex::new(queue)}
    }
}