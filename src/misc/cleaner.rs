use std::{ops::DerefMut, sync::{atomic::AtomicUsize, Mutex}};

#[derive(Default)]
pub struct Cleaner{
    list: Mutex<Vec<String>>,
    list_size: AtomicUsize
}

impl Cleaner{
    pub fn new() -> Self{
        Self::default()
    }

    pub fn add(&self, s: String)
    {
        let mut lock = self.list.lock().unwrap();
        lock.push(s);
        self.list_size.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        drop(lock);
    }

    pub fn add_multi<I>(&self, iter: I)
    where I: IntoIterator<Item = String>
    {
        let mut lock = self.list.lock().unwrap();
        lock.extend(iter);
        let len = lock.len();
        self.list_size.store(len, std::sync::atomic::Ordering::SeqCst);
        drop(lock);
    }

    pub fn clean(self){
        let list = self.list
            .into_inner()
            .unwrap();
        remove_files(list);
    }

    /// Cleans files if more than threshold files are tracked
    pub fn clean_if_more_than(&self, limit: usize)
    {
        println!("len: {}", self.list_size.load(std::sync::atomic::Ordering::SeqCst));
        if self.list_size.load(std::sync::atomic::Ordering::SeqCst) > limit {
            if let Ok(mut guard) = self.list.try_lock()
            {
                let mut swap = Vec::with_capacity(limit + 30);
                std::mem::swap( guard.deref_mut(), &mut swap);
                self.list_size.store(0, std::sync::atomic::Ordering::SeqCst);
                drop(guard);
                println!("removing {} files", swap.len());
                remove_files(swap);
            }
        }
    }
}

fn remove_files(list: Vec<String>)
{
    for s in list{
        let _ = std::fs::remove_file(&s);
    }
}