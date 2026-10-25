//! The worker pool behind the parallel kernels.
//!
//! One process-wide pool of `total - 1` worker threads plus the calling
//! thread, created on first use. Idle workers spin on a ticket counter for
//! about two milliseconds, so back-to-back gemvs within a token see
//! sub-microsecond dispatch, and then park on a condition variable, so a
//! process sitting at a prompt costs no CPU. Dispatch is serialised by a
//! mutex: concurrent callers (two sessions on two threads) take turns rather
//! than corrupting each other's task. The caller runs thread 0's share
//! itself and then waits for every worker's completion counter.

use std::any::Any;
use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::Range;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release, SeqCst};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicUsize};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// The most threads the pool runs (caller included). It bounds the
/// per-thread arrays dispatchers keep on the stack, such as the argmax
/// reduction's.
pub(crate) const MAX_THREADS: usize = 64;

/// How long an idle worker spins before parking: long enough to bridge the
/// gaps between gemvs in a token (norms, attention, sampling), short enough
/// that a process waiting at a prompt is idle.
const SPIN_BEFORE_PARK: Duration = Duration::from_millis(2);

type Task<'a> = &'a (dyn Fn(usize, usize) + Sync + 'a);

/// A worker's completion counter, on a cache line of its own.
#[repr(align(128))]
struct Slot {
    done: AtomicU32,
}

/// What the dispatching thread shares with one generation of workers.
struct Shared {
    ticket: AtomicU32,
    stop: AtomicBool,
    sleepers: AtomicU32,
    sleep_mu: Mutex<()>,
    sleep_cv: Condvar,
    /// The dispatch in flight: a `*const Task` into the dispatcher's frame,
    /// published by the ticket increment.
    task: AtomicPtr<()>,
    task_n: AtomicUsize,
    /// The first panic a worker caught during the dispatch in flight.
    panic: Mutex<Option<Box<dyn Any + Send>>>,
    slots: Box<[Slot]>,
}

struct Workers {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

struct Pool {
    /// Held for a whole dispatch or reconfiguration.
    workers: Mutex<Workers>,
    /// Caller + workers.
    n_total: AtomicUsize,
}

thread_local! {
    /// Set on pool workers, and on a dispatching thread while it runs its
    /// own share: a dispatch from there runs inline instead of deadlocking.
    static IN_TASK: Cell<bool> = const { Cell::new(false) };
}

static POOL: OnceLock<Pool> = OnceLock::new();

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn pool() -> &'static Pool {
    POOL.get_or_init(|| Pool::new(default_threads()))
}

/// `NNCPU_THREADS` when set to a positive number, else min(8, CPUs).
fn default_threads() -> usize {
    let env = std::env::var("NNCPU_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok());
    let n = match env {
        Some(n) if n > 0 => n,
        _ => thread::available_parallelism().map_or(1, |n| n.get()).min(8),
    };
    n.min(MAX_THREADS)
}

impl Pool {
    fn new(total: usize) -> Pool {
        let workers = Workers::start(total);
        let n_total = AtomicUsize::new(workers.threads.len() + 1);
        Pool {
            workers: Mutex::new(workers),
            n_total,
        }
    }

    fn configure(&self, total: usize) {
        let mut w = lock(&self.workers);
        if total == self.n_total.load(Relaxed) {
            return;
        }
        w.stop();
        *w = Workers::start(total);
        self.n_total.store(w.threads.len() + 1, Relaxed);
    }
}

impl Workers {
    fn start(total: usize) -> Workers {
        let n = total.clamp(1, MAX_THREADS) - 1;
        let shared = Arc::new(Shared {
            ticket: AtomicU32::new(0),
            stop: AtomicBool::new(false),
            sleepers: AtomicU32::new(0),
            sleep_mu: Mutex::new(()),
            sleep_cv: Condvar::new(),
            task: AtomicPtr::new(std::ptr::null_mut()),
            task_n: AtomicUsize::new(1),
            panic: Mutex::new(None),
            slots: (0..n)
                .map(|_| Slot {
                    done: AtomicU32::new(0),
                })
                .collect(),
        });
        let mut threads = Vec::with_capacity(n);
        for i in 0..n {
            let s = Arc::clone(&shared);
            let spawned = thread::Builder::new()
                .name(format!("nncpu-worker-{}", i + 1))
                .spawn(move || worker_loop(&s, i));
            match spawned {
                Ok(t) => threads.push(t),
                // Out of threads: run with the ones we have.
                Err(_) => break,
            }
        }
        Workers { shared, threads }
    }

    fn stop(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        let s = &self.shared;
        s.stop.store(true, SeqCst);
        s.ticket.fetch_add(1, SeqCst);
        {
            let _g = lock(&s.sleep_mu);
            s.sleep_cv.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn worker_loop(s: &Shared, i: usize) {
    IN_TASK.set(true);
    let mut last = 0u32;
    loop {
        last = wait_for_ticket(s, last);
        if s.stop.load(Acquire) {
            return;
        }
        let task = s.task.load(Relaxed) as *const Task<'_>;
        let n = s.task_n.load(Relaxed);
        // SAFETY: the dispatcher stored a pointer to a live `Task` before
        // advancing the ticket (whose acquire load made it visible here), and
        // it waits for this worker's `done` increment below before its frame,
        // and with it the task, goes away.
        let f = unsafe { &*task };
        if let Err(p) = panic::catch_unwind(AssertUnwindSafe(|| f(i + 1, n))) {
            lock(&s.panic).get_or_insert(p);
        }
        s.slots[i].done.fetch_add(1, Release);
    }
}

/// Spin, then park, until the ticket moves past `last`; returns the new one.
fn wait_for_ticket(s: &Shared, last: u32) -> u32 {
    let mut spin_start = Instant::now();
    let mut spins = 0u32;
    loop {
        let t = s.ticket.load(Acquire);
        if t != last {
            return t;
        }
        spins = spins.wrapping_add(1);
        if !spins.is_multiple_of(1024) {
            std::hint::spin_loop();
            continue;
        }
        if spin_start.elapsed() < SPIN_BEFORE_PARK {
            continue;
        }
        let mut g = lock(&s.sleep_mu);
        s.sleepers.fetch_add(1, SeqCst);
        while s.ticket.load(SeqCst) == last {
            g = s.sleep_cv.wait(g).unwrap_or_else(PoisonError::into_inner);
        }
        s.sleepers.fetch_sub(1, Relaxed);
        drop(g);
        spin_start = Instant::now();
    }
}

/// Set the number of threads the kernels use, the calling thread included:
/// 0 restores the default (`NNCPU_THREADS` if set, else the CPU count up to
/// 8); larger requests are capped at 64. Waits for any work in flight.
///
/// # Panics
/// When called from inside work running on the pool.
pub fn set_num_threads(n: usize) {
    assert!(!IN_TASK.get(), "set_num_threads called from inside a pool task");
    let total = if n == 0 { default_threads() } else { n.min(MAX_THREADS) };
    let mut created = false;
    let p = POOL.get_or_init(|| {
        created = true;
        Pool::new(total)
    });
    if !created {
        p.configure(total);
    }
}

/// The number of threads the kernels use, the calling thread included.
pub fn num_threads() -> usize {
    pool().n_total.load(Relaxed)
}

/// Run `f(tid, n_threads)` for every `tid` in `[0, n_threads)` and return
/// when all are done. The caller runs tid 0. `n_threads` is authoritative
/// (it can differ from [`num_threads`] if the pool was resized meanwhile),
/// and is 1 for a dispatch made from inside a task, which runs inline.
///
/// A panic in any share is re-raised here once every share has finished.
pub(crate) fn dispatch<F: Fn(usize, usize) + Sync>(f: F) {
    if IN_TASK.get() {
        f(0, 1);
        return;
    }
    let w = lock(&pool().workers);
    let nw = w.threads.len();
    if nw == 0 {
        drop(w);
        f(0, 1);
        return;
    }
    let s = &*w.shared;
    let task: Task<'_> = &f;
    s.task.store(&task as *const Task<'_> as *mut (), Relaxed);
    s.task_n.store(nw + 1, Relaxed);
    // Snapshot the done counters, advance the ticket, wake any parked
    // workers, run the caller's share, then wait for the rest.
    let mut base = [0u32; MAX_THREADS];
    for (b, slot) in base.iter_mut().zip(&s.slots[..nw]) {
        *b = slot.done.load(Acquire);
    }
    s.ticket.fetch_add(1, SeqCst);
    // Pairs with the sleeper's increment-then-recheck: either we see it
    // parked and notify under the lock, or it sees the new ticket.
    if s.sleepers.load(SeqCst) > 0 {
        let _g = lock(&s.sleep_mu);
        s.sleep_cv.notify_all();
    }
    IN_TASK.set(true);
    let mine = panic::catch_unwind(AssertUnwindSafe(|| f(0, nw + 1)));
    IN_TASK.set(false);
    // Even if our own share panicked, the workers still use `task` and `f`:
    // wait for every one of them before either can go away.
    for (b, slot) in base.iter().zip(&s.slots[..nw]) {
        while slot.done.load(Acquire) == *b {
            std::hint::spin_loop();
        }
    }
    let theirs = lock(&s.panic).take();
    drop(w);
    if let Err(p) = mine {
        panic::resume_unwind(p);
    }
    if let Some(p) = theirs {
        panic::resume_unwind(p);
    }
}

/// Run `f(first, last)` over disjoint ranges covering `[0, n)`, split evenly
/// across the pool; returns when every range is done. For coarse, uniform
/// work such as load-time weight decoding.
pub(crate) fn parallel_for<F: Fn(u32, u32) + Sync>(n: u32, f: F) {
    if n == 0 {
        return;
    }
    if n == 1 || num_threads() == 1 {
        f(0, n);
        return;
    }
    dispatch(|tid, nth| {
        let (first, last) = split_range(n, tid, nth);
        if first < last {
            f(first, last);
        }
    });
}

/// `[first, last)` of `total` items for thread `tid` of `n`: an even split,
/// with the remainder spread over the lowest tids.
pub(crate) fn split_range(total: u32, tid: usize, n: usize) -> (u32, u32) {
    let (t, n) = (tid as u32, n as u32);
    let per = total / n;
    let rem = total % n;
    let first = t * per + t.min(rem);
    (first, first + per + u32::from(t < rem))
}

/// A raw pointer pool tasks may share, for handing disjoint mutable
/// sub-slices of one buffer to different threads. Closures must capture the
/// whole `SyncPtr` (use its methods, not `.0`) to be `Sync`.
pub(crate) struct SyncPtr<T>(pub *mut T);

impl<T> Clone for SyncPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SyncPtr<T> {}

// SAFETY: a SyncPtr is only an address; dereferencing it (slice_mut) is
// unsafe and its caller promises the threads' accesses are disjoint, which
// makes sharing it like sending `&mut [T]` pieces: sound for `T: Send`.
unsafe impl<T: Send> Send for SyncPtr<T> {}
unsafe impl<T: Send> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    pub fn new(s: &mut [T]) -> Self {
        SyncPtr(s.as_mut_ptr())
    }

    pub fn as_ptr(self) -> *mut T {
        self.0
    }

    /// `len` elements from `start`.
    ///
    /// # Safety
    /// `[start, start + len)` must lie inside the slice this pointer came
    /// from, that slice's borrow must outlive `'a`, and nothing else may
    /// access the range while the returned slice is alive.
    pub unsafe fn slice_mut<'a>(self, start: usize, len: usize) -> &'a mut [T] {
        // SAFETY: the caller's contract.
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

/// Rows `[first, end)` of `n_vec` output vectors spaced `stride` elements
/// apart in one buffer: what one thread of a row-split [`for_rows`] writes.
pub(crate) struct RowSpan<'a, T> {
    base: *mut T,
    stride: usize,
    n_vec: usize,
    rows: Range<usize>,
    _out: PhantomData<&'a mut [T]>,
}

impl<T> RowSpan<'_, T> {
    #[inline]
    pub fn first(&self) -> usize {
        self.rows.start
    }

    #[inline]
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))] // only the AVX2 kernels ask
    pub fn end(&self) -> usize {
        self.rows.end
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    pub fn n_vec(&self) -> usize {
        self.n_vec
    }

    /// Vector `v`'s rows `[first, end)`.
    #[inline]
    pub fn vec(&mut self, v: usize) -> &mut [T] {
        assert!(v < self.n_vec, "RowSpan::vec: vector {v} of {}", self.n_vec);
        // SAFETY: for_rows checked that every vector's rows lie inside the
        // buffer it borrows for 'a, and gave this span rows no other thread
        // touches; `&mut self` keeps one such slice alive at a time.
        unsafe { std::slice::from_raw_parts_mut(self.base.add(v * self.stride + self.rows.start), self.rows.len()) }
    }
}

/// Split `units` work units of `unit` rows each evenly across the pool (or
/// run them all on the calling thread when `threaded` is false) and call
/// `f` once per thread with its rows of each of the `n_vec` vectors in
/// `out`, vector `v` starting at element `v * stride`.
pub(crate) fn for_rows<T: Send, F: Fn(&mut RowSpan<'_, T>) + Sync>(
    threaded: bool,
    units: u32,
    unit: usize,
    out: &mut [T],
    stride: usize,
    n_vec: usize,
    f: F,
) {
    let rows = units as usize * unit;
    if rows == 0 || n_vec == 0 {
        return;
    }
    assert!(
        n_vec == 1 || stride >= rows,
        "for_rows: vectors of {rows} rows overlap at stride {stride}"
    );
    let need = (n_vec - 1).checked_mul(stride).and_then(|v| v.checked_add(rows));
    assert!(need.is_some_and(|need| need <= out.len()), "for_rows: output too short");
    let base = SyncPtr::new(out);
    let run = |u0: u32, u1: u32| {
        let rows = u0 as usize * unit..u1 as usize * unit;
        f(&mut RowSpan {
            base: base.as_ptr(),
            stride,
            n_vec,
            rows,
            _out: PhantomData,
        })
    };
    if threaded {
        dispatch(|tid, nth| {
            let (u0, u1) = split_range(units, tid, nth);
            if u0 < u1 {
                run(u0, u1);
            }
        });
    } else {
        run(0, units);
    }
}

/// Serialises tests that resize the pool (other tests may run alongside).
#[cfg(test)]
pub(crate) static TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) fn test_lock() -> MutexGuard<'static, ()> {
    lock(&TEST_LOCK)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn split_range_is_even_and_complete() {
        for total in [0u32, 1, 5, 64, 1000, 1001] {
            for n in 1..=9usize {
                let mut next = 0;
                for tid in 0..n {
                    let (a, b) = split_range(total, tid, n);
                    assert_eq!(a, next);
                    let len = b - a;
                    assert!(len == total / n as u32 || len == total / n as u32 + 1);
                    next = b;
                }
                assert_eq!(next, total);
            }
        }
    }

    #[test]
    fn parallel_for_covers_every_index_once() {
        for n in [0u32, 1, 2, 7, 64, 1000, 4099] {
            let hits: Vec<AtomicU32> = (0..n).map(|_| AtomicU32::new(0)).collect();
            parallel_for(n, |a, b| {
                for i in a..b {
                    hits[i as usize].fetch_add(1, Relaxed);
                }
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "n = {n}");
        }
    }

    #[test]
    fn dispatch_runs_every_tid_once() {
        let seen = AtomicU64::new(0);
        let count = AtomicUsize::new(0);
        let n_seen = AtomicUsize::new(0);
        dispatch(|tid, n| {
            seen.fetch_or(1 << tid, Relaxed);
            count.fetch_add(1, Relaxed);
            n_seen.store(n, Relaxed);
        });
        let n = n_seen.load(Relaxed);
        assert_eq!(count.load(Relaxed), n);
        assert_eq!(seen.load(Relaxed), if n == 64 { u64::MAX } else { (1 << n) - 1 });
    }

    #[test]
    fn nested_dispatch_runs_inline() {
        let inner = AtomicUsize::new(0);
        dispatch(|_, _| {
            dispatch(|tid, n| {
                assert_eq!((tid, n), (0, 1));
                inner.fetch_add(1, Relaxed);
            });
        });
        assert!(inner.load(Relaxed) >= 1);
    }

    #[test]
    fn set_num_threads_resizes() {
        let _g = test_lock();
        set_num_threads(3);
        assert_eq!(num_threads(), 3);
        let count = AtomicUsize::new(0);
        dispatch(|_, n| {
            assert_eq!(n, 3);
            count.fetch_add(1, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 3);
        set_num_threads(1);
        assert_eq!(num_threads(), 1);
        set_num_threads(1000);
        assert_eq!(num_threads(), MAX_THREADS);
        set_num_threads(0);
        assert_eq!(num_threads(), default_threads());
    }

    #[test]
    fn panics_propagate_and_the_pool_survives() {
        let _g = test_lock();
        set_num_threads(4);
        for culprit in [1usize, 0, 3] {
            let r = panic::catch_unwind(AssertUnwindSafe(|| {
                dispatch(|tid, _| {
                    if tid == culprit {
                        panic!("task {tid} failed");
                    }
                })
            }));
            let msg = r.expect_err("the panic must reach the caller");
            assert_eq!(
                msg.downcast_ref::<String>().map(String::as_str),
                Some(format!("task {culprit} failed").as_str())
            );
            let count = AtomicUsize::new(0);
            dispatch(|_, _| {
                count.fetch_add(1, Relaxed);
            });
            assert_eq!(count.load(Relaxed), 4);
        }
        set_num_threads(0);
    }

    #[test]
    fn workers_wake_after_parking() {
        let _g = test_lock();
        set_num_threads(4);
        std::thread::sleep(SPIN_BEFORE_PARK * 5);
        let count = AtomicUsize::new(0);
        dispatch(|_, _| {
            count.fetch_add(1, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 4);
        set_num_threads(0);
    }

    #[test]
    fn for_rows_writes_each_row_of_each_vector_once() {
        let (rows, stride, n_vec) = (1003usize, 1010usize, 3usize);
        let mut out = vec![0u32; (n_vec - 1) * stride + rows];
        for threaded in [false, true] {
            out.fill(0);
            for_rows(threaded, rows as u32, 1, &mut out, stride, n_vec, |span| {
                let first = span.first();
                for v in 0..span.n_vec() {
                    for (i, o) in span.vec(v).iter_mut().enumerate() {
                        *o += (v * stride + first + i) as u32 + 1;
                    }
                }
            });
            for v in 0..n_vec {
                for r in 0..rows {
                    assert_eq!(out[v * stride + r], (v * stride + r) as u32 + 1);
                }
            }
        }
    }
}
