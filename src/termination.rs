//! Shared first-answer termination for Cube-and-Conquer components.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// A clone-cheap stop signal shared by the cuber, its CDCL companion, and all
/// conquer workers.
#[derive(Clone, Debug, Default)]
pub struct TerminationSignal {
    requested: Arc<AtomicBool>,
}

impl TerminationSignal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Publish that one sound component has decided the instance.
    pub fn request(&self) {
        self.requested.store(true, Ordering::Release);
    }

    /// Observe a previously published terminal result.
    pub fn is_requested(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}
