#![allow(clippy::needless_return)]
#[macro_use]
extern crate shadow_rs;

pub mod comm_ops;
pub mod communicators;
pub mod cuda_utils;
pub mod datatypes;
pub mod events;
pub mod kernels;
pub mod resource_pool;
pub mod telemetry;

use crate::comm_ops::CommOpTrait;
use crate::telemetry::{SCHEDULED_THREAD_POOL, TELEMETRY};
use cpp::cpp;
use datatypes::{BaguaBucket, BaguaTensor};
use events::BaguaEventChannel;
use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;

cpp! {{
#include <nccl.h>
#include <stdio.h>
#include <iostream>
#include <bagua_utils.h>
}}

#[derive(Error, Debug)]
pub enum BaguaCoreError {
    #[error("invalid bucket")]
    BucketError(String),
    #[error("invalid tensor")]
    TensorError(String),
    #[error("memory pool error")]
    MemPoolError(#[from] sized_object_pool::SizedPoolError),
    #[error("communication backend error")]
    BackendError(String),
    #[error("communicator error")]
    CommunicatorError(String),
    #[error("telemetry error")]
    TelemetryError(String),
    #[error("serialization error")]
    SerializationSerdeJsonError(#[from] serde_json::Error),
    #[error("internal channel error")]
    InternalChannelError(String),
    #[error("http error")]
    HttpCommunicationError(#[from] ureq::Error),
}

#[derive(Debug)]
pub struct BaguaScheduledCommOp {
    pub bucket: Arc<BaguaBucket>,
    pub op: Arc<dyn CommOpTrait + Send + Sync>,
    pub event_channel: BaguaEventChannel,
}

#[derive(Debug)]
pub struct BaguaCommOpChannels {
    pub schedule_channel_sender: flume::Sender<BaguaScheduledCommOp>,
    pub schedule_channel_receiver: flume::Receiver<BaguaScheduledCommOp>,
    pub not_waited_events_sender: flume::Sender<BaguaEventChannel>,
    pub not_waited_events_receiver: flume::Receiver<BaguaEventChannel>,
    pub post_backward_channel_sender: flume::Sender<BaguaScheduledCommOp>,
    pub post_backward_channel_receiver: flume::Receiver<BaguaScheduledCommOp>,
    pub not_waited_post_backward_events_sender: flume::Sender<BaguaEventChannel>,
    pub not_waited_post_backward_events_receiver: flume::Receiver<BaguaEventChannel>,
}

impl BaguaCommOpChannels {
    pub fn new(cap: usize) -> Self {
        let (sender, receiver) = flume::bounded(cap);
        let (ev_sender, ev_receiver) = flume::unbounded();
        let (post_backward_channel_sender, post_backward_channel_receiver) = flume::bounded(cap);
        let (post_backward_ev_sender, post_backwar_ev_receiver) = flume::unbounded();

        Self {
            schedule_channel_sender: sender,
            schedule_channel_receiver: receiver,
            post_backward_channel_sender,
            post_backward_channel_receiver,
            not_waited_post_backward_events_sender: post_backward_ev_sender,
            not_waited_events_sender: ev_sender,
            not_waited_events_receiver: ev_receiver,
            not_waited_post_backward_events_receiver: post_backwar_ev_receiver,
        }
    }
}

pub fn show_version() {
    shadow!(build);
    eprintln!("project_name: {}", build::PROJECT_NAME);
    eprintln!("is_debug: {}", shadow_rs::is_debug());
    eprintln!("version: {}", build::version());
    eprintln!("tag: {}", build::TAG);
    eprintln!("commit_hash: {}", build::COMMIT_HASH);
    eprintln!("commit_date: {}", build::COMMIT_DATE);
    eprintln!("build_os: {}", build::BUILD_OS);
    eprintln!("rust_version: {}", build::RUST_VERSION);
    eprintln!("build_time: {}", build::BUILD_TIME);
    eprintln!("NCCL version: {}", {
        let mut version = 0i32;
        let version_ptr = &mut version;
        unsafe {
            cpp::cpp!([version_ptr as "int *"]
            { NCCLCHECK(ncclGetVersion(version_ptr)); });
        }
        version
    });
}

#[derive(Debug)]
pub struct BaguaCommBackend {
    ordered_buckets: VecDeque<Arc<BaguaBucket>>,
    /// <tensor_id, bagua_bucket>
    bucket_mapping: HashMap<u64, Arc<BaguaBucket>>,
    channels: Arc<BaguaCommOpChannels>,
    managed_ptrs: HashSet<u64>,
    comm_worker: std::thread::JoinHandle<()>,
}

impl BaguaCommBackend {
    pub fn schedule_comm(&self, bucket: Arc<BaguaBucket>) -> Result<(), BaguaCoreError> {
        let event_channel = BaguaEventChannel::default();
        self.channels.schedule_channel_sender.send(BaguaScheduledCommOp {
            op: {
                let guard = bucket.inner.lock();
                guard.comm_op.clone().expect("bucket must have communication operator set before scheduled for communication")
            },
            bucket,
            event_channel: event_channel.clone(),
        }).map_err(|e| BaguaCoreError::InternalChannelError(format!("{:?}", e)))?;
        Ok(self
            .channels
            .not_waited_events_sender
            .send(event_channel)
            .map_err(|e| BaguaCoreError::InternalChannelError(format!("{:?}", e)))?)
    }

    fn should_schedule(&self) -> Result<bool, BaguaCoreError> {
        return match self.ordered_buckets.front() {
            None => Err(BaguaCoreError::BackendError(
                "ordered buckets not yet set in comm backend".into(),
            )),
            Some(bucket) => {
                if bucket.ready_for_comm() {
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        };
    }
}

impl BaguaCommBackend {
    pub fn new(schedule_channel_cap: usize, device_id: usize) -> BaguaCommBackend {
        unsafe {
            cpp::cpp!([device_id as "size_t"]
            { CUDACHECK(cudaSetDevice(device_id)); });
        }

        let channels = Arc::new(BaguaCommOpChannels::new(schedule_channel_cap));
        let channels_clone = channels.clone();

        BaguaCommBackend {
            ordered_buckets: Default::default(),
            bucket_mapping: Default::default(),
            channels,
            managed_ptrs: Default::default(),
            comm_worker: std::thread::spawn(move || {
                unsafe {
                    cpp::cpp!([device_id as "size_t"]
                { CUDACHECK(cudaSetDevice(device_id)); });
                }
                let _span = tracing::span!(tracing::Level::TRACE, "execute_ops");
                let _guard = _span.enter();
                loop {
                    let comm_op = channels_clone
                        .schedule_channel_receiver
                        .recv()
                        .expect("cannot receive new comm op");
                    tracing::debug!(
                        "worker received scheduled communication operation {:?}",
                        comm_op
                    );
                    comm_op
                        .op
                        .execute_background_communication(comm_op.bucket.clone(), &channels_clone);
                    tracing::debug!("comm op executed: {:?}", comm_op);
                    comm_op.event_channel.finish();
                    tracing::debug!("comm op marked finished: {:?}", comm_op);
                }
            }),
        }
    }

    /// calling a second time will overwrite previous buckets
    pub fn register_ordered_buckets(
        &mut self,
        buckets: &[&BaguaBucket],
    ) -> Result<(), BaguaCoreError> {
        self.wait_pending_comm_ops()?;
        self.managed_ptrs.clear();
        self.bucket_mapping.clear();
        self.ordered_buckets.clear();
        for bucket in buckets {
            let bucket = Arc::new((*bucket).clone());
            self.ordered_buckets.push_back(bucket.clone());
            for tensor in &bucket.inner.lock().tensors {
                if self.bucket_mapping.contains_key(&tensor.id)
                    || self.managed_ptrs.contains(&tensor.inner.read().raw.ptr)
                {
                    return Err(BaguaCoreError::TensorError(format!(
                        "duplicated tensor detected, id {}, ptr {}",
                        &tensor.id,
                        &tensor.inner.read().raw.ptr
                    )));
                }
                self.bucket_mapping.insert(tensor.id, bucket.clone());
                self.managed_ptrs.insert(tensor.inner.read().raw.ptr);
            }
        }
        Ok(())
    }

    pub fn mark_communication_ready(
        &mut self,
        tensor: &BaguaTensor,
        ready_cuda_event_ptr: u64,
    ) -> Result<(), BaguaCoreError> {
        tensor.mark_comm_ready(ready_cuda_event_ptr);
        while self.should_schedule()? {
            let bucket = self.ordered_buckets.pop_front().unwrap();
            tracing::debug!("bucket {:?} ready for communication", bucket);
            bucket.reset_comm_ready();
            let bucket_clone = bucket.clone();
            self.ordered_buckets.push_back(bucket);
            self.schedule_comm(bucket_clone)?;
        }
        Ok(())
    }

    pub fn wait_pending_comm_ops(&self) -> Result<usize, BaguaCoreError> {
        let _span = tracing::span!(tracing::Level::TRACE, "wait_pending_comm_ops");
        let _guard = _span.enter();
        let mut num_ev = 0;
        loop {
            let ev = self.channels.not_waited_events_receiver.try_recv();
            match ev {
                Ok(x) => {
                    tracing::debug!("waiting for comm ops event {:?}", x);
                    x.wait();
                    tracing::debug!("comm ops event {:?} finished", x);
                    num_ev += 1;
                }
                Err(_) => return Ok(num_ev),
            }
        }
    }

    pub fn start_upload_telemetry(&self, skip: bool) -> Result<(), BaguaCoreError> {
        SCHEDULED_THREAD_POOL.execute(move || match TELEMETRY.as_ref() {
            None => {}
            Some(x) => {
                let mut guard = x.lock();
                match skip {
                    true => {
                        guard.clear();
                    }
                    false => {
                        match guard.push_payload_and_clear() {
                            Ok(_) => {}
                            Err(x) => {
                                tracing::error!("{:?}", x)
                            }
                        };
                    }
                }
            }
        });
        Ok(())
    }

    pub fn execute_post_backward_comm_ops(&self) -> Result<usize, BaguaCoreError> {
        let mut num_ev = 0;
        loop {
            let comm_op = self.channels.post_backward_channel_receiver.try_recv();
            match comm_op {
                Ok(comm_op) => {
                    tracing::debug!("received post step communication operation {:?}", comm_op);
                    comm_op
                        .op
                        .execute_background_communication(comm_op.bucket.clone(), &self.channels);
                    tracing::debug!("comm op executed: {:?}", comm_op);
                    comm_op.event_channel.finish();
                    tracing::debug!("comm op marked finished: {:?}", comm_op);
                    num_ev += 1;
                }
                Err(_) => return Ok(num_ev),
            }
        }
    }

    pub fn wait_pending_post_backward_comm_ops(&self) -> Result<usize, BaguaCoreError> {
        let mut num_ev = 0;
        loop {
            let ev = self
                .channels
                .not_waited_post_backward_events_receiver
                .try_recv();
            match ev {
                Ok(x) => {
                    tracing::debug!("waiting for comm ops event {:?}", x);
                    x.wait();
                    tracing::debug!("comm ops event {:?} finished", x);
                    num_ev += 1;
                }
                Err(_) => return Ok(num_ev),
            }
        }
    }
}