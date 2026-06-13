//! GPU-accelerated hydraulic erosion via WGSL compute shader.
//!
//! Uses a standalone wgpu context for simplicity — the compute shader
//! runs on a separate device that shares the same physical GPU.
//!
//! Toggle: G key switches between CPU and GPU erosion mode.

use bevy::prelude::*;
use std::sync::Mutex;

use crate::config::{ErosionParams, MAP_SCALE};
use crate::terrain_data::TerrainResource;
use crate::ui::FrameCount;

// ── GPU-compatible ErosionParams ────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuErosionParams {
    grid_size: u32,
    map_scale: f32,
    max_age: u32,
    min_volume: f32,
    evaporation_rate: f32,
    deposition_rate: f32,
    entrainment: f32,
    gravity: f32,
    momentum_transfer: f32,
    cycles_per_frame: u32,
    frame_seed: u32,
    _pad: u32,
}

impl GpuErosionParams {
    fn from_cpu(params: &ErosionParams, grid_size: u32, frame_seed: u32) -> Self {
        Self {
            grid_size,
            map_scale: MAP_SCALE,
            max_age: params.max_age,
            min_volume: params.min_volume,
            evaporation_rate: params.evaporation_rate,
            deposition_rate: params.deposition_rate,
            entrainment: params.entrainment,
            gravity: params.gravity,
            momentum_transfer: params.momentum_transfer,
            cycles_per_frame: params.cycles_per_frame as u32,
            frame_seed,
            _pad: 0,
        }
    }
}

// ── GPU Context ─────────────────────────────────────────────────

struct GpuCtx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,

    heights: Option<wgpu::Buffer>,
    discharges: Option<wgpu::Buffer>,
    mom_x: Option<wgpu::Buffer>,
    mom_y: Option<wgpu::Buffer>,
    roots: Option<wgpu::Buffer>,
    track_d: Option<wgpu::Buffer>,
    track_x: Option<wgpu::Buffer>,
    track_y: Option<wgpu::Buffer>,
    params_buf: Option<wgpu::Buffer>,
    staging: Option<wgpu::Buffer>,

    buf_len: u64,
}

impl GpuCtx {
    fn new() -> Result<Self, String> {
        // Synchronous init using pollster
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());

        let adapter = match pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        )) {
            Ok(a) => a,
            Err(e) => return Err(format!("adapter error: {e:?}")),
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("erosion_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
        ))
        .map_err(|e| format!("{e:?}"))?;

        let shader_src = include_str!("../assets/shaders/erosion.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("erosion.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..9)
            .map(|binding| {
                let ty = match binding {
                    0 | 5..=7 => wgpu::BufferBindingType::Storage { read_only: false },
                    1..=4 => wgpu::BufferBindingType::Storage { read_only: true },
                    8 => wgpu::BufferBindingType::Uniform,
                    _ => unreachable!(),
                };
                wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty,
                        has_dynamic_offset: false,
                        min_binding_size: if binding == 8 {
                            std::num::NonZeroU64::new(std::mem::size_of::<GpuErosionParams>() as u64)
                        } else {
                            None
                        },
                    },
                    count: None,
                }
            })
            .collect();

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("erosion_bgl"),
            entries: &entries,
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("erosion_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("erosion_cp"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            bgl,
            heights: None,
            discharges: None,
            mom_x: None,
            mom_y: None,
            roots: None,
            track_d: None,
            track_x: None,
            track_y: None,
            params_buf: None,
            staging: None,
            buf_len: 0,
        })
    }

    fn ensure_bufs(&mut self, n: u64) {
        if self.buf_len == n {
            return;
        }
        let bytes = n * 4;
        let usage_s = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_sr = usage_s | wgpu::BufferUsages::COPY_SRC;

        self.heights = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("h"), size: bytes, usage: usage_sr, mapped_at_creation: false,
        }));
        self.discharges = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("d"), size: bytes, usage: usage_s, mapped_at_creation: false,
        }));
        self.mom_x = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mx"), size: bytes, usage: usage_s, mapped_at_creation: false,
        }));
        self.mom_y = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("my"), size: bytes, usage: usage_s, mapped_at_creation: false,
        }));
        self.roots = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("r"), size: bytes, usage: usage_s, mapped_at_creation: false,
        }));
        self.track_d = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("td"), size: bytes, usage: usage_sr, mapped_at_creation: false,
        }));
        self.track_x = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tx"), size: bytes, usage: usage_sr, mapped_at_creation: false,
        }));
        self.track_y = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ty"), size: bytes, usage: usage_sr, mapped_at_creation: false,
        }));
        self.staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stg"), size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }));
        self.params_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p"),
            size: std::mem::size_of::<GpuErosionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.buf_len = n;
    }

    fn run(
        &mut self,
        heights: &[f32],
        discharges: &[f32],
        mx: &[f32],
        my: &[f32],
        roots: &[f32],
        params: &GpuErosionParams,
        cycles: u32,
    ) -> Vec<f32> {
        let n = heights.len() as u64;
        if n == 0 {
            return vec![];
        }
        self.ensure_bufs(n);
        let nb = n as usize * 4;

        self.queue.write_buffer(self.heights.as_ref().unwrap(), 0, bytemuck::cast_slice(heights));
        self.queue.write_buffer(self.discharges.as_ref().unwrap(), 0, bytemuck::cast_slice(discharges));
        self.queue.write_buffer(self.mom_x.as_ref().unwrap(), 0, bytemuck::cast_slice(mx));
        self.queue.write_buffer(self.mom_y.as_ref().unwrap(), 0, bytemuck::cast_slice(my));
        self.queue.write_buffer(self.roots.as_ref().unwrap(), 0, bytemuck::cast_slice(roots));
        self.queue.write_buffer(self.params_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(params));

        let zeros = vec![0u8; nb];
        self.queue.write_buffer(self.track_d.as_ref().unwrap(), 0, &zeros);
        self.queue.write_buffer(self.track_x.as_ref().unwrap(), 0, &zeros);
        self.queue.write_buffer(self.track_y.as_ref().unwrap(), 0, &zeros);

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.heights.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.discharges.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.mom_x.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.mom_y.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.roots.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.track_d.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.track_x.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.track_y.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: self.params_buf.as_ref().unwrap().as_entire_binding() },
            ],
        });

        let wgs = ((cycles as u64 + 63) / 64) as u32;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        enc.copy_buffer_to_buffer(self.heights.as_ref().unwrap(), 0, self.staging.as_ref().unwrap(), 0, nb as u64);

        let idx = self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        });

        let slice = self.staging.as_ref().unwrap().slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if rx.recv().is_ok() {
            let mapped = slice.get_mapped_range();
            let out: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
            drop(mapped);
            self.staging.as_ref().unwrap().unmap();
            out
        } else {
            heights.to_vec()
        }
    }
}

// ── Bevy integration ────────────────────────────────────────────

#[derive(Resource)]
pub(crate) struct GpuErosionRes {
    ctx: Mutex<GpuCtx>,
    ready: bool,
}

#[derive(Resource, Default)]
pub(crate) struct GpuMode(pub bool);

pub(crate) struct GpuErosionPlugin;

impl Plugin for GpuErosionPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<GpuMode>();
    }
}

/// Initialize GPU context (call once, blocks briefly)
pub(crate) fn init_gpu(mut commands: Commands) {
    info!("Initializing GPU erosion context...");
    match GpuCtx::new() {
        Ok(ctx) => {
            info!("GPU erosion context ready");
            commands.insert_resource(GpuErosionRes {
                ctx: Mutex::new(ctx),
                ready: true,
            });
        }
        Err(e) => {
            error!("GPU init failed: {e}");
            commands.insert_resource(GpuErosionRes {
                ctx: Mutex::new(GpuCtx::new().unwrap()), // won't be used
                ready: false,
            });
        }
    }
}

/// Run GPU erosion (called in Update, blocks while GPU computes)
pub(crate) fn gpu_erode(
    mut terrain: ResMut<TerrainResource>,
    params: Res<ErosionParams>,
    frame: Res<FrameCount>,
    gpu: Res<GpuErosionRes>,
    mode: Res<GpuMode>,
) {
    if !gpu.ready || !mode.0 {
        return;
    }

    let n = terrain.size;
    let total = n * n;
    let mut h = Vec::with_capacity(total);
    let mut d = Vec::with_capacity(total);
    let mut mx = Vec::with_capacity(total);
    let mut my = Vec::with_capacity(total);
    let mut r = Vec::with_capacity(total);

    for z in 0..n {
        for x in 0..n {
            let c = terrain.cell(x, z);
            h.push(c.height);
            d.push(c.discharge);
            mx.push(c.momentum_x);
            my.push(c.momentum_y);
            r.push(c.root_density);
        }
    }

    let gp = GpuErosionParams::from_cpu(&params, n as u32, frame.0 as u32);
    let cycles = params.cycles_per_frame as u32;

    let result = {
        let mut ctx = gpu.ctx.lock().unwrap();
        ctx.run(&h, &d, &mx, &my, &r, &gp, cycles)
    };

    // Write back heights
    for z in 0..n {
        for x in 0..n {
            let idx = z * n + x;
            if idx < result.len() {
                terrain.cell_mut(x, z).height = result[idx];
            }
        }
    }
}
