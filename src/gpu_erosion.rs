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
    /// Precomputed: 1.0 - evaporation_rate
    one_minus_evaporation_rate: f32,
    /// Precomputed: 1.0 / one_minus_evaporation_rate
    inv_one_minus_evaporation_rate: f32,
}

impl GpuErosionParams {
    fn from_cpu(params: &ErosionParams, grid_size: u32, frame_seed: u32) -> Self {
        let one_minus = 1.0 - params.evaporation_rate;
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
            one_minus_evaporation_rate: one_minus,
            inv_one_minus_evaporation_rate: 1.0 / one_minus,
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
    /// Single staging buffer for readback (4 × n floats: h, dt, mx, my)
    staging: Option<wgpu::Buffer>,

    /// Cached bind group — reused across frames; rebuilt only when buffers resize.
    bg: Option<wgpu::BindGroup>,

    /// Scratch CPU buffers reused across frames to avoid per-frame allocations.
    scratch_h: Vec<f32>,
    scratch_d: Vec<f32>,
    scratch_mx: Vec<f32>,
    scratch_my: Vec<f32>,
    scratch_r: Vec<f32>,
    zeros: Vec<u8>,

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
            bg: None,
            scratch_h: Vec::new(),
            scratch_d: Vec::new(),
            scratch_mx: Vec::new(),
            scratch_my: Vec::new(),
            scratch_r: Vec::new(),
            zeros: Vec::new(),
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
        let staging_usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ;
        // Single staging buffer: 4 arrays (h, dt, mx, my) laid out consecutively
        self.staging = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stg"), size: bytes * 4, usage: staging_usage, mapped_at_creation: false,
        }));
        self.params_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("p"),
            size: std::mem::size_of::<GpuErosionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.buf_len = n;

        // Build cached bind group (rebuilt only when buffers resize)
        self.bg = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        }));
    }

    fn run(
        &mut self,
        params: &GpuErosionParams,
        cycles: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = self.scratch_h.len() as u64;
        if n == 0 {
            return (vec![], vec![], vec![], vec![]);
        }
        self.ensure_bufs(n);
        let nb = n as usize * 4;

        self.queue.write_buffer(self.heights.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.scratch_h));
        self.queue.write_buffer(self.discharges.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.scratch_d));
        self.queue.write_buffer(self.mom_x.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.scratch_mx));
        self.queue.write_buffer(self.mom_y.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.scratch_my));
        self.queue.write_buffer(self.roots.as_ref().unwrap(), 0, bytemuck::cast_slice(&self.scratch_r));
        self.queue.write_buffer(self.params_buf.as_ref().unwrap(), 0, bytemuck::bytes_of(params));

        let zeros = {
            self.zeros.resize(nb, 0u8);
            &self.zeros
        };
        self.queue.write_buffer(self.track_d.as_ref().unwrap(), 0, zeros);
        self.queue.write_buffer(self.track_x.as_ref().unwrap(), 0, zeros);
        self.queue.write_buffer(self.track_y.as_ref().unwrap(), 0, zeros);

        let bg = self.bg.as_ref().expect("bind group not created");

        let wgs = ((cycles as u64 + 63) / 64) as u32;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("enc") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass"), timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        enc.copy_buffer_to_buffer(self.heights.as_ref().unwrap(), 0, self.staging.as_ref().unwrap(), 0, nb as u64);
        enc.copy_buffer_to_buffer(self.track_d.as_ref().unwrap(), 0, self.staging.as_ref().unwrap(), nb as u64, nb as u64);
        enc.copy_buffer_to_buffer(self.track_x.as_ref().unwrap(), 0, self.staging.as_ref().unwrap(), 2 * nb as u64, nb as u64);
        enc.copy_buffer_to_buffer(self.track_y.as_ref().unwrap(), 0, self.staging.as_ref().unwrap(), 3 * nb as u64, nb as u64);

        let idx = self.queue.submit(std::iter::once(enc.finish()));
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(idx),
            timeout: None,
        });

        // Single readback of merged staging buffer
        let staging = self.staging.as_ref().unwrap();
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });

        let (h_out, dt_out, mx_out, my_out) = if rx.recv().is_ok() {
            let mapped = slice.get_mapped_range();
            let all: &[f32] = bytemuck::cast_slice(&mapped);
            let nf = n as usize;
            let h_out   = all[0..nf].to_vec();
            let dt_out  = all[nf..2*nf].to_vec();
            let mx_out  = all[2*nf..3*nf].to_vec();
            let my_out  = all[3*nf..4*nf].to_vec();
            drop(mapped);
            staging.unmap();
            (h_out, dt_out, mx_out, my_out)
        } else {
            (vec![], vec![], vec![], vec![])
        };

        (h_out, dt_out, mx_out, my_out)
    }
}

// ── Bevy integration ────────────────────────────────────────────

/// CPU-side cascade for a single cell (used for GPU post-processing)
fn cascade_cell(terrain: &mut TerrainResource, x: usize, z: usize, max_diff: f32, settling: f32) {
    let cur_h = terrain.height_at(x, z);
    let neighbors: [(i32, i32, f32); 8] = [
        (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
        (0, -1, 1.0),                   (0, 1, 1.0),
        (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414),
    ];

    for &(dx, dz, dist) in &neighbors {
        let nx = x as i32 + dx;
        let nz = z as i32 + dz;
        if terrain.oob_i32(nx, nz) {
            continue;
        }
        let nh = terrain.height_at(nx as usize, nz as usize);
        let diff = cur_h - nh;
        if diff == 0.0 {
            continue;
        }
        let excess = if nh > 0.1 {
            (diff.abs() - dist * max_diff).max(0.0)
        } else {
            diff.abs()
        };
        if excess <= 0.0 {
            continue;
        }
        let transfer = settling * excess / 2.0;
        if diff > 0.0 {
            terrain.cell_mut(x, z).height -= transfer;
            terrain.cell_mut(nx as usize, nz as usize).height += transfer;
        }
    }
}

#[derive(Resource)]
pub(crate) struct GpuErosionRes {
    ctx: Mutex<GpuCtx>,
    ready: bool,
}

#[derive(Resource)]
pub(crate) struct GpuMode(pub bool);

impl Default for GpuMode {
    fn default() -> Self {
        Self(true)
    }
}

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

    let gp = GpuErosionParams::from_cpu(&params, n as u32, frame.0 as u32);
    let cycles = params.cycles_per_frame as u32;

    let (h_out, dt_out, mx_out, my_out) = {
        let mut ctx = gpu.ctx.lock().unwrap();

        // Reuse scratch buffers — clear + fill from terrain
        ctx.scratch_h.clear();
        ctx.scratch_d.clear();
        ctx.scratch_mx.clear();
        ctx.scratch_my.clear();
        ctx.scratch_r.clear();
        for z in 0..n {
            for x in 0..n {
                let c = terrain.cell(x, z);
                ctx.scratch_h.push(c.height);
                ctx.scratch_d.push(c.discharge);
                ctx.scratch_mx.push(c.momentum_x);
                ctx.scratch_my.push(c.momentum_y);
                ctx.scratch_r.push(c.root_density);
            }
        }

        ctx.run(&gp, cycles)
    };

    // Write back heights
    for z in 0..n {
        for x in 0..n {
            let idx = z * n + x;
            if idx < h_out.len() {
                terrain.cell_mut(x, z).height = h_out[idx];
            }
        }
    }

    // Merge tracking via EMA (same as CPU erosion::erode does after particle loop)
    if dt_out.len() == total && mx_out.len() == total && my_out.len() == total {
        for z in 0..n {
            for x in 0..n {
                let idx = z * n + x;
                let cell = terrain.cell_mut(x, z);
                cell.discharge_track = dt_out[idx];
                cell.momentum_x_track = mx_out[idx];
                cell.momentum_y_track = my_out[idx];
            }
        }
        terrain.merge_tracking(params.learning_rate);
    }

    // Cascade (CPU-side — simple and avoids GPU double-buffer complexity)
    // Use a lightweight pass: iterate cells, smooth steep neighbors
    for z in 0..n {
        for x in 0..n {
            cascade_cell(&mut terrain, x, z, params.max_diff, params.settling_rate);
        }
    }
}
