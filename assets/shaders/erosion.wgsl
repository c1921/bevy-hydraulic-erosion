// Hydraulic Erosion — GPU Compute Shader (WGSL)
// Translates CPU Drop::descend() to GPU-parallel particle processing.
//
// SoA buffer layout (all f32, except tracking maps as atomic<u32> for CAS):
//   binding 0: heights[N]           — read_write
//   binding 1: discharges[N]        — read
//   binding 2: momentum_x[N]        — read
//   binding 3: momentum_y[N]        — read
//   binding 4: root_density[N]      — read
//   binding 5: discharge_track[N]   — read_write (atomic<u32> via CAS)
//   binding 6: momentum_x_track[N]  — read_write (atomic<u32> via CAS)
//   binding 7: momentum_y_track[N]  — read_write (atomic<u32> via CAS)
//   binding 8: params               — uniform (ErosionParams)

struct ErosionParams {
    grid_size: u32,          // = GRID_SIZE + 1
    map_scale: f32,          // = MAP_SCALE (80.0)
    max_age: u32,
    min_volume: f32,
    evaporation_rate: f32,
    deposition_rate: f32,
    entrainment: f32,
    gravity: f32,
    momentum_transfer: f32,
    cycles_per_frame: u32,   // particles to spawn per dispatch
    frame_seed: u32,         // random seed (changes each frame)
    one_minus_evap_rate: f32,      // precomputed: 1.0 - evaporation_rate
    inv_one_minus_evap_rate: f32,  // precomputed: 1.0 / one_minus_evap_rate
}

@group(0) @binding(0) var<storage, read_write> heights: array<f32>;
@group(0) @binding(1) var<storage, read> discharges: array<f32>;
@group(0) @binding(2) var<storage, read> momentum_x: array<f32>;
@group(0) @binding(3) var<storage, read> momentum_y: array<f32>;
@group(0) @binding(4) var<storage, read> root_density: array<f32>;
@group(0) @binding(5) var<storage, read_write> discharge_track: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> momentum_x_track: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> momentum_y_track: array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params: ErosionParams;

const GRID_STEP: f32 = 1.41421356237; // sqrt(2)

// ── RNG (wang hash) ─────────────────────────────────────────────

fn wang_hash(seed: u32) -> u32 {
    var h = seed;
    h = h ^ 0x9e3779b9u;
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    h = h ^ (h >> 16u);
    return h;
}

// ── Atomic float helpers ────────────────────────────────────────

// ── Terrain access helpers ──────────────────────────────────────

fn height_at(x: u32, z: u32) -> f32 {
    return heights[z * params.grid_size + x];
}

fn discharge_erf(idx: u32) -> f32 {
    // erf(0.4 * discharge)
    let d = discharges[idx];
    return erf_approx(0.4 * d);
}

fn read_discharge(x: u32, z: u32) -> f32 {
    return discharge_erf(z * params.grid_size + x);
}

// ── erf approximation (Abramowitz & Stegun) ─────────────────────

fn erf_approx(x: f32) -> f32 {
    let sign = sign(x);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
    return sign * y;
}

// ── Normal from heights ─────────────────────────────────────────

fn compute_normal(x: u32, z: u32) -> vec3<f32> {
    let s = vec3(1.0, params.map_scale, 1.0);
    let hc = height_at(x, z);
    var n = vec3(0.0);

    // (+X, +Z)
    if x + 1u < params.grid_size && z + 1u < params.grid_size {
        let hR = height_at(x, z + 1u);
        let hB = height_at(x + 1u, z);
        let v1 = vec3(0.0, (hR - hc) * params.map_scale, 1.0);
        let v2 = vec3(1.0, (hB - hc) * params.map_scale, 0.0);
        n += cross(v1, v2);
    }
    // (-X, -Z)
    if x > 0u && z > 0u {
        let hL = height_at(x, z - 1u);
        let hT = height_at(x - 1u, z);
        let v1 = vec3(0.0, (hL - hc) * params.map_scale, -1.0);
        let v2 = vec3(-1.0, (hT - hc) * params.map_scale, 0.0);
        n += cross(v1, v2);
    }
    // (+X, -Z)
    if x + 1u < params.grid_size && z > 0u {
        let hB = height_at(x + 1u, z);
        let hL = height_at(x, z - 1u);
        let v1 = vec3(1.0, (hB - hc) * params.map_scale, 0.0);
        let v2 = vec3(0.0, (hL - hc) * params.map_scale, -1.0);
        n += cross(v1, v2);
    }
    // (-X, +Z)
    if x > 0u && z + 1u < params.grid_size {
        let hT = height_at(x - 1u, z);
        let hR = height_at(x, z + 1u);
        let v1 = vec3(-1.0, (hT - hc) * params.map_scale, 0.0);
        let v2 = vec3(0.0, (hR - hc) * params.map_scale, 1.0);
        n += cross(v1, v2);
    }

    let len = length(n);
    if len > 0.0 {
        n = n / len;
    }
    return n;
}

// ── Out-of-bounds check ─────────────────────────────────────────

fn oob(x: i32, z: i32) -> bool {
    return x < 0i || z < 0i || x >= i32(params.grid_size) || z >= i32(params.grid_size);
}

// ── Particle descend (one step) ─────────────────────────────────
// Returns true to continue, false = particle died.

struct Particle {
    pos_x: f32,
    pos_z: f32,
    speed_x: f32,
    speed_z: f32,
    volume: f32,
    sediment: f32,
    age: u32,
    // Cached per-step values — recomputed only when cell changes
    cache_cx: u32,
    cache_cz: u32,
    cache_normal: vec3<f32>,
    cache_disc_erf: f32,
}

fn descend_step(p: ptr<function, Particle>) -> bool {
    let ipos_x = i32(floor((*p).pos_x));
    let ipos_z = i32(floor((*p).pos_z));

    if oob(ipos_x, ipos_z) {
        (*p).volume = 0.0;
        return false;
    }

    let ix = u32(ipos_x);
    let iz = u32(ipos_z);
    let idx = iz * params.grid_size + ix;

    let cell_height = heights[idx];

    // ── Termination ──
    if (*p).age > params.max_age {
        heights[idx] += (*p).sediment;
        return false;
    }
    if (*p).volume < params.min_volume {
        heights[idx] += (*p).sediment;
        return false;
    }

    // ── Effective deposition ──
    let eff_depo = params.deposition_rate * max(0.0, 1.0 - root_density[idx]);

    // ── Normal + gravity (cached per cell) ──
    if ix != (*p).cache_cx || iz != (*p).cache_cz {
        (*p).cache_cx = ix;
        (*p).cache_cz = iz;
        (*p).cache_normal = compute_normal(ix, iz);
        (*p).cache_disc_erf = discharge_erf(idx);
    }
    let n = (*p).cache_normal;
    (*p).speed_x += n.x * params.gravity / (*p).volume;
    (*p).speed_z += n.z * params.gravity / (*p).volume;

    // ── Momentum transfer ──
    let flow_x = momentum_x[idx];
    let flow_y = momentum_y[idx];
    let flow_len = sqrt(flow_x * flow_x + flow_y * flow_y);
    let speed_len = sqrt((*p).speed_x * (*p).speed_x + (*p).speed_z * (*p).speed_z);
    if flow_len > 0.0 && speed_len > 0.0 {
        let flow_dir_x = flow_x / flow_len;
        let flow_dir_y = flow_y / flow_len;
        let speed_dir_x = (*p).speed_x / speed_len;
        let speed_dir_z = (*p).speed_z / speed_len;
        let dot = flow_dir_x * speed_dir_x + flow_dir_y * speed_dir_z;
        let factor = dot / ((*p).volume + discharges[idx]) * params.momentum_transfer;
        (*p).speed_x += factor * flow_x;
        (*p).speed_z += factor * flow_y;
    }

    // ── Fixed step ──
    if speed_len > 0.0 {
        let s = GRID_STEP / speed_len;
        (*p).speed_x = (*p).speed_x * s;
        (*p).speed_z = (*p).speed_z * s;
    }
    (*p).pos_x += (*p).speed_x;
    (*p).pos_z += (*p).speed_z;

    // ── Track discharge/momentum (atomic CAS inline) ──
    {
        var old_d = atomicLoad(&discharge_track[idx]);
        loop {
            let old_val = bitcast<f32>(old_d);
            let new_val = old_val + (*p).volume;
            let r = atomicCompareExchangeWeak(&discharge_track[idx], old_d, bitcast<u32>(new_val));
            if r.exchanged { break; }
            old_d = r.old_value;
        }
    }
    {
        var old_x = atomicLoad(&momentum_x_track[idx]);
        loop {
            let old_val = bitcast<f32>(old_x);
            let new_val = old_val + (*p).volume * (*p).speed_x;
            let r = atomicCompareExchangeWeak(&momentum_x_track[idx], old_x, bitcast<u32>(new_val));
            if r.exchanged { break; }
            old_x = r.old_value;
        }
    }
    {
        var old_y = atomicLoad(&momentum_y_track[idx]);
        loop {
            let old_val = bitcast<f32>(old_y);
            let new_val = old_val + (*p).volume * (*p).speed_z;
            let r = atomicCompareExchangeWeak(&momentum_y_track[idx], old_y, bitcast<u32>(new_val));
            if r.exchanged { break; }
            old_y = r.old_value;
        }
    }

    // ── New position height ──
    let new_ipos_x = i32(floor((*p).pos_x));
    let new_ipos_z = i32(floor((*p).pos_z));
    var new_height: f32;
    if oob(new_ipos_x, new_ipos_z) {
        new_height = cell_height - 0.002;
    } else {
        let nix = u32(new_ipos_x);
        let niz = u32(new_ipos_z);
        if nix >= params.grid_size || niz >= params.grid_size {
            new_height = cell_height - 0.002;
        } else {
            new_height = height_at(nix, niz);
        }
    }

    // ── Mass transfer (uses cached discharge_erf) ──
    let disc = (*p).cache_disc_erf;
    var c_eq = (1.0 + params.entrainment * disc) * (cell_height - new_height);
    if c_eq < 0.0 {
        c_eq = 0.0;
    }
    let c_diff = c_eq - (*p).sediment;

    (*p).sediment += eff_depo * c_diff;
    // Height write is non-atomic (benign race)
    heights[idx] -= eff_depo * c_diff;

    // ── Evaporation (uses precomputed constants) ──
    (*p).sediment *= params.inv_one_minus_evap_rate;
    (*p).volume *= params.one_minus_evap_rate;

    // ── Final OOB ──
    if oob(i32(floor((*p).pos_x)), i32(floor((*p).pos_z))) {
        (*p).volume = 0.0;
        return false;
    }

    (*p).age += 1u;
    return true;
}

// ── Main entry point ────────────────────────────────────────────

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let particle_idx = gid.x;
    if particle_idx >= params.cycles_per_frame {
        return;
    }

    // Random starting position
    let seed_x = wang_hash(particle_idx * 2u + params.frame_seed);
    let seed_z = wang_hash(particle_idx * 2u + 1u + params.frame_seed);
    let start_x = seed_x % params.grid_size;
    let start_z = seed_z % params.grid_size;

    // Check min height
    let h0 = height_at(start_x, start_z);
    if h0 < 0.1 {
        return;
    }

    // Run particle
    var p: Particle;
    p.pos_x = f32(start_x);
    p.pos_z = f32(start_z);
    p.speed_x = 0.0;
    p.speed_z = 0.0;
    p.volume = 1.0;
    p.sediment = 0.0;
    p.age = 0u;
    p.cache_cx = params.grid_size; // sentinel: force initial compute
    p.cache_cz = params.grid_size;
    p.cache_normal = vec3(0.0);
    p.cache_disc_erf = 0.0;

    loop {
        if !descend_step(&p) {
            break;
        }
    }
}
