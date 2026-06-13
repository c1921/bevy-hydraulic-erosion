use bevy::{
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use noise::{NoiseFn, Perlin};

use crate::config::{CELL_SIZE, GRID_SIZE, HEIGHT_AMP, HEIGHT_SCALE, LACUNARITY, NOISE_SCALE, OCTAVES, PERSISTENCE, DischargeOverlay, ErosionParams, PauseState};
use crate::erosion;
use crate::terrain_data::TerrainResource;

// ── Color stops ─────────────────────────────────────────────────

fn lerp_color(a: LinearRgba, b: LinearRgba, t: f32) -> LinearRgba {
    LinearRgba::new(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
        a.alpha + (b.alpha - a.alpha) * t,
    )
}

const COLOR_STOPS: [(f32, LinearRgba); 5] = [
    (0.00, LinearRgba::new(0.15, 0.35, 0.10, 1.0)),
    (0.30, LinearRgba::new(0.25, 0.55, 0.15, 1.0)),
    (0.55, LinearRgba::new(0.55, 0.50, 0.20, 1.0)),
    (0.80, LinearRgba::new(0.40, 0.28, 0.15, 1.0)),
    (1.00, LinearRgba::new(0.85, 0.83, 0.80, 1.0)),
];

fn height_to_color(h: f32, h_min: f32, h_max: f32) -> LinearRgba {
    let h_range = h_max - h_min;
    let t = if h_range > 0.0 {
        ((h - h_min) / h_range).clamp(0.0, 1.0)
    } else {
        0.5
    };
    if t <= COLOR_STOPS[0].0 {
        COLOR_STOPS[0].1
    } else if t >= COLOR_STOPS[4].0 {
        COLOR_STOPS[4].1
    } else {
        let mut lo = 0;
        while COLOR_STOPS[lo + 1].0 < t {
            lo += 1;
        }
        let s = (t - COLOR_STOPS[lo].0) / (COLOR_STOPS[lo + 1].0 - COLOR_STOPS[lo].0);
        lerp_color(COLOR_STOPS[lo].1, COLOR_STOPS[lo + 1].1, s)
    }
}

/// 将 discharge (erf 值) 映射为蓝→青→白热力图
fn discharge_to_color(d: f32) -> LinearRgba {
    // d is erf(0.4*discharge) ∈ [0, 1)
    let t = (d * 3.0).clamp(0.0, 1.0); // 放大以增强对比
    if t < 0.5 {
        // 蓝 → 青
        let s = t / 0.5;
        lerp_color(
            LinearRgba::new(0.0, 0.0, 0.3, 1.0),
            LinearRgba::new(0.0, 0.8, 0.8, 1.0),
            s,
        )
    } else {
        // 青 → 白
        let s = (t - 0.5) / 0.5;
        lerp_color(
            LinearRgba::new(0.0, 0.8, 0.8, 1.0),
            LinearRgba::new(1.0, 1.0, 1.0, 1.0),
            s,
        )
    }
}

// ── Terrain mesh handle resource ────────────────────────────────

#[derive(Resource)]
pub(crate) struct TerrainMeshHandle {
    pub handle: Handle<Mesh>,
    /// 上次高度范围（用于颜色条件更新）
    pub last_h_min: f32,
    pub last_h_max: f32,
}

// ── Startup system: create terrain data + build mesh ────────────

pub(crate) fn setup_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let n = GRID_SIZE + 1;
    let perlin = Perlin::new(42);
    let mut terrain = TerrainResource::new(n);

    // -- generate raw heights --
    for z in 0..n {
        for x in 0..n {
            let mut h = 0.0f64;
            let mut freq = NOISE_SCALE;
            let mut amp = 1.0;
            for _ in 0..OCTAVES {
                h += perlin.get([x as f64 * freq, z as f64 * freq]) * amp;
                freq *= LACUNARITY;
                amp *= PERSISTENCE;
            }
            terrain.set_height(x, z, h as f32 * HEIGHT_AMP);
        }
    }

    // -- normalize to [0, 1] --
    let h_min = terrain.cells_iter().map(|c| c.height).fold(f32::INFINITY, f32::min);
    let h_max = terrain.cells_iter().map(|c| c.height).fold(f32::NEG_INFINITY, f32::max);
    if h_max > h_min {
        let inv_range = 1.0 / (h_max - h_min);
        for cell in terrain.cells_mut_iter() {
            cell.height = (cell.height - h_min) * inv_range;
        }
    }

    // -- precompute indices (never change) --
    let indices = build_indices(n);

    // -- build full mesh --
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    write_full_mesh(&mut mesh, &terrain, &indices, &DischargeOverlay(false));
    let mesh_handle = meshes.add(mesh);

    // -- insert resources --
    commands.insert_resource(terrain);
    commands.insert_resource(TerrainMeshHandle {
        handle: mesh_handle.clone(),
        last_h_min: 0.0,
        last_h_max: 1.0,
    });

    // -- spawn mesh entity --
    commands.spawn((
        Mesh3d(mesh_handle),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::default(),
    ));
}

// ── Indices precomputation ──────────────────────────────────────

fn build_indices(n: usize) -> Vec<u32> {
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for z in 0..(n - 1) {
        for x in 0..(n - 1) {
            let a = (z * n + x) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    indices
}

// ── Full mesh write (startup only) ──────────────────────────────

fn write_full_mesh(mesh: &mut Mesh, terrain: &TerrainResource, indices: &[u32], overlay: &DischargeOverlay) {
    let n = terrain.size;
    let cell = CELL_SIZE;
    let height_scale = HEIGHT_AMP * HEIGHT_SCALE;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * n);
    let mut heights: Vec<f32> = Vec::with_capacity(n * n);

    for z in 0..n {
        for x in 0..n {
            let h = terrain.height_at(x, z);
            heights.push(h);
            positions.push([x as f32 * cell, h * height_scale, z as f32 * cell]);
        }
    }

    let h_min = heights.iter().cloned().fold(f32::INFINITY, f32::min);
    let h_max = heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // colors
    let colors = if overlay.0 {
        build_discharge_colors(terrain, n)
    } else {
        build_height_colors(&heights, h_min, h_max)
    };

    // normals
    let normals = build_normals(&positions, indices);

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices.to_vec()));
}

// ── Incremental update (per-frame) ──────────────────────────────

fn write_mesh_positions_normals(mesh: &mut Mesh, terrain: &TerrainResource) {
    let n = terrain.size;
    let cell = CELL_SIZE;
    let height_scale = HEIGHT_AMP * HEIGHT_SCALE;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * n);
    for z in 0..n {
        for x in 0..n {
            let h = terrain.height_at(x, z);
            positions.push([x as f32 * cell, h * height_scale, z as f32 * cell]);
        }
    }

    let normals = build_normals(&positions, &[]); // indices not needed if we pass them

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
}

fn write_mesh_colors(mesh: &mut Mesh, terrain: &TerrainResource, overlay: &DischargeOverlay) {
    let n = terrain.size;
    let colors = if overlay.0 {
        build_discharge_colors(terrain, n)
    } else {
        let mut heights: Vec<f32> = Vec::with_capacity(n * n);
        for z in 0..n {
            for x in 0..n {
                heights.push(terrain.height_at(x, z));
            }
        }
        let h_min = heights.iter().cloned().fold(f32::INFINITY, f32::min);
        let h_max = heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        build_height_colors(&heights, h_min, h_max)
    };
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
}

// ── Color builders ──────────────────────────────────────────────

fn build_height_colors(heights: &[f32], h_min: f32, h_max: f32) -> Vec<[f32; 4]> {
    heights
        .iter()
        .map(|&h| height_to_color(h, h_min, h_max).to_f32_array())
        .collect()
}

fn build_discharge_colors(terrain: &TerrainResource, n: usize) -> Vec<[f32; 4]> {
    let mut colors = Vec::with_capacity(n * n);
    for z in 0..n {
        for x in 0..n {
            let d = terrain.discharge_at(x, z);
            colors.push(discharge_to_color(d).to_f32_array());
        }
    }
    colors
}

// ── Normal builder ──────────────────────────────────────────────

fn build_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let n = positions.len();
    let mut normals = vec![Vec3::ZERO; n];

    // If indices provided, use them; otherwise iterate all triangles
    if !indices.is_empty() {
        for tri in indices.chunks(3) {
            accumulate_face_normal(&mut normals, positions, tri[0] as usize, tri[1] as usize, tri[2] as usize);
        }
    } else {
        // reconstruct indices pattern: grid of quads
        let side = (n as f64).sqrt() as usize;
        for z in 0..(side - 1) {
            for x in 0..(side - 1) {
                let a = z * side + x;
                let b = a + 1;
                let c = a + side;
                let d = c + 1;
                accumulate_face_normal(&mut normals, positions, a, c, b);
                accumulate_face_normal(&mut normals, positions, b, c, d);
            }
        }
    }

    normals
        .iter()
        .map(|n| n.normalize_or_zero().to_array())
        .collect()
}

fn accumulate_face_normal(normals: &mut [Vec3], positions: &[[f32; 3]], i0: usize, i1: usize, i2: usize) {
    let a = Vec3::from_array(positions[i0]);
    let b = Vec3::from_array(positions[i1]);
    let c = Vec3::from_array(positions[i2]);
    let face_normal = (b - a).cross(c - a);
    normals[i0] += face_normal;
    normals[i1] += face_normal;
    normals[i2] += face_normal;
}

// ── Update: erosion ─────────────────────────────────────────────

pub(crate) fn update_erosion(
    mut terrain: ResMut<TerrainResource>,
    params: Res<ErosionParams>,
    pause: Res<PauseState>,
) {
    if !pause.0 {
        erosion::erode(&mut terrain, &params);
    }
}

// ── Update: mesh refresh ────────────────────────────────────────

pub(crate) fn update_mesh(
    terrain: Res<TerrainResource>,
    mut handle: ResMut<TerrainMeshHandle>,
    overlay: Res<DischargeOverlay>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    let Some(mesh) = meshes.get_mut(&handle.handle) else {
        return;
    };

    // positions + normals: every frame
    write_mesh_positions_normals(mesh, &terrain);

    // colors: only when height range drifts > 1% or overlay toggled
    let n = terrain.size;
    let mut h_min = f32::INFINITY;
    let mut h_max = f32::NEG_INFINITY;
    for z in 0..n {
        for x in 0..n {
            let h = terrain.height_at(x, z);
            h_min = h_min.min(h);
            h_max = h_max.max(h);
        }
    }
    let range = h_max - h_min;
    let last_range = handle.last_h_max - handle.last_h_min;
    let drift = if last_range > 0.0 { (range - last_range).abs() / last_range } else { 1.0 };

    if drift > 0.01 || overlay.is_changed() {
        write_mesh_colors(mesh, &terrain, &overlay);
        handle.last_h_min = h_min;
        handle.last_h_max = h_max;
    }
}

// ── Input: pause toggle ─────────────────────────────────────────

pub(crate) fn toggle_pause(
    keys: Res<ButtonInput<KeyCode>>,
    mut pause: ResMut<PauseState>,
) {
    if keys.just_pressed(KeyCode::Space) {
        pause.0 = !pause.0;
    }
}

// ── Input: overlay toggle ───────────────────────────────────────

pub(crate) fn toggle_overlay(
    keys: Res<ButtonInput<KeyCode>>,
    mut overlay: ResMut<DischargeOverlay>,
) {
    if keys.just_pressed(KeyCode::KeyM) {
        overlay.0 = !overlay.0;
    }
}

// ── Input: parameter tuning ─────────────────────────────────────

pub(crate) fn tune_params(
    keys: Res<ButtonInput<KeyCode>>,
    mut params: ResMut<ErosionParams>,
) {
    if keys.just_pressed(KeyCode::KeyO) {
        params.cycles_per_frame = (params.cycles_per_frame as i32 + 256).max(64) as usize;
        info!("cycles_per_frame: {}", params.cycles_per_frame);
    }
    if keys.just_pressed(KeyCode::KeyP) {
        params.cycles_per_frame = (params.cycles_per_frame as i32 - 256).max(64) as usize;
        info!("cycles_per_frame: {}", params.cycles_per_frame);
    }
    if keys.just_pressed(KeyCode::KeyK) {
        params.entrainment = (params.entrainment + 5.0).min(50.0);
        info!("entrainment: {:.1}", params.entrainment);
    }
    if keys.just_pressed(KeyCode::KeyL) {
        params.entrainment = (params.entrainment - 5.0).max(1.0);
        info!("entrainment: {:.1}", params.entrainment);
    }
}
