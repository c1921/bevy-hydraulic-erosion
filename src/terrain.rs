use bevy::{
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use noise::{NoiseFn, Perlin};

use crate::config::{CELL_SIZE, GRID_SIZE, HEIGHT_AMP, HEIGHT_SCALE, LACUNARITY, NOISE_SCALE, OCTAVES, PERSISTENCE, PauseState};
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

// ── Terrain mesh handle resource ────────────────────────────────

#[derive(Resource)]
pub(crate) struct TerrainMeshHandle(pub Handle<Mesh>);

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

    // -- build mesh from terrain data --
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    write_mesh_attributes(&mut mesh, &terrain);
    let mesh_handle = meshes.add(mesh);

    // -- insert resource + store handle --
    commands.insert_resource(terrain);
    commands.insert_resource(TerrainMeshHandle(mesh_handle.clone()));

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

// ── Mesh attribute writing (reusable) ───────────────────────────

/// 将 TerrainResource 的高度数据写入 Mesh 的顶点缓冲（positions + normals + colors + indices）
fn write_mesh_attributes(mesh: &mut Mesh, terrain: &TerrainResource) {
    let n = terrain.size;
    let cell = CELL_SIZE;
    let height_scale = HEIGHT_AMP * HEIGHT_SCALE;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * n);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * n);
    let mut normals = vec![Vec3::ZERO; n * n];
    let mut indices: Vec<u32> = Vec::new();

    // -- collect heights & positions --
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

    // -- vertex colors --
    for &h in &heights {
        let c = height_to_color(h, h_min, h_max);
        colors.push(c.to_f32_array());
    }

    // -- indices (two triangles per quad) --
    for z in 0..(n - 1) {
        for x in 0..(n - 1) {
            let a = (z * n + x) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    // -- smooth normals --
    for tri in indices.chunks(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;
        let a = Vec3::from_array(positions[i0]);
        let b = Vec3::from_array(positions[i1]);
        let c = Vec3::from_array(positions[i2]);
        let face_normal = (b - a).cross(c - a);
        normals[i0] += face_normal;
        normals[i1] += face_normal;
        normals[i2] += face_normal;
    }
    let normals_arr: Vec<[f32; 3]> = normals
        .iter()
        .map(|n| n.normalize_or_zero().to_array())
        .collect();

    // -- write into mesh (replace existing attributes) --
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals_arr);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.remove_indices();
    mesh.insert_indices(Indices::U32(indices));
}

// ── Update: erosion ─────────────────────────────────────────────

pub(crate) fn update_erosion(
    mut terrain: ResMut<TerrainResource>,
    params: Res<crate::config::ErosionParams>,
    pause: Res<PauseState>,
) {
    if !pause.0 {
        erosion::erode(&mut terrain, &params);
    }
}

// ── Update: mesh refresh ────────────────────────────────────────

pub(crate) fn update_mesh(
    terrain: Res<TerrainResource>,
    handle: Res<TerrainMeshHandle>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if let Some(mesh) = meshes.get_mut(&handle.0) {
        write_mesh_attributes(mesh, &terrain);
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
