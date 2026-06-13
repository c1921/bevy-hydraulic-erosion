use bevy::{
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};
use noise::{NoiseFn, Perlin};

// ── Terrain config ──────────────────────────────────────────────

pub(crate) const TERRAIN_SIZE: usize = 512;
pub(crate) const CELL_SIZE: f32 = 2.0;
const NOISE_SCALE: f64 = 0.015;
const HEIGHT_AMP: f32 = 30.0;
const OCTAVES: usize = 6;
const LACUNARITY: f64 = 2.0;
const PERSISTENCE: f64 = 0.5;

// ── Startup system ──────────────────────────────────────────────

pub(crate) fn spawn_terrain(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let mesh = generate_terrain_mesh();
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::default(),
    ));
}

// ── Mesh generation ─────────────────────────────────────────────

fn lerp_color(a: LinearRgba, b: LinearRgba, t: f32) -> LinearRgba {
    LinearRgba::new(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
        a.alpha + (b.alpha - a.alpha) * t,
    )
}

fn generate_terrain_mesh() -> Mesh {
    let perlin = Perlin::new(42);
    let n = TERRAIN_SIZE + 1;
    let cell = CELL_SIZE;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(n * n);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(n * n);
    let mut normals = vec![Vec3::ZERO; n * n];
    let mut indices: Vec<u32> = Vec::new();

    // -- vertices (two-pass: first collect heights, then normalize & colour) --
    let mut heights: Vec<f32> = Vec::with_capacity(n * n);
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
            let h = h as f32 * HEIGHT_AMP;
            heights.push(h);
            positions.push([x as f32 * cell, h, z as f32 * cell]);
        }
    }

    let h_min = heights.iter().cloned().fold(f32::INFINITY, f32::min);
    let h_max = heights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let h_range = h_max - h_min;

    let stops: [(f32, LinearRgba); 5] = [
        (0.00, LinearRgba::new(0.15, 0.35, 0.10, 1.0)),
        (0.30, LinearRgba::new(0.25, 0.55, 0.15, 1.0)),
        (0.55, LinearRgba::new(0.55, 0.50, 0.20, 1.0)),
        (0.80, LinearRgba::new(0.40, 0.28, 0.15, 1.0)),
        (1.00, LinearRgba::new(0.85, 0.83, 0.80, 1.0)),
    ];

    for &h in &heights {
        let t = if h_range > 0.0 {
            ((h - h_min) / h_range).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let c = if t <= stops[0].0 {
            stops[0].1
        } else if t >= stops[4].0 {
            stops[4].1
        } else {
            let mut lo = 0;
            while stops[lo + 1].0 < t {
                lo += 1;
            }
            let s = (t - stops[lo].0) / (stops[lo + 1].0 - stops[lo].0);
            lerp_color(stops[lo].1, stops[lo + 1].1, s)
        };
        colors.push(c.to_f32_array());
    }

    // -- indices --
    for z in 0..TERRAIN_SIZE {
        for x in 0..TERRAIN_SIZE {
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

    // -- assemble --
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals_arr);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}
