use bevy::{
    input::mouse::MouseWheel,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use noise::{NoiseFn, Perlin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_resource(ClearColor(Color::srgb(0.4, 0.6, 0.9)))
        .add_systems(Startup, setup)
        .add_systems(Update, orbit_camera)
        .run();
}

// ── Terrain config ──────────────────────────────────────────────

const TERRAIN_SIZE: usize = 512;
const CELL_SIZE: f32 = 2.0;
const NOISE_SCALE: f64 = 0.015;
const HEIGHT_AMP: f32 = 30.0;
const OCTAVES: usize = 6;
const LACUNARITY: f64 = 2.0;
const PERSISTENCE: f64 = 0.5;

// ── Camera controller resource ──────────────────────────────────

#[derive(Resource)]
struct CameraController {
    yaw: f32,
    pitch: f32,
    distance: f32,
    center: Vec3,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.8,
            distance: 250.0,
            center: Vec3::new(
                TERRAIN_SIZE as f32 * CELL_SIZE / 2.0,
                5.0,
                TERRAIN_SIZE as f32 * CELL_SIZE / 2.0,
            ),
        }
    }
}

// ── Startup ─────────────────────────────────────────────────────

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // ── terrain mesh ──
    let mesh = generate_terrain_mesh();
    commands.spawn((
        Mesh3d(meshes.add(mesh)),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE, // multiplied with vertex colors
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::default(),
    ));

    // ── directional light (sun) ──
    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -0.7,
            1.2,
            0.0,
        )),
    ));

    // ── ambient light ──
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 400.0,
    });

    // ── camera ──
    let ctrl = CameraController::default();
    let pos = camera_position(&ctrl);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(pos).looking_at(ctrl.center, Vec3::Y),
    ));
    commands.insert_resource(ctrl);
}

// ── Terrain mesh generation ─────────────────────────────────────

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
    let n = TERRAIN_SIZE + 1; // vertices per side
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

    // colour stops — realistic topographic (no water)
    let stops: [(f32, LinearRgba); 5] = [
        (0.00, LinearRgba::new(0.15, 0.35, 0.10, 1.0)), // lowland dark green
        (0.30, LinearRgba::new(0.25, 0.55, 0.15, 1.0)), // mid green
        (0.55, LinearRgba::new(0.55, 0.50, 0.20, 1.0)), // yellow-tan
        (0.80, LinearRgba::new(0.40, 0.28, 0.15, 1.0)), // brown
        (1.00, LinearRgba::new(0.85, 0.83, 0.80, 1.0)), // grey-white peak
    ];

    for &h in &heights {
        let t = if h_range > 0.0 {
            ((h - h_min) / h_range).clamp(0.0, 1.0)
        } else {
            0.5
        };
        // linear interpolate between stops
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

    // -- indices (two triangles per quad) --
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

    // -- assemble mesh --
    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals_arr);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));

    mesh
}

// ── Orbit camera system ─────────────────────────────────────────

fn camera_position(ctrl: &CameraController) -> Vec3 {
    let yaw = ctrl.yaw;
    let pitch = ctrl.pitch;
    let d = ctrl.distance;
    ctrl.center
        + Vec3::new(
            d * pitch.cos() * yaw.sin(),
            d * pitch.sin(),
            d * pitch.cos() * yaw.cos(),
        )
}

fn orbit_camera(
    mut query: Query<&mut Transform, With<Camera3d>>,
    mut ctrl: ResMut<CameraController>,
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    mut scroll_ev: EventReader<MouseWheel>,
) {
    let Ok(mut cam_transform) = query.get_single_mut() else {
        return;
    };
    let dt = time.delta_secs();

    // ── orbit (arrow keys) ──
    if keys.pressed(KeyCode::ArrowLeft) {
        ctrl.yaw -= 1.5 * dt;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        ctrl.yaw += 1.5 * dt;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        ctrl.pitch += 1.0 * dt;
        ctrl.pitch = ctrl
            .pitch
            .clamp(0.05, std::f32::consts::FRAC_PI_2 - 0.02);
    }
    if keys.pressed(KeyCode::ArrowDown) {
        ctrl.pitch -= 1.0 * dt;
        ctrl.pitch = ctrl
            .pitch
            .clamp(0.05, std::f32::consts::FRAC_PI_2 - 0.02);
    }

    // ── zoom (scroll wheel) ──
    for ev in scroll_ev.read() {
        ctrl.distance -= ev.y * 30.0;
        ctrl.distance = ctrl.distance.clamp(30.0, 600.0);
    }

    // ── pan (WASD) ──
    let forward = Vec3::new(ctrl.yaw.sin(), 0.0, ctrl.yaw.cos()).normalize();
    let right = Vec3::new(ctrl.yaw.cos(), 0.0, -ctrl.yaw.sin()).normalize();

    let mut pan = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        pan -= forward;
    }
    if keys.pressed(KeyCode::KeyS) {
        pan += forward;
    }
    if keys.pressed(KeyCode::KeyA) {
        pan -= right;
    }
    if keys.pressed(KeyCode::KeyD) {
        pan += right;
    }
    if pan.length_squared() > 0.0 {
        let speed = 60.0 * dt * (ctrl.distance / 150.0);
        ctrl.center += pan.normalize_or_zero() * speed;
    }

    // ── apply ──
    *cam_transform = Transform::from_translation(camera_position(&ctrl))
        .looking_at(ctrl.center, Vec3::Y);
}
