use bevy::{input::mouse::AccumulatedMouseScroll, prelude::*};

use crate::config::{CELL_SIZE, GRID_SIZE, HEIGHT_AMP, HEIGHT_SCALE};

// ── Camera controller resource ──────────────────────────────────

#[derive(Resource)]
pub(crate) struct CameraController {
    pub(crate) yaw: f32,
    pub(crate) pitch: f32,
    pub(crate) distance: f32,
    pub(crate) center: Vec3,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            yaw: std::f32::consts::FRAC_PI_4,
            pitch: 0.8,
            distance: 250.0,
            center: Vec3::new(
                GRID_SIZE as f32 * CELL_SIZE / 2.0,
                HEIGHT_AMP * HEIGHT_SCALE / 2.0,
                GRID_SIZE as f32 * CELL_SIZE / 2.0,
            ),
        }
    }
}

// ── Startup system ──────────────────────────────────────────────

pub(crate) fn spawn_camera(mut commands: Commands) {
    let ctrl = CameraController::default();
    let pos = camera_position(&ctrl);
    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(pos).looking_at(ctrl.center, Vec3::Y),
    ));
    commands.insert_resource(ctrl);
}

// ── Orbit camera system ─────────────────────────────────────────

pub(crate) fn camera_position(ctrl: &CameraController) -> Vec3 {
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

pub(crate) fn orbit_camera(
    mut query: Query<&mut Transform, With<Camera3d>>,
    mut ctrl: ResMut<CameraController>,
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    scroll: Res<AccumulatedMouseScroll>,
) {
    let Ok(mut cam_transform) = query.single_mut() else {
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
    ctrl.distance -= scroll.delta.y * 30.0;
    ctrl.distance = ctrl.distance.clamp(30.0, 600.0);

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
