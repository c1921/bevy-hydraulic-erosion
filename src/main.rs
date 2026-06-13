mod camera;
mod cell;
mod config;
mod erosion;
mod gpu_erosion;
mod lighting;
mod terrain;
mod terrain_data;
mod ui;
mod vegetation;

use bevy::prelude::*;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins,
            FrameTimeDiagnosticsPlugin::default(),
            gpu_erosion::GpuErosionPlugin,
        ))
        .init_resource::<config::ErosionParams>()
        .init_resource::<config::PauseState>()
        .init_resource::<config::DischargeOverlay>()
        .init_resource::<vegetation::PlantParams>()
        .init_resource::<vegetation::VegetationPlants>()
        .init_resource::<ui::FrameCount>()
        .add_systems(Startup, (
            terrain::setup_terrain,
            lighting::spawn_light,
            camera::spawn_camera,
            ui::setup_ui,
            gpu_erosion::init_gpu,
        ))
        .add_systems(PreUpdate, (
            terrain::toggle_pause,
            terrain::toggle_overlay,
            terrain::tune_params,
            toggle_gpu_mode,
        ))
        .add_systems(Update, (
            terrain::update_erosion,
            gpu_erosion::gpu_erode,
            vegetation::update_vegetation,
            terrain::update_mesh,
            vegetation::draw_plants,
            ui::update_ui,
            camera::orbit_camera,
        ))
        .run();
}

fn toggle_gpu_mode(
    keys: Res<ButtonInput<KeyCode>>,
    mut gpu: ResMut<gpu_erosion::GpuMode>,
    mut pause: ResMut<config::PauseState>,
) {
    if keys.just_pressed(KeyCode::KeyG) {
        gpu.0 = !gpu.0;
        pause.0 = false; // unpause when switching
        info!("GPU erosion: {}", if gpu.0 { "ON (GPU compute)" } else { "OFF (CPU)" });
    }
}
