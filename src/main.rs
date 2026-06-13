mod camera;
mod cell;
mod config;
mod erosion;
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
        ))
        .add_systems(PreUpdate, (
            terrain::toggle_pause,
            terrain::toggle_overlay,
            terrain::tune_params,
        ))
        .add_systems(Update, (
            terrain::update_erosion,
            vegetation::update_vegetation,
            terrain::update_mesh,
            vegetation::draw_plants,
            ui::update_ui,
            camera::orbit_camera,
        ))
        .run();
}
