mod camera;
mod cell;
mod config;
mod erosion;
mod lighting;
mod terrain;
mod terrain_data;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<config::ErosionParams>()
        .init_resource::<config::PauseState>()
        .add_systems(Startup, (
            terrain::setup_terrain,
            lighting::spawn_light,
            camera::spawn_camera,
        ))
        .add_systems(PreUpdate, terrain::toggle_pause)
        .add_systems(Update, (
            terrain::update_erosion,
            terrain::update_mesh,
            camera::orbit_camera,
        ))
        .run();
}
