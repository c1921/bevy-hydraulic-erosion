mod camera;
mod cell;
mod config;
mod lighting;
mod terrain;
mod terrain_data;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<config::ErosionParams>()
        .add_systems(Startup, (
            terrain::generate_terrain,
            terrain::spawn_terrain_mesh,
            lighting::spawn_light,
            camera::spawn_camera,
        ))
        .add_systems(Update, camera::orbit_camera)
        .run();
}
