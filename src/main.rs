mod camera;
mod lighting;
mod terrain;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, (lighting::spawn_light, terrain::spawn_terrain, camera::spawn_camera))
        .add_systems(Update, camera::orbit_camera)
        .run();
}
