use bevy::light::{CascadeShadowConfigBuilder, DirectionalLightShadowMap};
use bevy::prelude::*;

// ── Startup system ──────────────────────────────────────────────

pub(crate) fn spawn_light(mut commands: Commands) {
    // sky color
    commands.insert_resource(ClearColor(Color::srgb(0.4, 0.6, 0.9)));

    // directional light (sun)
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
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            maximum_distance: 1500.0,
            first_cascade_far_bound: 200.0,
            ..default()
        }
        .build(),
    ));

    // ambient light
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: 400.0,
        ..default()
    });

    // shadow map resolution
    commands.insert_resource(DirectionalLightShadowMap { size: 4096 });
}
