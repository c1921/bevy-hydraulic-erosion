use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin};
use bevy::prelude::*;

use crate::config::{DischargeOverlay, ErosionParams, PauseState};
use crate::vegetation::VegetationPlants;

// ── Marker ──────────────────────────────────────────────────────

#[derive(Component)]
pub(crate) struct InfoText;

// ── Frame counter ───────────────────────────────────────────────

#[derive(Resource, Default)]
pub(crate) struct FrameCount(pub u64);

// ── Startup ─────────────────────────────────────────────────────

pub(crate) fn setup_ui(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(16.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            GlobalZIndex(1000),
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Hydraulic Erosion"),
                TextFont {
                    font_size: 22.0,
                    ..default()
                },
                TextColor(Color::WHITE),
                InfoText,
            ));
        });
}

// ── Update ──────────────────────────────────────────────────────

pub(crate) fn update_ui(
    mut frame: ResMut<FrameCount>,
    pause: Res<PauseState>,
    overlay: Res<DischargeOverlay>,
    params: Res<ErosionParams>,
    plants: Res<VegetationPlants>,
    diagnostics: Res<DiagnosticsStore>,
    mut query: Query<&mut Text, With<InfoText>>,
) {
    frame.0 += 1;

    let Ok(mut text) = query.single_mut() else {
        return;
    };

    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|d| d.smoothed())
        .map(|fps| fps as u32)
        .unwrap_or(0);

    let total_particles = frame.0 * params.cycles_per_frame as u64;

    let status = if pause.0 { "⏸ PAUSED" } else { "▶ RUNNING" };
    let heatmap = if overlay.0 { "ON [M]" } else { "OFF [M]" };

    text.0 = format!(
        "Hydraulic Erosion\n\
         ─────────────────\n\
         Status:      {status}\n\
         Heatmap:     {heatmap}\n\
         FPS:         {fps}\n\
         Frame:       {}\n\
         Particles:   {}\n\
         Plants:      {}\n\
         ── Params ──\n\
         Cycles/frame: {}  [O/P]\n\
         Entrainment:  {:.1}  [K/L]",
        frame.0,
        total_particles,
        plants.plants.len(),
        params.cycles_per_frame,
        params.entrainment,
    );
}

