use bevy::prelude::*;

// ── Global constants ────────────────────────────────────────────

/// 地形网格分辨率（顶点数 = GRID_SIZE + 1）
pub(crate) const GRID_SIZE: usize = 512;

/// Tile 大小（为未来分块预埋，当前等于 GRID_SIZE 即单 tile）
pub(crate) const TILE_SIZE: usize = 512;

/// Mesh 空间中每个格子的尺寸
pub(crate) const CELL_SIZE: f32 = 2.0;

/// 用于法线计算时的高度缩放因子（对应 C++ quad::mapscale）
pub(crate) const MAP_SCALE: f32 = 80.0;

// ── Noise generation defaults ───────────────────────────────────

pub(crate) const NOISE_SCALE: f64 = 0.015;
pub(crate) const HEIGHT_AMP: f32 = 30.0;
pub(crate) const OCTAVES: usize = 6;
pub(crate) const LACUNARITY: f64 = 2.0;
pub(crate) const PERSISTENCE: f64 = 0.5;

// ── Erosion parameters ──────────────────────────────────────────

/// 渲染高度缩放倍数（相对 HEIGHT_AMP）
pub(crate) const HEIGHT_SCALE: f32 = 3.0;

/// 暂停侵蚀模拟
#[derive(Resource, Default)]
pub(crate) struct PauseState(pub bool);

/// M 键切换：显示 discharge 热力图
#[derive(Resource, Default)]
pub(crate) struct DischargeOverlay(pub bool);

#[derive(Resource)]
pub(crate) struct ErosionParams {
    pub learning_rate: f32,
    pub max_age: u32,
    pub min_volume: f32,
    pub evaporation_rate: f32,
    pub deposition_rate: f32,
    pub entrainment: f32,
    pub gravity: f32,
    pub momentum_transfer: f32,
    pub max_diff: f32,
    pub settling_rate: f32,
    pub cycles_per_frame: usize,
}

impl Default for ErosionParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            max_age: 500,
            min_volume: 0.01,
            evaporation_rate: 0.001,
            deposition_rate: 0.1,
            entrainment: 10.0,
            gravity: 1.0,
            momentum_transfer: 1.0,
            max_diff: 0.01,
            settling_rate: 0.8,
            cycles_per_frame: 512,
        }
    }
}
