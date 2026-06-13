use bevy::{color::palettes::css, math::Vec2, prelude::*};
use rand::Rng;

use crate::config::PauseState;
use crate::terrain_data::TerrainResource;

// ── Plant parameters ────────────────────────────────────────────

#[derive(Resource)]
pub(crate) struct PlantParams {
    pub max_size: f32,
    pub grow_rate: f32,
    /// 法线 y 分量阈值：低于此值视为太陡
    pub max_steep: f32,
    /// discharge (经 erf) 阈值：高于此值无法存活
    pub max_discharge: f32,
    /// 海拔阈值：高于此值无法存活
    pub max_height: f32,
}

impl Default for PlantParams {
    fn default() -> Self {
        Self {
            max_size: 1.5,
            grow_rate: 0.05,
            max_steep: 0.8,
            max_discharge: 0.3,
            max_height: 0.8,
        }
    }
}

// ── Plant ────────────────────────────────────────────────────────

pub(crate) struct Plant {
    /// 网格坐标（整数 + 亚像素）
    pub pos: Vec2,
    /// 当前大小
    pub size: f32,
}

impl Plant {
    fn grow(&mut self, params: &PlantParams) {
        self.size += params.grow_rate * (params.max_size - self.size);
    }
}

// ── Vegetation resource ─────────────────────────────────────────

#[derive(Resource, Default)]
pub(crate) struct VegetationPlants {
    pub plants: Vec<Plant>,
}

// ── Root density helpers ────────────────────────────────────────

/// 在 pos 及其 8 邻域增减 root_density（与 C++ `Plant::root()` 完全一致）
fn root_apply(terrain: &mut TerrainResource, pos: Vec2, factor: f32) {
    let ix = pos.x as i32;
    let iz = pos.y as i32;

    // (dx, dz, weight)
    let kernel: [(i32, i32, f32); 9] = [
        (0, 0, 1.0),
        (1, 0, 0.6),
        (-1, 0, 0.6),
        (0, 1, 0.6),
        (0, -1, 0.6),
        (-1, -1, 0.4),
        (1, -1, 0.4),
        (-1, 1, 0.4),
        (1, 1, 0.4),
    ];

    for &(dx, dz, w) in &kernel {
        let nx = ix + dx;
        let nz = iz + dz;
        if terrain.oob_i32(nx, nz) {
            continue;
        }
        terrain.cell_mut(nx as usize, nz as usize).root_density += factor * w;
    }
}

// ── Spawn / Die checks ──────────────────────────────────────────

fn can_spawn(pos: Vec2, terrain: &TerrainResource, params: &PlantParams) -> bool {
    let ix = pos.x as usize;
    let iz = pos.y as usize;
    if terrain.oob(ix, iz) {
        return false;
    }
    if terrain.discharge_at(ix, iz) >= params.max_discharge {
        return false;
    }
    let n = terrain.normal(ix, iz);
    if n.y < params.max_steep {
        return false;
    }
    if terrain.height_at(ix, iz) >= params.max_height {
        return false;
    }
    true
}

fn should_die(pos: Vec2, terrain: &TerrainResource, params: &PlantParams) -> bool {
    let ix = pos.x as usize;
    let iz = pos.y as usize;
    if terrain.oob(ix, iz) {
        return true;
    }
    if terrain.discharge_at(ix, iz) >= params.max_discharge {
        return true;
    }
    if terrain.height_at(ix, iz) >= params.max_height {
        return true;
    }
    // 1/1000 随机死亡
    if rand::rng().random_range(0..1000) == 0 {
        return true;
    }
    false
}

fn try_reproduce(
    parent_pos: Vec2,
    terrain: &TerrainResource,
    params: &PlantParams,
) -> Option<Vec2> {
    let mut rng = rand::rng();
    // 5% 概率
    if rng.random_range(0..20) != 0 {
        return None;
    }
    let npos = parent_pos + Vec2::new(
        rng.random_range(-4..5) as f32,
        rng.random_range(-4..5) as f32,
    );
    let nx = npos.x as i32;
    let nz = npos.y as i32;
    if terrain.oob_i32(nx, nz) {
        return None;
    }
    let ux = nx as usize;
    let uz = nz as usize;
    if terrain.discharge_at(ux, uz) >= params.max_discharge {
        return None;
    }
    // root_density 抑制繁殖
    let rd = terrain.cell(ux, uz).root_density;
    if rng.random_range(0.0..1.0) <= rd {
        return None;
    }
    let n = terrain.normal(ux, uz);
    if n.y <= params.max_steep {
        return None;
    }
    Some(npos)
}

// ── Update system ───────────────────────────────────────────────

pub(crate) fn update_vegetation(
    mut terrain: ResMut<TerrainResource>,
    mut plants: ResMut<VegetationPlants>,
    params: Res<PlantParams>,
    pause: Res<PauseState>,
) {
    if pause.0 {
        return;
    }

    let mut rng = rand::rng();
    let size = terrain.size;

    // (a) 随机位置尝试生成一株
    {
        let x = rng.random_range(0..size);
        let z = rng.random_range(0..size);
        let pos = Vec2::new(x as f32, z as f32);
        if can_spawn(pos, &terrain, &params) {
            plants.plants.push(Plant { pos, size: 0.0 });
            root_apply(&mut terrain, pos, 1.0);
        }
    }

    // (b) 遍历现有植物
    let mut i = 0;
    while i < plants.plants.len() {
        plants.plants[i].grow(&params);

        if should_die(plants.plants[i].pos, &terrain, &params) {
            root_apply(&mut terrain, plants.plants[i].pos, -1.0);
            plants.plants.swap_remove(i);
            // 不递增 i，继续检查换过来的植物
            continue;
        }

        // 尝试繁殖
        if let Some(child_pos) = try_reproduce(plants.plants[i].pos, &terrain, &params) {
            plants.plants.push(Plant { pos: child_pos, size: 0.0 });
            root_apply(&mut terrain, child_pos, 1.0);
        }

        i += 1;
    }
}

// ── Gizmo visualization ─────────────────────────────────────────

pub(crate) fn draw_plants(
    plants: Res<VegetationPlants>,
    mut gizmos: Gizmos,
) {
    use crate::config::CELL_SIZE;

    for plant in &plants.plants {
        // 网格坐标 → 世界坐标（xz 平面）
        let world_pos = Vec3::new(
            plant.pos.x * CELL_SIZE,
            0.0, // y 由 height 决定，在 update_mesh 之后读取
            plant.pos.y * CELL_SIZE,
        );
        // 读取实际高度（从 gizmo 无法直接访问 TerrainResource）
        // 使用简化的 y：在 height=0.5 处绘制，半径 = size * 2
        let y = 0.5 * crate::config::HEIGHT_AMP * crate::config::HEIGHT_SCALE;
        let center = Vec3::new(world_pos.x, y + plant.size * 2.0, world_pos.z);
        let radius = (plant.size * 3.0).max(0.5);
        gizmos.sphere(center, radius, css::GREEN);
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_params() -> PlantParams {
        PlantParams::default()
    }

    fn make_flat_terrain(size: usize) -> TerrainResource {
        let mut t = TerrainResource::new(size);
        for z in 0..size {
            for x in 0..size {
                t.set_height(x, z, 0.5);
            }
        }
        t
    }

    #[test]
    fn grow_increases_size() {
        let mut plant = Plant { pos: Vec2::ZERO, size: 0.1 };
        let params = make_params();
        plant.grow(&params);
        assert!(plant.size > 0.1);
    }

    #[test]
    fn can_spawn_on_flat_ground() {
        let t = make_flat_terrain(8);
        let params = make_params();
        // Flat terrain at h=0.5, discharge=0, steepness OK
        assert!(can_spawn(Vec2::new(3.0, 3.0), &t, &params));
    }

    #[test]
    fn cannot_spawn_on_steep() {
        let mut t = make_flat_terrain(8);
        // 制造陡坡：高度急剧变化 → normal.y 很小
        for x in 0..8 {
            t.set_height(x, 3, 1.0 - 0.5 * x as f32);
        }
        // 注意：can_spawn 使用 normal()，要求 n.y >= max_steep (0.8)
        // 陡坡上 n.y 会显著低于 0.8
        let params = make_params();
        assert!(!can_spawn(Vec2::new(3.0, 3.0), &t, &params));
    }

    #[test]
    fn should_die_at_high_discharge() {
        let mut t = make_flat_terrain(8);
        // 设置高 discharge（erf(0.4*d) ≥ 0.3 需要 d 足够大）
        t.cell_mut(3, 3).discharge = 2.0; // erf(0.8) ≈ 0.74 > 0.3
        let params = make_params();
        assert!(should_die(Vec2::new(3.0, 3.0), &t, &params));
    }

    #[test]
    fn root_applies_to_self_and_neighbors() {
        let mut t = make_flat_terrain(8);
        root_apply(&mut t, Vec2::new(3.0, 3.0), 1.0);

        // 中心
        assert!((t.cell(3, 3).root_density - 1.0).abs() < 0.001);
        // 4 邻
        assert!((t.cell(4, 3).root_density - 0.6).abs() < 0.001);
        assert!((t.cell(2, 3).root_density - 0.6).abs() < 0.001);
        // 4 角
        assert!((t.cell(4, 4).root_density - 0.4).abs() < 0.001);
    }

    #[test]
    fn root_removal_reverses() {
        let mut t = make_flat_terrain(8);
        root_apply(&mut t, Vec2::new(3.0, 3.0), 1.0);
        root_apply(&mut t, Vec2::new(3.0, 3.0), -1.0);

        // 中心应该归零
        assert!((t.cell(3, 3).root_density).abs() < 0.001);
        assert!((t.cell(4, 3).root_density).abs() < 0.001);
    }
}
