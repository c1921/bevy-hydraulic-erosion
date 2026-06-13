use bevy::{math::Vec3, prelude::*};

use crate::cell::Cell;
use crate::config::MAP_SCALE;

/// 地形核心数据 — 网格中所有 Cell 的扁平存储
#[derive(Resource)]
pub(crate) struct TerrainResource {
    /// 顶点网格尺寸 = GRID_SIZE + 1
    pub size: usize,
    /// 扁平存储，索引 = z * size + x
    pub cells: Vec<Cell>,
}

impl TerrainResource {
    /// 分配 size×size 个零值 Cell
    pub fn new(size: usize) -> Self {
        Self {
            size,
            cells: vec![Cell::default(); size * size],
        }
    }

    // ── 索引 ─────────────────────────────────────────────────

    /// 将 (x, z) 转为扁平索引
    #[inline]
    fn index(&self, x: usize, z: usize) -> usize {
        z * self.size + x
    }

    // ── 不可变访问 ───────────────────────────────────────────

    /// 获取 Cell 的不可变引用
    #[inline]
    pub fn cell(&self, x: usize, z: usize) -> &Cell {
        &self.cells[self.index(x, z)]
    }

    /// 获取指定位置的高度
    #[inline]
    pub fn height_at(&self, x: usize, z: usize) -> f32 {
        self.cell(x, z).height
    }

    /// 获取 discharge（经 erf 变换，匹配 C++ 行为）
    #[inline]
    pub fn discharge_at(&self, x: usize, z: usize) -> f32 {
        let d = self.cell(x, z).discharge;
        erf_approx(0.4 * d)
    }

    // ── 可变访问 ─────────────────────────────────────────────

    /// 获取 Cell 的可变引用
    #[inline]
    pub fn cell_mut(&mut self, x: usize, z: usize) -> &mut Cell {
        let idx = self.index(x, z);
        &mut self.cells[idx]
    }

    /// 设置指定位置的高度
    #[inline]
    pub fn set_height(&mut self, x: usize, z: usize, h: f32) {
        self.cell_mut(x, z).height = h;
    }

    // ── 边界检查 ─────────────────────────────────────────────

    /// 越界检查（接受 i32 以匹配粒子坐标搜索模式）
    #[inline]
    pub fn oob_i32(&self, x: i32, z: i32) -> bool {
        x < 0 || z < 0 || x >= self.size as i32 || z >= self.size as i32
    }

    /// 越界检查（usize 版本）
    #[inline]
    pub fn oob(&self, x: usize, z: usize) -> bool {
        x >= self.size || z >= self.size
    }

    // ── 法线计算 ─────────────────────────────────────────────
    ///
    /// 计算 (x, z) 处的法线。
    /// 使用四个对角平面平均（与 C++ `_normal()` 一致）。
    ///
    /// 注意：仅在地形内部使用，调用方需保证 (x, z) 及其 ±1 邻居均不越界。
    pub fn normal(&self, x: usize, z: usize) -> Vec3 {
        let s = Vec3::new(1.0, MAP_SCALE, 1.0);
        let h = |dx: i32, dz: i32| -> f32 {
            self.height_at((x as i32 + dx) as usize, (z as i32 + dz) as usize)
        };
        let hc = h(0, 0);
        let mut n = Vec3::ZERO;

        // 平面 (+X, +Z): cross((0, hR-h, 1)*s, (1, hB-h, 0)*s)
        if !self.oob(x + 1, z + 1) {
            n += cross_plane(
                Vec3::new(0.0, h(0, 1) - hc, 1.0) * s,
                Vec3::new(1.0, h(1, 0) - hc, 0.0) * s,
            );
        }

        // 平面 (-X, -Z): cross((0, hL-h, -1)*s, (-1, hT-h, 0)*s)
        if x > 0 && z > 0 {
            n += cross_plane(
                Vec3::new(0.0, h(0, -1) - hc, -1.0) * s,
                Vec3::new(-1.0, h(-1, 0) - hc, 0.0) * s,
            );
        }

        // 平面 (+X, -Z): cross((1, hB-h, 0)*s, (0, hL-h, -1)*s)
        if !self.oob(x + 1, z) && z > 0 {
            n += cross_plane(
                Vec3::new(1.0, h(1, 0) - hc, 0.0) * s,
                Vec3::new(0.0, h(0, -1) - hc, -1.0) * s,
            );
        }

        // 平面 (-X, +Z): cross((-1, hT-h, 0)*s, (0, hR-h, 1)*s)
        if x > 0 && !self.oob(x, z + 1) {
            n += cross_plane(
                Vec3::new(-1.0, h(-1, 0) - hc, 0.0) * s,
                Vec3::new(0.0, h(0, 1) - hc, 1.0) * s,
            );
        }

        if n.length() > 0.0 {
            n.normalize()
        } else {
            n
        }
    }

    // ── Tracking 操作 ──────────────────────────────────────────

    /// 清零所有 cell 的 tracking maps（每次 erode 循环前调用）
    pub fn clear_tracking(&mut self) {
        for cell in &mut self.cells {
            cell.discharge_track = 0.0;
            cell.momentum_x_track = 0.0;
            cell.momentum_y_track = 0.0;
        }
    }

    /// 指数滑动平均：将 tracking maps 合并到持久场
    pub fn merge_tracking(&mut self, learning_rate: f32) {
        let lr = learning_rate;
        let one_minus_lr = 1.0 - lr;
        for cell in &mut self.cells {
            cell.discharge = one_minus_lr * cell.discharge + lr * cell.discharge_track;
            cell.momentum_x = one_minus_lr * cell.momentum_x + lr * cell.momentum_x_track;
            cell.momentum_y = one_minus_lr * cell.momentum_y + lr * cell.momentum_y_track;
        }
    }

    // ── 迭代器 ───────────────────────────────────────────────

    /// 所有 Cell 的可变引用迭代器（跨整个网格）
    pub fn cells_mut_iter(&mut self) -> impl Iterator<Item = &mut Cell> {
        self.cells.iter_mut()
    }

    /// 所有 Cell 的不可变引用迭代器
    pub fn cells_iter(&self) -> impl Iterator<Item = &Cell> {
        self.cells.iter()
    }
}

// ── 辅助函数 ────────────────────────────────────────────────────

fn cross_plane(a: Vec3, b: Vec3) -> Vec3 {
    a.cross(b)
}

/// erf 函数的近似（Abramowitz & Stegun 7.1.26）
fn erf_approx(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

// ── 测试 ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_creates_flat_terrain() {
        let t = TerrainResource::new(4);
        assert_eq!(t.size, 4);
        assert_eq!(t.cells.len(), 16);
        for z in 0..4 {
            for x in 0..4 {
                assert_eq!(t.height_at(x, z), 0.0);
            }
        }
    }

    #[test]
    fn index_and_access() {
        let mut t = TerrainResource::new(4);
        t.set_height(2, 3, 5.0);
        assert_eq!(t.height_at(2, 3), 5.0);
        assert_eq!(t.cell(2, 3).height, 5.0);
    }

    #[test]
    fn oob_checks() {
        let t = TerrainResource::new(4);
        assert!(!t.oob(0, 0));
        assert!(!t.oob(3, 3));
        assert!(t.oob(4, 0));
        assert!(t.oob(0, 4));
        assert!(t.oob_i32(-1, 0));
        assert!(t.oob_i32(0, -1));
        assert!(t.oob_i32(4, 0));
    }

    #[test]
    fn normal_on_flat_terrain_points_up() {
        let t = TerrainResource::new(4);
        let n = t.normal(1, 1);
        // Flat terrain: each diagonal plane gives (0,1,0), sum normalizes to (0,1,0)
        assert!((n - Vec3::Y).length() < 0.001);
    }

    #[test]
    fn normal_on_slope_points_downhill() {
        let mut t = TerrainResource::new(4);
        // Slope descending toward +x (height lowers as x increases)
        t.set_height(0, 1, 0.0);
        t.set_height(1, 1, -1.0 / MAP_SCALE);
        t.set_height(2, 1, -2.0 / MAP_SCALE);
        let n = t.normal(1, 1);
        // The normal returned by this method points downhill (in gradient direction)
        // This is the expected behavior for particle acceleration
        assert!(n.y > 0.0, "normal should point upward");
        assert!(n.x > 0.0, "normal should point downhill (+x), got n.x={}", n.x);
    }

    #[test]
    fn clear_tracking_zeros_all() {
        let mut t = TerrainResource::new(4);
        // Set tracking to non-zero values
        for cell in t.cells_mut_iter() {
            cell.discharge_track = 1.0;
            cell.momentum_x_track = 2.0;
            cell.momentum_y_track = 3.0;
        }
        t.clear_tracking();
        for cell in t.cells_iter() {
            assert_eq!(cell.discharge_track, 0.0);
            assert_eq!(cell.momentum_x_track, 0.0);
            assert_eq!(cell.momentum_y_track, 0.0);
        }
    }

    #[test]
    fn merge_tracking_ema() {
        let mut t = TerrainResource::new(2);
        // persistent = 0 initially, track = 10, lr = 0.5
        t.cell_mut(0, 0).discharge = 0.0;
        t.cell_mut(0, 0).discharge_track = 10.0;
        t.cell_mut(0, 0).momentum_x = 0.0;
        t.cell_mut(0, 0).momentum_x_track = 4.0;
        t.cell_mut(0, 0).momentum_y = 2.0;
        t.cell_mut(0, 0).momentum_y_track = 6.0;

        t.merge_tracking(0.5);

        // discharge = 0.5*0 + 0.5*10 = 5
        assert!((t.cell(0, 0).discharge - 5.0).abs() < 0.001);
        // momentum_x = 0.5*0 + 0.5*4 = 2
        assert!((t.cell(0, 0).momentum_x - 2.0).abs() < 0.001);
        // momentum_y = 0.5*2 + 0.5*6 = 4
        assert!((t.cell(0, 0).momentum_y - 4.0).abs() < 0.001);
    }
}
