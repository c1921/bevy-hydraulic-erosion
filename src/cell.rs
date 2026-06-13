use bytemuck::{Pod, Zeroable};

/// 地形格点 — 所有侵蚀相关数据
///
/// `#[repr(C)]` + `Pod` 保证内存布局与 C 兼容，未来可直接映射为 GPU storage buffer。
#[repr(C)]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
pub(crate) struct Cell {
    // ── 持久场 (persistent fields) ──
    pub height: f32,
    pub discharge: f32,
    pub momentum_x: f32,
    pub momentum_y: f32,

    // ── 逐帧跟踪缓存 (tracking maps) ──
    /// 每轮 erosion cycle 开始时清零
    pub discharge_track: f32,
    pub momentum_x_track: f32,
    pub momentum_y_track: f32,

    // ── 植被影响 ──
    /// 根系密度 [0, ~)，减少侵蚀
    pub root_density: f32,
}

// 编译期保证：GPU storage buffer 要求 Pod + Zeroable，且 size = 32 bytes
const _: () = assert!(std::mem::size_of::<Cell>() == 32);
const _: () = assert!(std::mem::align_of::<Cell>() == 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<Cell>(), 32);
    }

    #[test]
    fn cell_align_is_4() {
        assert_eq!(std::mem::align_of::<Cell>(), 4);
    }

    #[test]
    fn cell_default_is_zero() {
        let c = Cell::default();
        assert_eq!(c.height, 0.0);
        assert_eq!(c.discharge, 0.0);
        assert_eq!(c.momentum_x, 0.0);
        assert_eq!(c.momentum_y, 0.0);
        assert_eq!(c.discharge_track, 0.0);
        assert_eq!(c.momentum_x_track, 0.0);
        assert_eq!(c.momentum_y_track, 0.0);
        assert_eq!(c.root_density, 0.0);
    }

    #[test]
    fn cell_is_pod_and_zeroable() {
        // Compile-time guaranteed by derive macros;
        // runtime sanity: round-trip through bytes
        let c = Cell {
            height: 0.5,
            discharge: 0.1,
            momentum_x: -0.2,
            momentum_y: 0.3,
            discharge_track: 0.01,
            momentum_x_track: -0.02,
            momentum_y_track: 0.03,
            root_density: 0.0,
        };
        let bytes = bytemuck::bytes_of(&c);
        let c2: &Cell = bytemuck::from_bytes(bytes);
        assert_eq!(c2.height, 0.5);
        assert_eq!(c2.momentum_x, -0.2);
    }
}
