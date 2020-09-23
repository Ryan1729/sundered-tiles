///! `unit` in this case refers to the range [0.0, 1.0]

pub use f32_is;

macro_rules! tuple_new_type {
    (struct $struct_name: ident, macro_rules! $macro_name: ident) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $struct_name(F32);
    
        impl From<$struct_name> for f32 {
            fn from(thing: $struct_name) -> Self {
                Self::from(thing.0)
            }
        }

        impl $struct_name {
            const fn new_saturating(f: f32) -> Self {
                $struct_name(F32::new_saturating(f))
            }
        }

        #[macro_export]
        macro_rules! $macro_name {
            ($f32: literal) => {
                $crate::$struct_name::from_f32($f32)
            };
            ($e: expr) => {
                $crate::$struct_name($e)
            };
        }
    }
}

tuple_new_type!{struct W, macro_rules! w}
tuple_new_type!{struct H, macro_rules! h}

#[macro_export]
macro_rules! const_assert_valid {
    ($f32: literal) => {
        #[allow(unknown_lints, eq_op)]
        const _: [(); 0 - !{
            $crate::f32_is::is_pos_zero!($float)
            || $crate::f32_is::pos_normal_or_infinity!($float)
         } as usize] = [];
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct F32(f32);

impl From<F32> for f32 {
    fn from(f: F32) -> Self {
        f.0
    }
}

impl F32 {
    pub const ZERO: Self = F32(0.0);
    pub const MIN_POSITIVE: Self = F32(f32::MIN_POSITIVE);
    pub const ONE: Self = F32(1.0);
    pub const TWO: Self = F32(2.0);
    pub const INFINITY: Self = F32(f32::INFINITY);

    pub const fn new_saturating(f: f32) -> Self {
        Self(
            if f32_is::pos_normal_or_infinity!(f) {
                f
            } else {
                // NaN ends up here
                0.0
            }
        )
    }
}

#[test]
fn new_saturating_saturates_properly_on_these_edge_cases() {
    assert_eq!(F32::new_saturating(-f32::INFINITY), F32::ZERO);
    assert_eq!(F32::new_saturating(-2.0), F32::ZERO);
    assert_eq!(F32::new_saturating(-1.0), F32::ZERO);

    assert_eq!(F32::new_saturating(-f32::MIN_POSITIVE), F32::ZERO);
    assert_eq!(F32::new_saturating(-f32::MIN_POSITIVE / 2.0), F32::ZERO);

    assert_eq!(F32::new_saturating(-0.0), F32::ZERO);
    assert_eq!(F32::new_saturating(f32::NAN), F32::ZERO);
    assert_eq!(F32::new_saturating(0.0), F32::ZERO);

    assert_eq!(F32::new_saturating(f32::MIN_POSITIVE / 2.0), F32::ZERO);
    assert_eq!(F32::new_saturating(f32::MIN_POSITIVE), F32::MIN_POSITIVE);

    assert_eq!(F32::new_saturating(1.0), F32::ONE);
    assert_eq!(F32::new_saturating(2.0), F32::TWO);
    assert_eq!(F32::new_saturating(f32::INFINITY), F32::INFINITY);
}