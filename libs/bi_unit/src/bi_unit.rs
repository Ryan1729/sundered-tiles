///! `bi_unit` is short for bilateral-unit. AKA the range [-1.0, 1.0]
///! This range has the following advantages:
///! * uses half of the available precision of a floating point number as 
///!   opposed to say [0.0, 1.0] whch only uses a quarter of it.
///! * +0.0, (the natural default) is in the middle of the range (subjective)

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

        #[macro_export]
        macro_rules! $macro_name {
            ($float: literal) => {{
                const_assert_valid!($float);

                $crate::$struct_name::new_saturating($float)
            }};
            ($float: expr) => {
                $crate::$struct_name::new_saturating($float)
            };
        }

        impl $struct_name {
            pub const fn new_saturating(f: f32) -> Self {
                Self(F32::new_saturating(f))
            }
        }
    }
}

tuple_new_type!{struct X, macro_rules! x}
tuple_new_type!{struct Y, macro_rules! y}

#[macro_export]
macro_rules! const_assert_valid {
    ($f32: literal) => {
        #[allow(unknown_lints, eq_op)]
        const _: [(); 0 - !{
            $crate::f32_is::is_pos_zero!($float)
            || $crate::f32_is::is_pos_normal_and_one_or_below!($float)
            || $crate::f32_is::is_neg_normal_and_above_negative_one!($float)
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
    pub const MIN: Self = F32(-1.0);
    pub const NEG_MIN_POSITIVE: Self = F32(-f32::MIN_POSITIVE);
    pub const ZERO: Self = F32(0.0);
    pub const MIN_POSITIVE: Self = F32(f32::MIN_POSITIVE);
    pub const MAX: Self = F32(1.0);

    pub const fn new_saturating(f: f32) -> Self {
        Self(
            // This is known incorrect, to test the tests.
            if f32_is::pos_normal_and_one_or_below!(f) {
                f
            } else if f32_is::neg_normal_and_above_negative_one!(f) {
                f
            } else {
                0.0
            }
        )
    }
}

#[test]
fn new_saturating_saturates_properly_on_these_edge_cases() {
    assert_eq!(F32::new_saturating(-f32::INFINITY), F32::MIN);
    assert_eq!(F32::new_saturating(-2.0), F32::MIN);
    assert_eq!(F32::new_saturating(-1.0), F32::MIN);

    assert_eq!(F32::new_saturating(-f32::MIN_POSITIVE), F32::NEG_MIN_POSITIVE);
    assert_eq!(F32::new_saturating(-f32::MIN_POSITIVE / 2.0), F32::ZERO);

    assert_eq!(F32::new_saturating(-0.0), F32::ZERO);
    assert_eq!(F32::new_saturating(f32::NAN), F32::ZERO);
    assert_eq!(F32::new_saturating(0.0), F32::ZERO);

    assert_eq!(F32::new_saturating(f32::MIN_POSITIVE / 2.0), F32::ZERO);
    assert_eq!(F32::new_saturating(f32::MIN_POSITIVE), F32::MIN_POSITIVE);

    assert_eq!(F32::new_saturating(1.0), F32::MAX);
    assert_eq!(F32::new_saturating(2.0), F32::MAX);
    assert_eq!(F32::new_saturating(f32::INFINITY), F32::MAX);
}