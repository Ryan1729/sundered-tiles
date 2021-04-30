///! `bi_unit` is short for bilateral-unit. AKA the range [-1.0, 1.0]
///! This range has the following advantages:
///! * uses half of the available precision of a floating point number as 
///!   opposed to say [0.0, 1.0] whch only uses a quarter of it.
///! * +0.0, (the natural default) is in the middle of the range (subjective)

pub use f32_is;

macro_rules! tuple_new_type {
    (
        struct $struct_name: ident,
        macro_rules! 
        $macro_name: ident 
        $_macro_name: ident
        $lt_macro_name: ident
        $_lt_macro_name: ident
        $gt_macro_name: ident
        $_gt_macro_name: ident
    ) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $struct_name(F32);
    
        impl From<$struct_name> for f32 {
            fn from(thing: $struct_name) -> Self {
                Self::from(thing.0)
            }
        }

        /// If you run into an error about `new_saturating` not being allowed in 
        /// `const`s try `$macro_name!(const expression)` instead of 
        /// `$macro_name!(expression)`.
        #[macro_export]
        macro_rules! $_macro_name {
            ($float: literal) => {{
                $macro_name!(const $float)
            }};
            (const $float: expr) => {{
                $crate::const_assert_valid_bi_unit!($float);

                $crate::$struct_name::new_unchecked($float)
            }};
            ($float: expr) => {
                $crate::$struct_name::new_saturating($float)
            };
        }

        /// Re-export the macro so it also lives in this module, as well as at the
        /// root of this crate.
        pub use $_macro_name as $macro_name;

        impl $struct_name {
            pub fn new_saturating(f: f32) -> Self {
                Self(F32::new_saturating(f))
            }

            /// This exists for use in the construction macro, where a const 
            /// assertion performs the checks, allowing this to be a const fn
            /// even though float operations are not allowed in const fn, on 
            /// stable, as of this writing.
            /// Use outside of that macro is heavily discouraged.
            pub const fn new_unchecked(f: f32) -> Self {
                Self(F32::new_unchecked(f))
            }

            pub const fn to_f32(self) -> f32 {
                self.0.0
            }
        }

        impl core::cmp::PartialEq for $struct_name {
            fn eq(&self, other: &Self) -> bool {
                self.0.eq(&other.0)
            }
        }

        impl core::cmp::Eq for $struct_name {}

        impl core::cmp::Ord for $struct_name {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.0.cmp(&other.0)
            }
        }
        
        impl core::cmp::PartialOrd for $struct_name {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        #[macro_export]
        macro_rules! $_lt_macro_name {
            ($x_a: expr, $x_b: expr) => {{
                f32_is::lt!(
                    $crate::$struct_name::to_f32($x_a), 
                    $crate::$struct_name::to_f32($x_b)
                )
            }}
        }

        pub use $_lt_macro_name as $lt_macro_name;

        #[macro_export]
        macro_rules! $_gt_macro_name {
            ($x_a: expr, $x_b: expr) => {{
                f32_is::gt!(
                    $crate::$struct_name::to_f32($x_a), 
                    $crate::$struct_name::to_f32($x_b)
                )
            }}
        }

        pub use $_gt_macro_name as $gt_macro_name;
    }
}

tuple_new_type!{struct X, macro_rules! x _x x_lt _x_lt x_gt _x_gt}
tuple_new_type!{struct Y, macro_rules! y _y y_lt _y_lt y_gt _y_gt}

/// Only works with literals or other expressions that are allowed in `const`s.
#[macro_export]
macro_rules! const_assert_valid_bi_unit {
    ($float: literal) => {
        const_assert_valid_bi_unit!({$float})
    };
    ($float: expr) => {
        #[allow(unknown_lints, eq_op)]
        const _: [(); 0 - !{
            $crate::f32_is::pos_zero!($float)
            || $crate::f32_is::pos_normal_and_one_or_below!($float)
            || $crate::f32_is::neg_normal_and_above_negative_one!($float)
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

impl core::cmp::PartialEq for F32 {
    fn eq(&self, other: &Self) -> bool {
        // We rely on the fact that an `F32` should not contain a NaN.
        self.0.eq(&other.0)
    }
}

impl core::cmp::Eq for F32 {}

impl core::cmp::Ord for F32 {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // We rely on the fact that an `F32` should not contain a NaN.
        self.0.partial_cmp(&other.0).expect("comparing F32 failed!")
    }
}

impl core::cmp::PartialOrd for F32 {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Only works with literals or other expressions that are allowed in `const`s.
macro_rules! F32 {
    ($float: expr) => {{
        const_assert_valid_bi_unit!($float);

        F32::new_unchecked($float)
    }};
}

impl F32 {
    #![allow(unused)]

    pub const MIN: Self = F32!(-1.0);
    pub const NEG_MIN_POSITIVE: Self = F32!(-f32::MIN_POSITIVE);
    pub const ZERO: Self = F32!(0.0);
    pub const MIN_POSITIVE: Self = F32!(f32::MIN_POSITIVE);
    pub const MAX: Self = F32!(1.0);
}

impl F32 {
    pub fn new_saturating(f: f32) -> Self {
        Self(
            // This is known incorrect, to test the tests.
            if f32_is::pos_normal_and_one_or_below!(f)
            || f32_is::neg_normal_and_above_negative_one!(f) {
                f
            } else if f < -1.0 {
                -1.0
            } else if f > 1.0 {
                1.0
            } else {
                // NaN ends up here
                0.0
            }
        )
    }

    const fn new_unchecked(f: f32) -> Self {
        Self(f)
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

#[test]
fn x_lt_is_usable_in_const() {
    const A: bool = x_lt!(x!(0.25), x!(0.75));
    assert!(A);
}

#[test]
fn x_gt_is_usable_in_const() {
    const A: bool = x_gt!(x!(0.75), x!(0.25));
    assert!(A);
}