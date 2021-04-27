///! `unit` in this case refers to the range [0.0, 1.0]

pub use f32_is;

macro_rules! tuple_new_type {
    (struct $struct_name: ident, macro_rules! $macro_name: ident $_macro_name: ident) => {
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
                $crate::const_assert_valid_unit!($float);

                $struct_name::new_unchecked($float)
            }};
            ($float: expr) => {
                $struct_name::new_saturating($float)
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
            pub fn new_unchecked(f: f32) -> Self {
                Self(F32::new_unchecked(f))
            }
        }
    }
}

tuple_new_type!{struct W, macro_rules! w _w}
tuple_new_type!{struct H, macro_rules! h _h}
tuple_new_type!{struct Proportion, macro_rules! proportion _proportion}

use core::ops::Mul;

impl Mul<W> for Proportion {
    type Output = W;

    fn mul(self, rhs: W) -> Self::Output {
        let p: f32 = self.into();
        let w: f32 = rhs.into();

        w!(p * w)
    }
}

impl Mul<Proportion> for W {
    type Output = Self;

    fn mul(self, rhs: Proportion) -> Self::Output {
        let w: f32 = self.into();
        let p: f32 = rhs.into();

        w!(w * p)
    }
}

impl Mul<H> for Proportion {
    type Output = H;

    fn mul(self, rhs: H) -> Self::Output {
        let p: f32 = self.into();
        let h: f32 = rhs.into();

        h!(p * h)
    }
}

impl Mul<Proportion> for H {
    type Output = Self;

    fn mul(self, rhs: Proportion) -> Self::Output {
        let h: f32 = self.into();
        let p: f32 = rhs.into();

        h!(h * p)
    }
}

/// Only works with literals or other expressions that are allowed in `const`s.
#[macro_export]
macro_rules! const_assert_valid_unit {
    ($float: literal) => {
        const_assert_valid_unit!({$float})
    };
    ($float: expr) => {
        #[allow(unknown_lints, eq_op)]
        const _: [(); 0 - !{
            $crate::f32_is::pos_zero!($float)
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
        const_assert_valid_unit!($float);

        F32::new_unchecked($float)
    }};
}

impl F32 {
    #![allow(unused)]

    pub const ZERO: Self = F32!(0.0);
    pub const MIN_POSITIVE: Self = F32!(f32::MIN_POSITIVE);
    pub const ONE: Self = F32!(1.0);
    pub const TWO: Self = F32!(2.0);
    pub const INFINITY: Self = F32!(f32::INFINITY);
}

impl F32 {
    pub fn new_saturating(f: f32) -> Self {
        Self(
            if f32_is::pos_normal_or_infinity!(f) {
                f
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