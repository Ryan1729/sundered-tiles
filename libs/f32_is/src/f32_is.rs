#[macro_export]
macro_rules! pos_normal_or_infinity {
    ($f32: expr) => {
        $f32 >= f32::MIN_POSITIVE
    }
}

#[macro_export]
macro_rules! pos_zero {
    ($f32: expr) => {{
        let f: f32 = $f32;

        f == 0.0 && {
            // This is a const version of f32::is_sign_positive
            let f_bits = unsafe {
                core::mem::transmute::<f32, u32>(f)
            };

            f_bits & 0x8000_0000 == 0
        }
    }}
}

#[macro_export]
macro_rules! pos_normal_and_one_or_below {
    ($f32: expr) => {{
        let f: f32 = $f32;

        f >= f32::MIN_POSITIVE && f <= 1.0
    }}
}

#[macro_export]
macro_rules! neg_normal_and_above_negative_one {
    ($f32: expr) => {{
        let f: f32 = $f32;

        f >= -1.0 && f <= -f32::MIN_POSITIVE
    }}
}

#[macro_export]
macro_rules! lt {
    ($f32_a: expr, $f32_b: expr) => {{
        // This is a const version of `<`
        let a_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_a)
        };

        let b_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_b)
        };

        // Without this transformation sorting [1., 2., -1., -2.] in ascending order
        // with this operator will result in [ -1., -2., 1., 2.,] instead of the
        // expected [-2., -1., 1., 2.,].
        let a_two_comp = a_bits ^ ((a_bits >> 31) & 0x7fff_ffff);
        let b_two_comp = b_bits ^ ((b_bits >> 31) & 0x7fff_ffff);

        a_two_comp < b_two_comp
    }}
}

#[macro_export]
macro_rules! le {
    ($f32_a: expr, $f32_b: expr) => {{
        // This is a const version of `<=`
        let a_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_a)
        };

        let b_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_b)
        };

        // Without this transformation sorting [1., 2., -1., -2.] in ascending order
        // with this operator will result in [ -1., -2., 1., 2.,] instead of the
        // expected [-2., -1., 1., 2.,].
        let a_two_comp = a_bits ^ ((a_bits >> 31) & 0x7fff_ffff);
        let b_two_comp = b_bits ^ ((b_bits >> 31) & 0x7fff_ffff);

        a_two_comp <= b_two_comp
    }}
}

#[macro_export]
macro_rules! gt {
    ($f32_a: expr, $f32_b: expr) => {{
        // This is a const version of `>`
        let a_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_a)
        };

        let b_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_b)
        };

        // Without this transformation sorting [1., 2., -1., -2.] in ascending order
        // with this operator will result in [-1., -2., 1., 2.,] instead of the
        // expected [-2., -1., 1., 2.,].
        let a_two_comp = a_bits ^ ((a_bits >> 31) & 0x7fff_ffff);
        let b_two_comp = b_bits ^ ((b_bits >> 31) & 0x7fff_ffff);

        a_two_comp > b_two_comp
    }}
}

#[macro_export]
macro_rules! ge {
    ($f32_a: expr, $f32_b: expr) => {{
        // This is a const version of `>=`
        let a_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_a)
        };

        let b_bits = unsafe {
            core::mem::transmute::<f32, i32>($f32_b)
        };

        // Without this transformation sorting [1., 2., -1., -2.] in ascending order
        // with this operator will result in [-1., -2., 1., 2.] instead of the
        // expected [-2., -1., 1., 2.].
        let a_two_comp = a_bits ^ ((a_bits >> 31) & 0x7fff_ffff);
        let b_two_comp = b_bits ^ ((b_bits >> 31) & 0x7fff_ffff);

        a_two_comp >= b_two_comp
    }}
}

#[cfg(test)]
mod tests {
    macro_rules! use_the_macros {
        () => {{
            pos_normal_or_infinity!(1.0)
            && !pos_zero!(-0.0)
            && pos_normal_and_one_or_below!(0.5)
            && neg_normal_and_above_negative_one!(-0.5)
            && lt!(-0.5,  0.5)
            && le!(-0.5, -0.5)
            && gt!( 0.5, -0.5)
            && ge!( 0.5,  0.5)
        }}
    }

    #[test]
    fn the_macros_are_usable_in_consts() {
        const A: bool = use_the_macros!();
        assert!(A)
    }

    /* Floating point operations are currently not permitted in const fn on stable 
    so this doesn't compile.
    #[test]
    fn the_macros_are_usable_in_const_fn() {
        const fn i_am_const_fn() -> bool {
            use_the_macros!()
        }
        
        assert!(i_am_const_fn());
    }
    */

    #[test]
    fn sorting_with_lt_works_as_claimed() {
        use core::cmp::Ordering::*;
        let unsorted = [1., 2., -1., -2.];
        let expected = [-2., -1., 1., 2.];

        let mut sorted = unsorted;
        // We rely on the list having no equal elements.
        sorted.sort_by(|a, b| if lt!(*a, *b) {
            Less
        } else {
            Greater
        });

        assert_eq!(sorted, expected);
    }

    #[test]
    fn sorting_with_le_works_as_claimed() {
        use core::cmp::Ordering::*;
        let unsorted = [1., 2., -1., -2.];
        let expected = [-2., -1., 1., 2.];

        let mut sorted = unsorted;
        // We rely on the list having no equal elements.
        sorted.sort_by(|a, b| if le!(*a, *b) {
            Less
        } else {
            Greater
        });

        assert_eq!(sorted, expected);
    }

    #[test]
    fn sorting_with_gt_works_as_claimed() {
        use core::cmp::Ordering::*;
        let unsorted = [1., 2., -1., -2.];
        let expected = [-2., -1., 1., 2.];

        let mut sorted = unsorted;
        // We rely on the list having no equal elements.
        sorted.sort_by(|a, b| if gt!(*a, *b) {
            Greater
        } else {
            Less
        });

        assert_eq!(sorted, expected);
    }

    #[test]
    fn sorting_with_ge_works_as_claimed() {
        use core::cmp::Ordering::*;
        let unsorted = [1., 2., -1., -2.];
        let expected = [-2., -1., 1., 2.];

        let mut sorted = unsorted;
        // We rely on the list having no equal elements.
        sorted.sort_by(|a, b| if ge!(*a, *b) {
            Greater
        } else {
            Less
        });

        assert_eq!(sorted, expected);
    }
}