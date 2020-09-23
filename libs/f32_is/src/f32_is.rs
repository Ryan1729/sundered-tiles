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

#[cfg(test)]
mod tests {
    macro_rules! use_the_macros {
        () => {{
            pos_normal_or_infinity!(1.0)
            && !pos_zero!(-0.0)
            && pos_normal_and_one_or_below!(0.5)
            && neg_normal_and_above_negative_one!(-0.5)
        }}
    }

    #[test]
    fn the_macros_are_usable_in_consts() {
        assert!(use_the_macros!())
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
}