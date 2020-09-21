#![no_std]
#![deny(unused)]


#[derive(Clone, Copy)]
pub enum Input {
    Up,
    Down,
    Left,
    Right,
    Interact,
}

pub enum Command {
    Sprite(Sprite),
}

pub struct SpriteSpec {
    sprite: SpriteKind,
    x: X,
    y: Y,
}

mod bi_unit {
    ///! `bi_unit` is short for bilateral-unit. AKA the range [-1.0, 1.0]
    ///! This range has the following advantages:
    ///! * uses half of the available precision of a floating point number as 
    ///!   opposed to say [0.0, 1.0] whch only uses a quarter of it.
    ///! * +0.0, (the natural default) is in the middle of the range (subjective)
    
    macro_rules! tuple_new_type {
        (struct $struct_name: ident, macro_rules! $macro_name) => {
            #[derive(Clone, Copy, Debug, Default)]
            pub struct $struct_name(F32);
        
            impl From<$struct_name> for f32 {
                fn from(thing: $struct_name) -> Self {
                    Self::from(thing.0)
                }
            }

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
                $crate::is_pos_zero!($float)
                || $crate::is_pos_normal_and_one_or_below!($float)
                || $crate::is_neg_normal_and_above_negative_one!($float)
             } as usize] = [];
        }
    }

    #[macro_export]
    macro_rules! is_pos_zero {
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
    macro_rules! is_pos_normal_and_one_or_below {
        ($f32: expr) => {{
            let f: f32 = $f32;

            f >= f32::MIN_POSITIVE && f <= 1.0
        }}
    }

    #[macro_export]
    macro_rules! is_neg_normal_and_above_negative_one {
        ($f32: expr) => {{
            let f: f32 = $f32;

            f >= -1.0 && f <= -f32::MIN_POSITIVE
        }}
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
                if is_pos_zero!(f) {
                    f
                } else if is_pos_normal_and_one_or_below!(f) {
                    f
                } else if is_neg_normal_and_above_negative_one!(f) {
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
}
pub use bi_unit::{X, Y};

mod unit {
    ///! `unit` in this case refers to the range [0.0, 1.0]

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
                const fn from_f32(f: f32) -> Self {
                    if 
                }
            }

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

    #[derive(Clone, Copy, Debug, Default)]
    struct F32(f32);

    impl From<F32> for f32 {
        fn from(f: F32) -> Self {
            f.0
        }
    }
}
pub use unit::{W, H};

#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub x: X,
    pub y: Y,
}

impl Point {
    const fn minimum(a: Self, b: Self) -> Self {
        if a.x < b.x && a.x < b.y {
            a
        } else {
            b
        }
    }

    const fn maximum(a: Self, b: Self) -> Self {
        if a.x > b.x && a.x > b.y {
            a
        } else {
            b
        }
    }
}

/// A min/max Rect. This way of defining a rectangle has nicer behaviour when 
/// clamping the rectangle within a rectangular area, than say an x,y,w,h version.
/// The fields aren't public so we can maintain the min/max relationship internally.
pub struct Rect {
    min: Point,
    max: Point,
}

impl Rect {
    pub const fn new_xyxy(x1: X, y1: Y, x2: X, y2: Y) -> Self {
        Self::new(Point{x: x1, y: y1}, Point{x: x2, y: y2})
    }

    pub const fn new(a: Point, b: Point) -> Self {
        Self {
            min: Point::minimum(a, b),
            max: Point::maximum(a, b),
        }
    }

    pub const fn min(&self) -> Point {
        self.min
    }

    pub const fn max(&self) -> Point {
        self.max
    }

    pub const fn wh(&self) -> (W, H) {
        (
            self.max.x - self.min.x,
            self.max.y - self.min.y,
        )
    }
}

macro_rules! rect_xyxy {
    (
        $min_x: literal,
        $min_y: literal,
        $max_x: literal,
        $max_y: literal $(,)?
    ) => {
        Rect::new_xyxy(
            bi_unit::x!($min_x),
            bi_unit::y!($min_y),
            bi_unit::x!($max_x),
            bi_unit::y!($max_y),
        )
    }
}

#[test]
fn wh_gives_expected_results_on_these_rects() {
    let w0_h0 = rect_xyxy!();

    assert_eq!(w0_h0 .wh(), (unit::w!(1.0), unit::h!(2.0)));

    let w1_h2 = rect_xyxy!(
        -0.5,
        -1.0,
        0.5,
        1.0,
    );

    assert_eq!(w1_h2.wh(), (unit::w!(1.0), unit::h!(2.0)));

    let w2_h1 = rect_xyxy!(
        -1.0,
        -0.5,
        1.0,
        0.5,
    );

    assert_eq!(w2_h1.wh(), (unit::w!(2.0), unit::h!(1.0)));

    let w_half_h_quarter = rect_xyxy!(
        -0.5,
        0.0,
        0.0,
        0.25,
    );

    assert_eq!(w_half_h_quarter.wh(), (unit::w!(0.5), unit::h!(0.25)));
}

const TILES_RECT: Rect = rect_xyxy!(
    -0.5,
    -0.5,
    0.5,
    0.5,
);

pub enum SpriteKind {
    Blank,
    Red,
    Green,
    Blue,
    Selectrum,
}

#[derive(Debug, Default)]
pub struct State {
    ui_pos: UIPos,
}

#[derive(Clone, Copy, Debug)]
enum UIPos {
    Tile(tile::X, tile::Y),
}

impl UIPos {
    const fn xy(&self) -> (X, Y) {
        use UIPos::*;

        match self {
            Tile(ref tx, ref ty) => {
                let (w, h) = TILES_RECT.wh();
                let min = TILES_RECT.min();

                (
                    min.x + w * tx.proportion(),
                    min.y + h * ty.proportion(),
                )
            }
        }
    }
}

impl Default for UIPos {
    fn default() -> Self {
        Self::Tile(<_>::default(), <_>::default())
    }
}

pub fn update(state: &mut State, commands: &mut Vec<Command>, input: Input) {
    use Input::*;
    use UIPos::*;

    commands.clear();

    match (input, &mut state.ui_pos) {
        (Up, Tile(_, ref mut y)) => {
            if let Some(new_y) = y.checked_sub_one() {
                *y = new_y;
            }
        },
        (Down, Tile(_, ref mut y)) => {
            if let Some(new_y) = y.checked_add_one() {
                *y = new_y;
            }
        },
        (Left, Tile(ref mut x, _)) => {
            if let Some(new_x) = x.checked_sub_one() {
                *x = new_x;
            }
        },
        (Right, Tile(ref mut x, _)) => {
            if let Some(new_x) = x.checked_add_one() {
                *x = new_x;
            }
        },
        (Interact, _) => {
            
        },
    }
    
    let (x, y) = state.ui_pos.xy();

    commands.push(Sprite(SpriteSpec{
        sprite: SpriteKind::Selectrum,
        x,
        y,
    }));
}

mod checked {
    pub trait AddOne: Sized {
        fn checked_add_one(&self) -> Option<Self>;
    }

    pub trait SubOne: Sized {
        fn checked_sub_one(&self) -> Option<Self>;
    }
}
use checked::{AddOne, SubOne};

mod tile {
    use crate::{
        checked::{
            AddOne,
            SubOne,
        },
        unit,
    };

    use core::convert::TryInto;

    macro_rules! tuple_new_type {
        ($struct_name: ident) => {
            #[derive(Clone, Copy, Debug, Default)]
            pub struct $struct_name(Coord);
        
            impl AddOne for $struct_name {
                fn checked_add_one(&self) -> Option<Self> {
                    self.0.checked_add_one().map($struct_name)
                }
            }
        
            impl SubOne for $struct_name {
                fn checked_sub_one(&self) -> Option<Self> {
                    self.0.checked_sub_one().map($struct_name)
                }
            }

            impl $struct_name {
                pub fn proportion(&self) -> uint::F32 {
                    self.0.proportion()
                }
            }
        }
    }

    tuple_new_type!{X}
    tuple_new_type!{Y}

    macro_rules! coord_def {
        ($( ($variants: ident => $number: literal) ),+ $(,)?) => {
            #[derive(Clone, Copy, Debug)]
            #[repr(u8)]
            /// We only want to handle displaying at most 2 decimal digits for any 
            /// distance from one tile to another. Since we're using Manhattan 
            /// distance, if we keep the value of any coordinate in the range 
            /// [0, 50), then that preseves the desired property.
            enum Coord {
                $($variants,)+
            }

            impl Coord {
                const COUNT: u8 = {
                    let mut count = 0;
                    
                    $(
                        // I think some reference to the vars is needed to use 
                        // the repetitions.
                        let _ = $number;

                        count += 1;
                    )+

                    count
                };
            }

            impl From<Coord> for u8 {
                fn from(coord: Coord) -> u8 {
                    match coord {
                        $(Coord::$variants => $number,)+
                    }
                }
            }

            impl core::convert::TryFrom<u8> for Coord {
                type Error = ();

                fn try_from(byte :u8) -> Result<Self, Self::Error> {
                    match byte {
                        $($number => Ok(Coord::$variants),)+
                        _ => Err(()),
                    }
                }
            }
        }
    }

    coord_def!{
        (C0 => 0),
        (C1 => 1),
        (C2 => 2),
        (C3 => 3),
        (C4 => 4),
        (C5 => 5),
        (C6 => 6),
        (C7 => 7),
        (C8 => 8),
        (C9 => 9),
        (C10 => 10),
        (C11 => 11),
        (C12 => 12),
        (C13 => 13),
        (C14 => 14),
        (C15 => 15),
        (C16 => 16),
        (C17 => 17),
        (C18 => 18),
        (C19 => 19),
        (C20 => 20),
        (C21 => 21),
        (C22 => 22),
        (C23 => 23),
        (C24 => 24),
        (C25 => 25),
        (C26 => 26),
        (C27 => 27),
        (C28 => 28),
        (C29 => 29),
        (C30 => 30),
        (C31 => 31),
        (C32 => 32),
        (C33 => 33),
        (C34 => 34),
        (C35 => 35),
        (C36 => 36),
        (C37 => 37),
        (C38 => 38),
        (C39 => 39),
        (C40 => 40),
        (C41 => 41),
        (C42 => 42),
        (C43 => 43),
        (C44 => 44),
        (C45 => 45),
        (C46 => 46),
        (C47 => 47),
        (C48 => 48),
        (C49 => 49),
    }

    impl Default for Coord {
        fn default() -> Self {
            Self::C0
        }
    }

    impl AddOne for Coord {
        fn checked_add_one(&self) -> Option<Self> {
            (*self as u8).checked_add(1)
                .and_then(|byte| byte.try_into().ok())
        }
    }

    impl SubOne for Coord {
        fn checked_sub_one(&self) -> Option<Self> {
            (*self as u8).checked_sub(1)
                .and_then(|byte| byte.try_into().ok())
        }
    }

    impl Coord {
        fn proportion(&self) -> uint::F32 {
            (u8::from(*self) as f32) / (Self::COUNT as f32)
        }
    }
}
