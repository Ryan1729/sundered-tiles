#![no_std]
#![deny(unused)]

pub trait ClearableStorage<A> {
    fn clear(&mut self);

    fn push(&mut self, a: A);
}

#[derive(Clone, Copy)]
pub enum Input {
    Up,
    Down,
    Left,
    Right,
    Interact,
}

pub enum Command {
    Sprite(SpriteSpec),
}

pub struct SpriteSpec {
    sprite: SpriteKind,
    x: X,
    y: Y,
}

pub use bi_unit::{X, Y, x, y};

pub use unit::{W, H, w, h};

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
            unit::w!(f32::from(self.max.x) - f32::from(self.min.x)),
            unit::h!(f32::from(self.max.y) - f32::from(self.min.y)),
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

pub fn update(
    state: &mut State,
    commands: &mut dyn ClearableStorage<Command>,
    input: Input
) {
    use Input::*;
    use UIPos::*;
    use Command::*;

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
                pub fn proportion(&self) -> unit::Proportion {
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
        fn proportion(&self) -> unit::Proportion {
            (u8::from(*self) as f32) / (Self::COUNT as f32)
        }
    }
}
