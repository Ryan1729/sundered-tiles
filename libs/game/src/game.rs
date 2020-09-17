#![no_std]

#[derive(Debug, Default)]
pub struct State {
    ui_pos: UIPos,
}

#[derive(Clone, Copy)]
pub enum Input {
    Up,
    Down,
    Left,
    Right,
    Interact,
}

#[derive(Clone, Copy, Debug)]
enum UIPos {
    Tile(tile::X, tile::Y),
}

impl Default for UIPos {
    fn default() -> Self {
        Self::Tile(<_>::default(), <_>::default())
    }
}

pub fn update(state: &mut State, input: Input) {
    use Input::*;
    use UIPos::*;

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
    use crate::checked::{
        AddOne,
        SubOne
    };

    use core::convert::TryInto;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct X(Coord);

    impl AddOne for X {
        fn checked_add_one(&self) -> Option<Self> {
            self.0.checked_add_one().map(X)
        }
    }

    impl SubOne for X {
        fn checked_sub_one(&self) -> Option<Self> {
            self.0.checked_sub_one().map(X)
        }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Y(Coord);

    impl AddOne for Y {
        fn checked_add_one(&self) -> Option<Self> {
            self.0.checked_add_one().map(Y)
        }
    }

    impl SubOne for Y {
        fn checked_sub_one(&self) -> Option<Self> {
            self.0.checked_sub_one().map(Y)
        }
    }

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

            impl core::convert::TryInto<Coord> for u8 {
                type Error = ();

                fn try_into(self) -> Result<Coord, Self::Error> {
                    match self {
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
}
