// TODO stack string, assuming we realy care about no std
//#![no_std]
#![deny(unused)]

pub trait ClearableStorage<A> {
    fn clear(&mut self);

    fn push(&mut self, a: A);
}

pub type Seed = [u8; 16];

type Xs = [core::num::Wrapping<u32>; 4];

fn xorshift(xs: &mut Xs) -> u32 {
    let mut t = xs[3];

    xs[3] = xs[2];
    xs[2] = xs[1];
    xs[1] = xs[0];

    t ^= t << 11;
    t ^= t >> 8;
    xs[0] = t ^ xs[0] ^ (xs[0] >> 19);

    xs[0].0
}

#[allow(unused)]
fn xs_u32(xs: &mut Xs, min: u32, one_past_max: u32) -> u32 {
    (xorshift(xs) % (one_past_max - min)) + min
}

#[allow(unused)]
fn new_seed(rng: &mut Xs) -> Seed {
    let s0 = xorshift(rng).to_le_bytes();
    let s1 = xorshift(rng).to_le_bytes();
    let s2 = xorshift(rng).to_le_bytes();
    let s3 = xorshift(rng).to_le_bytes();

    [
        s0[0], s0[1], s0[2], s0[3],
        s1[0], s1[1], s1[2], s1[3],
        s2[0], s2[1], s2[2], s2[3],
        s3[0], s3[1], s3[2], s3[3],
    ]
}

fn xs_from_seed(mut seed: Seed) -> Xs {
    // 0 doesn't work as a seed, so use this one instead.
    if seed == [0; 16] {
        seed = 0xBAD_5EED_u128.to_le_bytes();
    }

    macro_rules! wrap {
        ($i0: literal, $i1: literal, $i2: literal, $i3: literal) => {
            core::num::Wrapping(
                u32::from_le_bytes([
                    seed[$i0],
                    seed[$i1],
                    seed[$i2],
                    seed[$i3],
                ])
            )
        }
    }

    [
        wrap!( 0,  1,  2,  3),
        wrap!( 4,  5,  6,  7),
        wrap!( 8,  9, 10, 11),
        wrap!(12, 13, 14, 15),
    ]
}

use floats;

pub use floats::{f32_is, const_assert_valid_bi_unit, const_assert_valid_unit};

pub use floats::bi_unit::{self, X, Y, x, y, x_lt, x_gt, y_lt, y_gt};

pub use floats::unit::{self, W, H, w, h, proportion, Proportion};

pub const COORD_COUNT: tile::Count = tile::Coord::COUNT;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct XY {
    pub x: X,
    pub y: Y,
}

pub mod draw;

pub use draw::{
    DrawLength,
    DrawX,
    DrawY, 
    DrawXY,
    DrawW,
    DrawH,
    DrawWH,
    SpriteKind
};

#[derive(Clone, Copy, Debug)]
enum UiPos {
    Tile(tile::XY),
}

impl UiPos {
    fn xy(&self, sizes: &draw::Sizes) -> DrawXY {
        use UiPos::*;

        match self {
            Tile(txy) => {
                draw::tile_xy_to_draw(sizes, *txy)
            }
        }
    }
}

impl Default for UiPos {
    fn default() -> Self {
        Self::Tile(<_>::default())
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
    use crate::{
        proportion,
        Proportion,
        unit,
        checked::{
            AddOne,
            SubOne,
        },
        Xs,
        xs_u32,
    };

    // An amount of tiles, which are usually arranged in a line.
    pub type Count = u8;

    use core::convert::TryInto;

    macro_rules! tuple_new_type {
        ($struct_name: ident) => {
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

    #[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
    pub struct XY {
        pub x: X,
        pub y: Y,
    }

    impl XY {
        // It would probably be possible to make this a constant with more coord_def
        // macro-trickery, but I'm not sure whether there would be a benefit to 
        // doing so, given that then two `Coord::COUNT * Coord::COUNT` arrays would
        // need to be in the cache at the same time.
        pub fn all() -> impl Iterator<Item = XY> {
            Coord::ALL.iter()
                .flat_map(|&yc|
                    Coord::ALL.iter()
                        .map(move |&xc| (Y(yc), X(xc)))
                )
                .map(|(y, x)| Self {
                    x,
                    y,
                })
        }

        pub(crate) fn all_center_spiralish() -> sprialish::Iter {
            sprialish::Iter::starting_at(XY {
                x: X(Coord::CENTER),
                y: Y(Coord::CENTER),
            })
        }
    }

    mod sprialish {
        use super::*;

        #[derive(Debug)]
        enum NextAction {
            MoveDiagonally,
            MoveUp,
            MoveLeft,
            MoveDown,
            MoveRight,
        }
        impl Default for NextAction {
            fn default() -> Self { Self::MoveDiagonally }
        }

        #[derive(Debug)]
        pub(crate) struct Iter {
            current: Option<XY>,
            bounding_rect: Rect,
            next_action: NextAction,
        }

        impl Iter {
            pub(crate) fn starting_at(start: XY) -> Self {
                Self {
                    current: Some(start),
                    bounding_rect: Rect::min_max(start, start),
                    next_action: <_>::default(),
                }
            }

            pub(crate) fn bounding_rect(&self) -> Rect {
                self.bounding_rect
            }
        }

        impl Iterator for Iter {
            type Item = XY;
    
            fn next(&mut self) -> Option<Self::Item> {
                let last = self.current;
    
                if let Some(current) = last {
                    use NextAction::*;
                    match self.next_action {
                        MoveDiagonally => {
                            match (
                                self.bounding_rect.min.x.checked_sub_one(),
                                self.bounding_rect.min.y.checked_sub_one(),
                                self.bounding_rect.max.x.checked_add_one(),
                                self.bounding_rect.max.y.checked_add_one(),
                            ) {
                                (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) => {
                                    self.bounding_rect = Rect::xyxy(min_x, min_y, max_x, max_y);
                                    self.current = Some(self.bounding_rect.max);
                                },
                                _ => {
                                    self.current = None;
                                }
                            }

                            self.next_action = MoveUp;
                        },
                        MoveUp => {
                            let new = current.y.checked_sub_one()
                                .map(|y| XY {
                                    y,
                                    ..current
                                });

                            if let Some(new) = new {
                                if new.y == self.bounding_rect.min.y {
                                    self.next_action = MoveLeft;
                                }
                            }

                            self.current = new;
                        },
                        MoveLeft => {
                            let new = current.x.checked_sub_one()
                                .map(|x| XY {
                                    x,
                                    ..current
                                });

                            if let Some(new) = new {
                                if new.x == self.bounding_rect.min.x {
                                    self.next_action = MoveDown;
                                }
                            }

                            self.current = new;
                        },
                        MoveDown => {
                            let new = current.y.checked_add_one()
                                .map(|y| XY {
                                    y,
                                    ..current
                                });

                            if let Some(new) = new {
                                if new.y == self.bounding_rect.max.y {
                                    self.next_action = MoveRight;
                                }
                            }

                            self.current = new;
                        },
                        MoveRight => {
                            let new = current.x.checked_add_one()
                                .map(|x| XY {
                                    x,
                                    ..current
                                });

                            if let Some(new) = new {
                                if Some(new.x) == self.bounding_rect.max.x.checked_sub_one() {
                                    self.next_action = MoveDiagonally;
                                }
                            }

                            self.current = new;
                        },
                    }
                }

                last
            }
        }
    }

    #[test]
    fn all_center_spiralish_produces_the_expected_initial_values() {
        const C: usize = Coord::CENTER_INDEX;
        macro_rules! exp {
            ($x: expr, $y: expr) => {
                Some(XY { 
                    x: X(Coord::ALL[$x]),
                    y: Y(Coord::ALL[$y])
                })
            }
        }

        let mut iter = XY::all_center_spiralish();

        const COUNT: usize = 20;
        let actual = {
            let mut actual = [None;COUNT];
            for i in 0..COUNT {
                actual[i] = iter.next();
            }
            actual
        };

        let expected: [Option<_>; COUNT] = [
            exp!(C    , C    ),
            exp!(C + 1, C + 1),
            exp!(C + 1, C    ),
            exp!(C + 1, C - 1),
            exp!(C    , C - 1),
            exp!(C - 1, C - 1),
            exp!(C - 1, C    ),
            exp!(C - 1, C + 1),
            exp!(C    , C + 1),
            exp!(C + 2, C + 2),
            exp!(C + 2, C + 1),
            exp!(C + 2, C    ),
            exp!(C + 2, C - 1),
            exp!(C + 2, C - 2),
            exp!(C + 1, C - 2),
            exp!(C    , C - 2),
            exp!(C - 1, C - 2),
            exp!(C - 2, C - 2),
            exp!(C - 2, C - 1),
            exp!(C - 2, C    ),
        ];

        assert_eq!(actual, expected);
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Rect {
        pub(crate) min: XY,
        pub(crate) max: XY
    }

    macro_rules! rect {
        ($min_x: expr, $min_y: expr, $max_x: expr, $max_y: expr $(,)?) => {
            Rect {
                min: XY {
                    x: $min_x,
                    y: $min_y,
                },
                max: XY {
                    x: $max_x,
                    y: $max_y,
                },
            }
        };
        ($min: expr, $max: expr $(,)?) => {
            Rect {
                min: $min,
                max: $max,
            }
        }
    }

    impl Rect {
        /// This exists because macro importing is complicated
        fn xyxy(min_x: X, min_y: Y, max_x: X, max_y: Y) -> Self {
            rect!(min_x, min_y, max_x, max_y)
        }
        /// This exists because macro importing is complicated
        fn min_max(min: XY, max: XY) -> Self {
            rect!(min, max)
        }
    }

    pub(crate) fn random_xy_in_rect(rng: &mut Xs, rect: Rect) -> XY {
        XY {
            x: X(to_coord_or_default(
                (xs_u32(
                    rng,
                    u8::from(rect.min.x.0) as _,
                    u8::from(rect.max.x.0) as _
                )) as Count
            )),
            y: Y(to_coord_or_default(
                (xs_u32(
                    rng,
                    u8::from(rect.min.y.0) as _,
                    u8::from(rect.max.y.0) as _
                )) as Count
            )),
        }
    }

    pub fn xy_to_i(xy: XY) -> usize {
        u8::from(xy.y.0) as usize * Coord::COUNT as usize
        + u8::from(xy.x.0) as usize
    }

    pub fn i_to_xy(index: usize) -> XY {
        XY {
            x: X(to_coord_or_default(
                (index % Coord::COUNT as usize) as Count
            )),
            y: Y(to_coord_or_default(
                ((index % (TILES_LENGTH as usize) as usize) 
                / Coord::COUNT as usize) as Count
            )),
        }
    }

    #[test]
    fn i_to_xy_then_xy_to_i_is_identity_for_these_examples() {
        macro_rules! is_ident {
            ($index: expr) => {
                assert_eq!($index, xy_to_i(i_to_xy($index)));
            }
        }

        const COUNT: usize = Coord::COUNT as _;

        is_ident!(0);
        is_ident!(COUNT - 1);
        is_ident!(COUNT);
        is_ident!(COUNT + 1);
        is_ident!(2 * COUNT - 1);
        is_ident!(2 * COUNT);
        is_ident!(2 * COUNT + 1);
        is_ident!(COUNT * COUNT - 1);

        // This seems like the best response in cases where we start with an invalid
        // index.
        macro_rules! is_ident_mod_count_squared {
            ($index: expr) => {
                assert_eq!(
                    ($index) % (COUNT * COUNT),
                    xy_to_i(i_to_xy($index)),
                );
            }
        }
        is_ident_mod_count_squared!(COUNT * COUNT);
        is_ident_mod_count_squared!((COUNT * COUNT) + 1);
    }

    #[test]
    fn xy_to_i_then_i_to_xy_is_identity_for_these_examples() {
        macro_rules! is_ident {
            ($index: expr) => {
                assert_eq!($index, i_to_xy(xy_to_i($index)));
            }
        }

        use Coord::*;

        is_ident!(XY {x: X(C0), y: Y(C0)});
        is_ident!(XY {x: X(C49), y: Y(C0)});
        is_ident!(XY {x: X(C0), y: Y(C1)});
        is_ident!(XY {x: X(C1), y: Y(C1)});
        is_ident!(XY {x: X(C49), y: Y(C1)});
        is_ident!(XY {x: X(C0), y: Y(C2)});
        is_ident!(XY {x: X(C1), y: Y(C2)});
        is_ident!(XY {x: X(C49), y: Y(C49)});
    }

    fn to_coord_or_default(n: Count) -> Coord {
        core::convert::TryFrom::try_from(n).unwrap_or_default()
    }

    pub type Distance = u8;
    pub(crate) fn manhattan_distance(a: XY, b: XY) -> Distance {
        ((u8::from(a.x.0) as i8 - u8::from(b.x.0) as i8).abs() 
        + (u8::from(a.y.0) as i8 - u8::from(b.y.0) as i8).abs()) as Distance
    }

    macro_rules! coord_def {
        ($( ($variants: ident => $number: literal) ),+ $(,)?) => {
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            #[repr(u8)]
            /// We only want to handle displaying at most 2 decimal digits for any 
            /// distance from one tile to another. Since we're using Manhattan 
            /// distance, if we keep the value of any coordinate in the range 
            /// [0, 50), then that preseves the desired property.
            pub enum Coord {
                $($variants,)+
            }

            impl Coord {
                pub const COUNT: Count = {
                    let mut count = 0;
                    
                    $(
                        // I think some reference to the vars is needed to use 
                        // the repetitions.
                        let _ = $number;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Coord; Self::COUNT as usize] = [
                    $(Coord::$variants,)+
                ];
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
            proportion!((u8::from(*self) as f32) / (Self::COUNT as f32))
        }

        // Currently there are an even amount of Coords, so there is no true center.
        const CENTER_INDEX: usize = Coord::ALL.len() / 2;

        const CENTER: Coord = Coord::ALL[Self::CENTER_INDEX];
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum HintSpec {
        GoalIsOneUpFrom,
        GoalIsOneDownFrom,
        GoalIsOneLeftFrom,
        GoalIsOneRightFrom,
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Kind {
        Empty,
        Red(Visibility, DistanceIntel),
        RedStar(Visibility),
        Green(Visibility, DistanceIntel),
        GreenStar(Visibility),
        Blue(Visibility, DistanceIntel),
        BlueStar(Visibility),
        Goal(Visibility),
        Hint(Visibility, HintSpec)
    }

    impl Default for Kind {
        fn default() -> Self {
            Self::Empty
        }
    }

    pub(crate) fn get_visibility(kind: Kind) -> Option<Visibility> {
        use Kind::*;
        match kind {
            Empty => None,
            Red(vis, _)
            | RedStar(vis)
            | Green(vis, _)
            | GreenStar(vis)
            | Blue(vis, _)
            | BlueStar(vis)
            | Goal(vis)
            | Hint(vis, _) => Some(vis),
        }
    }

    pub(crate) fn set_visibility(kind: Kind, vis: Visibility) -> Kind {
        use Kind::*;
        match kind {
            Empty => Empty,
            Red(_, intel) => Red(vis, intel),
            RedStar(_) => RedStar(vis),
            Green(_, intel) => Green(vis, intel),
            GreenStar(_) => GreenStar(vis),
            Blue(_, intel) => Blue(vis, intel),
            BlueStar(_) => BlueStar(vis),
            Goal(_) => Goal(vis),
            Hint(_, hint_spec) => Hint(vis, hint_spec),
        }
    }

    pub(crate) fn is_hidden(kind: Kind) -> bool {
        matches!(
            get_visibility(kind),
            Some(Visibility::Hidden)
        )
    }

    pub(crate) fn is_goal(kind: Kind) -> bool {
        matches!(kind, Kind::Goal(_))
    }

    pub(crate) fn kind_description(kind: Kind) -> &'static str {
        use Kind::*;
        match kind {
            Empty => "an empty space",
            Red(..) => "a red tile",
            RedStar(..) => "the red star tile",
            Green(..) => "a green tile",
            GreenStar(..) => "the green star tile",
            Blue(..) => "a blue tile",
            BlueStar(..) => "the blue star tile",
            Goal(..) => "the goal tile",
            Hint(..) => "a hint tile",
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Visibility {
        Hidden,
        Shown
    }

    impl Default for Visibility {
        fn default() -> Self {
            Self::Hidden
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum IntelDigit {
        // One would only ever tell you almost nothing (>1) or exactly how far (=1)
        Two,
        Three,
        Four,
        Five,
        Six,
        Seven,
        Eight,
        Nine,
    }

    impl Default for IntelDigit {
        fn default() -> Self {
            Self::Two
        }
    }

    impl From<IntelDigit> for Distance {
        fn from(digit: IntelDigit) -> Self {
            use IntelDigit::*;
            match digit {
                Two => 2,
                Three => 3,
                Four => 4,
                Five => 5,
                Six => 6,
                Seven => 7,
                Eight => 8,
                Nine => 9,
            }
        }
    }

    impl IntelDigit {
        pub(crate) fn from_rng(rng: &mut Xs) -> Self {
            use IntelDigit::*;
            match xs_u32(rng, 0, 8) {
                0 => Two,
                1 => Three,
                2 => Four,
                3 => Five,
                4 => Six,
                5 => Seven,
                6 => Eight,
                _ => Nine,
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum PrevNext {
        Prev,
        Next
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum DistanceIntel {
        Full,
        PartialAmount(IntelDigit),
        PartialColour(PrevNext),
    }

    impl Default for DistanceIntel {
        fn default() -> Self {
            Self::Full
        }
    }

    impl DistanceIntel {
        pub(crate) fn from_rng(rng: &mut Xs) -> Self {
            use DistanceIntel::*;
            use PrevNext::*;
            match xs_u32(rng, 0, 3) {
                0 => Full,
                1 => PartialAmount(IntelDigit::from_rng(rng)),
                _ => if xs_u32(rng, 0, 2) == 1 {
                    PartialColour(Prev)
                } else {
                    PartialColour(Next)
                },
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Colour {
        Red,
        Green,
        Blue
    }

    pub const TILES_LENGTH: u32 = Coord::COUNT as u32 * Coord::COUNT as u32;
}
pub use tile::TILES_LENGTH;

/// A Tile should always be at a particular position, but that position should be 
/// derivable from the tiles location in the tiles array, so it doesn't need to be
/// stored. But, we often want to get the tile's data and it's location as a single
/// thing. This is why we have both `Tile` and `TileData`
#[derive(Copy, Clone, Debug, Default)]
struct TileData {
    kind: tile::Kind,
}

#[derive(Copy, Clone, Debug, Default)]
struct Tile {
    #[allow(unused)]
    xy: tile::XY,
    data: TileData
}

#[derive(Clone, Debug)]
pub struct Tiles {
    tiles: [TileData; TILES_LENGTH as _],
    red_star_xy: tile::XY,
    green_star_xy: tile::XY,
    blue_star_xy: tile::XY,
    goal_xy: tile::XY,
}

impl Default for Tiles {
    fn default() -> Self {
        Self {
            tiles: [TileData::default(); TILES_LENGTH as _],
            red_star_xy: tile::XY::default(),
            green_star_xy: tile::XY::default(),
            blue_star_xy: tile::XY::default(),
            goal_xy: tile::XY::default(),
        }
    }
}

impl Tiles {
    fn from_rng(rng: &mut Xs, level: Level) -> Self {
        let mut tiles = [TileData::default(); TILES_LENGTH as _];

        use tile::{Kind::*, Visibility::*, HintSpec::*, DistanceIntel::{self, *}};
        use Level::*;

        const SCALE_FACTOR: usize = 512;

        let mut tiles_remaining = match level {
            One => SCALE_FACTOR * 1,
            Two => SCALE_FACTOR * 2,
            Three => SCALE_FACTOR * 3,
        };
        // TODO: get rect from the iter once it's done, and select a uniformly 
        // random tile within that rect in `set_random_tile!`.
        let mut xy_iter = tile::XY::all_center_spiralish();
        for xy in &mut xy_iter {
            let kind = match xs_u32(rng, 0, 15) {
                1|4|7 => Red(Hidden, Full),
                2|5|8 => Green(Hidden, Full),
                3|6|9 => Blue(Hidden, Full),
                10 => Red(Hidden, DistanceIntel::from_rng(rng)),
                11 => Green(Hidden, DistanceIntel::from_rng(rng)),
                12 => Blue(Hidden,  DistanceIntel::from_rng(rng)),
                _ => Empty,
            };

            let i = tile::xy_to_i(xy);

            tiles[i] = TileData {
                kind,
                ..<_>::default()
            };

            tiles_remaining -= 1;
            if tiles_remaining == 0 {
                break
            }
        }

        // We expect this to be the smallest rect that covers the iterated tiles.
        // If we add multiple islands of tiles, then we'll need to pull from each 
        // bounding rect uniformly at random.
        let bounding_rect = xy_iter.bounding_rect();

        macro_rules! set_random_tile {
            ($vis: ident, $($from: pat)|+ => $to: expr) => {{
                let mut selected_xy = None;
                let start_xy = tile::random_xy_in_rect(rng, bounding_rect);
                let mut index = tile::xy_to_i(start_xy);

                for _ in 0..TILES_LENGTH as usize {
                    if let $($from)|+ = tiles[index].kind {
                        tiles[index].kind = $to;
                        selected_xy = Some(tile::i_to_xy(index));
                        break
                    }

                    index += 1;
                    index %= TILES_LENGTH as usize;
                }

                if cfg!(debug_assertions) {
                    selected_xy.expect("set_random_tile found no tile!")
                } else {
                    selected_xy.unwrap_or_default()
                }
            }}
        }

        let red_star_xy = set_random_tile!(vis, Red(vis, _) => RedStar(vis));
        let green_star_xy = set_random_tile!(vis, Green(vis, _) => GreenStar(vis));
        let blue_star_xy = set_random_tile!(vis, Blue(vis, _) => BlueStar(vis));

        let goal_xy = set_random_tile!(
            vis,
            Red(vis, _)|Green(vis, _)|Blue(vis, _) => Goal(vis)
        );
        // TODO Only add these sometimes? Maybe based on difficulty?
        let _ = set_random_tile!(
            vis,
            Red(vis, _)|Green(vis, _)|Blue(vis, _) => Hint(vis, GoalIsOneUpFrom)
        );
        let _ = set_random_tile!(
            vis,
            Red(vis, _)|Green(vis, _)|Blue(vis, _) => Hint(vis, GoalIsOneDownFrom)
        );
        let _ = set_random_tile!(
            vis,
            Red(vis, _)|Green(vis, _)|Blue(vis, _) => Hint(vis, GoalIsOneLeftFrom)
        );
        let _ = set_random_tile!(
            vis,
            Red(vis, _)|Green(vis, _)|Blue(vis, _) => Hint(vis, GoalIsOneRightFrom)
        );

        Self {
            tiles,
            red_star_xy,
            green_star_xy,
            blue_star_xy,
            goal_xy,
        }
    }
}

fn get_tile(tiles: &Tiles, xy: tile::XY) -> Tile {
    Tile {
        xy,
        data: tiles.tiles[tile::xy_to_i(xy)]
    }
}

fn set_tile(tiles: &mut Tiles, tile: Tile) {
    tiles.tiles[tile::xy_to_i(tile.xy)] = tile.data;
}

fn get_star_xy(tiles: &Tiles, colour: tile::Colour) -> tile::XY {
    use tile::Colour::*;
    match colour {
        Red => tiles.red_star_xy,
        Green => tiles.green_star_xy,
        Blue => tiles.blue_star_xy,
    }
}

fn get_goal_xy(tiles: &Tiles) -> tile::XY {
    tiles.goal_xy
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Level {
    One,
    Two,
    Three
}

impl Default for Level {
    fn default() -> Self {
        Self::One
    }
}

#[must_use]
fn next_level(level: Level) -> Level {
    use Level::*;
    match level {
        One => Two,
        Two => Three,
        Three => Three
    }
}

// 64k digs ought to be enough for anybody!
type Digs = u16;

#[derive(Debug, Default)]
struct Board {
    ui_pos: UiPos,
    tiles: Tiles,
    rng: Xs,
    level: Level,
    digs: Digs
}

impl Board {
    fn from_seed(seed: Seed, level: Level, previous_digs: Digs) -> Self {
        let mut rng = xs_from_seed(seed);

        let tiles = Tiles::from_rng(&mut rng, level);

        let mut non_empty_count: Digs = 0;
        for i in 0..tile::TILES_LENGTH as usize {
            if matches!(tiles.tiles[i].kind, tile::Kind::Empty) {
                continue
            }
            non_empty_count += 1;
        }

        const STAR_TYPES_COUNT: Digs = 3;

        let digs = core::cmp::max(
            non_empty_count >> 3,
            // You get at least one mistake per star type
            STAR_TYPES_COUNT * 2
            // for the goal
            + 1
        ) + previous_digs;

        Self {
            rng,
            tiles,
            digs,
            ..<_>::default()
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum InputSpeed {
    Standard,
    Fast,
}

impl Default for InputSpeed {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Tool {
    Selectrum,
    Ruler(tile::XY)
}

impl Default for Tool {
    fn default() -> Self {
        Self::Selectrum
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum ViewMode {
    Clean,
    ShowAllDistances,
    HideRevealed,
}

impl Default for ViewMode {
    fn default() -> Self {
        Self::Clean
    }
}

#[derive(Debug, Default)]
pub struct State {
    sizes: draw::Sizes,
    board: Board,
    input_speed: InputSpeed,
    tool: Tool,
    view_mode: ViewMode,
}

impl State {
    pub fn from_seed(seed: Seed) -> Self {
        Self {
            board: Board::from_seed(seed, <_>::default(), <_>::default()),
            ..<_>::default()
        }
    }
}

pub fn sizes(state: &State) -> draw::Sizes {
    state.sizes.clone()
}

fn is_last_level(state: &State) -> bool {
    next_level(state.board.level) == state.board.level
}

pub type InputFlags = u16;

pub const INPUT_UP_PRESSED: InputFlags              = 0b0000_0000_0000_0001;
pub const INPUT_DOWN_PRESSED: InputFlags            = 0b0000_0000_0000_0010;
pub const INPUT_LEFT_PRESSED: InputFlags            = 0b0000_0000_0000_0100;
pub const INPUT_RIGHT_PRESSED: InputFlags           = 0b0000_0000_0000_1000;

pub const INPUT_UP_DOWN: InputFlags                 = 0b0000_0000_0001_0000;
pub const INPUT_DOWN_DOWN: InputFlags               = 0b0000_0000_0010_0000;
pub const INPUT_LEFT_DOWN: InputFlags               = 0b0000_0000_0100_0000;
pub const INPUT_RIGHT_DOWN: InputFlags              = 0b0000_0000_1000_0000;

pub const INPUT_INTERACT_PRESSED: InputFlags        = 0b0000_0001_0000_0000;
pub const INPUT_FAST_PRESSED: InputFlags            = 0b0000_0010_0000_0000;
pub const INPUT_TOOL_LEFT_PRESSED: InputFlags       = 0b0000_0100_0000_0000;
pub const INPUT_TOOL_RIGHT_PRESSED: InputFlags      = 0b0000_1000_0000_0000;
pub const INPUT_UI_RESET_PRESSED: InputFlags        = 0b0001_0000_0000_0000;
pub const INPUT_VIEW_MODE_LEFT_PRESSED: InputFlags  = 0b0010_0000_0000_0000;
pub const INPUT_VIEW_MODE_RIGHT_PRESSED: InputFlags = 0b0100_0000_0000_0000;

#[derive(Clone, Copy, Debug)]
enum Input {
    NoChange,
    Up,
    Down,
    Left,
    Right,
    Interact,
}

impl Input {
    fn from_flags(flags: InputFlags, input_speed: InputSpeed, tool: Tool) -> Self {
        use Input::*;
        use InputSpeed::*;
        match input_speed {
            Fast => if INPUT_INTERACT_PRESSED & flags != 0 {
                // We disallow Interact during FastMovement if the tool is not
                // undoable, to prevent non-undoable mistakes
                match tool {
                    Tool::Selectrum => NoChange,
                    Tool::Ruler(_) => Interact
                }
            } else if INPUT_UP_DOWN & flags != 0 {
                Up
            } else if INPUT_DOWN_DOWN & flags != 0 {
                Down
            } else if INPUT_LEFT_DOWN & flags != 0 {
                Left
            } else if INPUT_RIGHT_DOWN & flags != 0 {
                Right
            } else {
                NoChange
            },
            Standard => if INPUT_INTERACT_PRESSED & flags != 0 {
                Interact
            } else if INPUT_UP_PRESSED & flags != 0 {
                Up
            } else if INPUT_DOWN_PRESSED & flags != 0 {
                Down
            } else if INPUT_LEFT_PRESSED & flags != 0 {
                Left
            } else if INPUT_RIGHT_PRESSED & flags != 0 {
                Right
            } else {
                NoChange
            },
        }
    }
}

pub fn update(
    state: &mut State,
    commands: &mut dyn ClearableStorage<draw::Command>,
    input_flags: InputFlags,
    draw_wh: DrawWH,
) {
    use Input::*;
    use UiPos::*;
    use draw::{SpriteSpec, TextSpec, TextKind, Command::*};
    use Tool::*;
    use ViewMode::*;

    if draw_wh != state.sizes.draw_wh {
        state.sizes = draw::fresh_sizes(draw_wh);
    }

    commands.clear();

    if INPUT_FAST_PRESSED & input_flags != 0 {
        state.input_speed = if let InputSpeed::Fast = state.input_speed {
            InputSpeed::Standard
        } else {
            InputSpeed::Fast
        };
    } 

    if INPUT_TOOL_LEFT_PRESSED & input_flags != 0 {
        state.tool = match state.tool {
            Selectrum => match state.board.ui_pos {
                Tile(xy) => Ruler(xy)
            },
            Ruler(_) => Selectrum,
        };
    }

    if INPUT_TOOL_RIGHT_PRESSED & input_flags != 0 {
        state.tool = match state.tool {
            Selectrum => match state.board.ui_pos {
                Tile(xy) => Ruler(xy)
            },
            Ruler(_) => Selectrum,
        };
    }

    if INPUT_VIEW_MODE_LEFT_PRESSED & input_flags != 0 {
        state.view_mode = match state.view_mode {
            Clean => HideRevealed,
            ShowAllDistances => Clean,
            HideRevealed => ShowAllDistances,
        };
    }

    if INPUT_VIEW_MODE_RIGHT_PRESSED & input_flags != 0 {
        state.view_mode = match state.view_mode {
            Clean => ShowAllDistances,
            ShowAllDistances => HideRevealed,
            HideRevealed => Clean,
        };
    }

    macro_rules! do_ui_reset {
        () => {
            state.input_speed = <_>::default();
            state.tool = <_>::default();
            state.view_mode = <_>::default();
        }
    }

    if INPUT_UI_RESET_PRESSED & input_flags != 0 {
        do_ui_reset!();
    }

    let input = Input::from_flags(input_flags, state.input_speed, state.tool);

    let mut interacted = false;

    match (input, &mut state.board.ui_pos) {
        (NoChange, _) => {},
        (Up, Tile(ref mut xy)) => {
            if let Some(new_y) = xy.y.checked_sub_one() {
                xy.y = new_y;
            }
        },
        (Down, Tile(ref mut xy)) => {
            if let Some(new_y) = xy.y.checked_add_one() {
                xy.y = new_y;
            }
        },
        (Left, Tile(ref mut xy)) => {
            if let Some(new_x) = xy.x.checked_sub_one() {
                xy.x = new_x;
            }
        },
        (Right, Tile(ref mut xy)) => {
            if let Some(new_x) = xy.x.checked_add_one() {
                xy.x = new_x;
            }
        },
        (Interact, Tile(ref xy)) => {
            use Tool::*;
            match state.tool {
                Selectrum => {
                    let mut tile = get_tile(&state.board.tiles, *xy);

                    let started_hidden = tile::is_hidden(tile.data.kind);

                    if started_hidden &&
                        (state.board.digs > 0 || cfg!(debug_assertions)) {
                        state.board.digs = state.board.digs.saturating_sub(1);
                        tile.data.kind = tile::set_visibility(
                            tile.data.kind,
                            tile::Visibility::Shown
                        );

                        set_tile(&mut state.board.tiles, tile);

                        macro_rules! reveal_all_matching {
                            ($variant: ident) => {{
                                // TODO particles that say "+1", "Extra dig", "1UP"
                                // or something.
                                state.board.digs = state.board.digs.saturating_add(1);

                                for index in 0..TILES_LENGTH as usize {
                                    if let $variant(Hidden, intel) = state.board.tiles.tiles[index].kind {
                                        state.board.tiles.tiles[index].kind = $variant(Shown, intel);
                                    }
                                }
                            }}
                        }

                        match tile.data.kind {
                            RedStar(_) => reveal_all_matching!(Red),
                            GreenStar(_) => reveal_all_matching!(Green),
                            BlueStar(_) => reveal_all_matching!(Blue),
                            _ => {}
                        }
                    }
        
                    use tile::{Kind::*, Visibility::*};
        
                    if !started_hidden && tile::is_goal(tile.data.kind) {
                        if is_last_level(state) {
                            for xy in tile::XY::all() {
                                set_tile(&mut state.board.tiles, crate::Tile {
                                    xy,
                                    data: TileData {
                                        kind: Goal(Shown),
                                        ..<_>::default()
                                    }
                                });
                            }
                        } else {
                            let level = state.board.level;
                            state.board = Board::from_seed(
                                new_seed(&mut state.board.rng),
                                state.board.level,
                                state.board.digs,
                            );
                            state.board.level = next_level(level);
                        }
                    }
                    
                    interacted = true;
                },
                Ruler(ref mut pos) => {
                    if *pos == *xy {
                        // We find that we want to be able to press `Interact` after
                        // measuring with the ruler, and have a dig happen. But we
                        // don't want to make mistakes easy to make, so we make it
                        // require multiple presses to do that.
                        do_ui_reset!();
                    } else {
                        *pos = *xy;
                    }
                }
            }
        },
    }

    let goal_sprite = if is_last_level(state) {
        SpriteKind::TerminalGoal
    } else {
        SpriteKind::InstrumentalGoal
    };

    for txy in tile::XY::all() {
        let tiles = &state.board.tiles;
        let tile = get_tile(tiles, txy);

        let xy = draw::tile_xy_to_draw(&state.sizes, txy);

        use tile::{Kind::*, Visibility::*, DistanceIntel::*, PrevNext::*};

        let sprite = match tile.data.kind {
            Empty => continue,
            Red(Hidden, _)
            | RedStar(Hidden)
            | Green(Hidden, _)
            | GreenStar(Hidden)
            | Blue(Hidden, _)
            | BlueStar(Hidden)
            | Goal(Hidden)
            | Hint(Hidden, _) => SpriteKind::Hidden,
            Red(Shown, PartialColour(Prev)) => SpriteKind::BlueRed,
            Red(Shown, PartialColour(Next)) => SpriteKind::RedGreen,
            Red(Shown, _) => SpriteKind::Red,
            RedStar(Shown) => SpriteKind::RedStar,
            Green(Shown, PartialColour(Prev)) => SpriteKind::RedGreen,
            Green(Shown, PartialColour(Next)) => SpriteKind::GreenBlue,
            Green(Shown, _) => SpriteKind::Green,
            GreenStar(Shown) => SpriteKind::GreenStar,
            Blue(Shown, PartialColour(Prev)) => SpriteKind::GreenBlue,
            Blue(Shown, PartialColour(Next)) => SpriteKind::BlueRed,
            Blue(Shown, _) => SpriteKind::Blue,
            BlueStar(Shown) => SpriteKind::BlueStar,
            Goal(Shown) => goal_sprite,
            Hint(Shown, _) => SpriteKind::Hint,
        };

        let draw_tile = match state.view_mode {
            Clean | ShowAllDistances => true,
            HideRevealed => matches!(sprite, SpriteKind::Hidden),
        };

        if draw_tile {
            commands.push(Sprite(SpriteSpec{
                sprite,
                xy
            }));
        }

        if matches!(state.view_mode, ShowAllDistances|Clean) {
            let distance_info = match tile.data.kind {
                Red(Shown, intel) => Some((
                    get_star_xy(tiles, tile::Colour::Red),
                    intel,
                )),
                Green(Shown, intel) => Some((
                    get_star_xy(tiles, tile::Colour::Green),
                    intel,
                )),
                Blue(Shown, intel) => Some((
                    get_star_xy(tiles, tile::Colour::Blue),
                    intel,
                )),
                _ => None,
            };

            if let Some((star_xy, intel)) = distance_info {
                let should_draw_distance = if matches!(state.view_mode, Clean) {
                    tile::is_hidden(get_tile(tiles, star_xy).data.kind)
                } else {
                    // We already checked for HideRevealed above
                    true
                };

                if should_draw_distance {
                    use tile::{Distance, DistanceIntel::*};

                    let distance = tile::manhattan_distance(txy, star_xy);

                    // We could technically avoid this allocation since there 
                    // are only finitely many needed strings here.
                    let text = match intel {
                        PartialAmount(digit) => {
                            let digit_distance = Distance::from(digit);
                            if distance == digit_distance {
                                format!("{}{}", distance / 10, distance % 10)
                            } else if distance > digit_distance {
                                format!(">{}", digit_distance)
                            } else /* distance < digit_distance */{
                                format!("<{}", digit_distance)
                            }
                        },
                        _ => format!("{}{}", distance / 10, distance % 10),
                    };

                    commands.push(Text(TextSpec {
                        text,
                        xy,
                        wh: DrawWH {
                            w: state.sizes.tile_side_length,
                            h: state.sizes.tile_side_length,
                        },
                        kind: TextKind::DistanceMarker,
                    }));
                }
            }
        }
    }

    match state.tool {
        Selectrum => {},
        Ruler(pos) => {
            commands.push(Sprite(SpriteSpec{
                sprite: SpriteKind::RulerEnd,
                xy: draw::tile_xy_to_draw(&state.sizes, pos),
            }));
        }
    }

    if !interacted {
        commands.push(Sprite(SpriteSpec{
            sprite: SpriteKind::Selectrum,
            xy: state.board.ui_pos.xy(&state.sizes),
        }));
    }

    const HINT_TILES_PER_ROW: usize = 3;
    const HINT_TILES_PER_COLUMN: usize = 3;
    const HINT_TILES_COUNT: usize = HINT_TILES_PER_ROW * HINT_TILES_PER_COLUMN;

    let hint_info = {
        let tiles = &state.board.tiles;
        let hint_spec = match state.board.ui_pos {
            Tile(txy) => {
                let tile = get_tile(tiles, txy);

                use tile::{Kind::*, Visibility::*};
                match tile.data.kind {
                    Hint(Shown, hint_spec) => Some(hint_spec),
                    _ => None
                }
            }
        };

        if let Some(hint_spec) = hint_spec {
            use tile::HintSpec::*;

            let goal_xy = get_goal_xy(tiles);
            let (direction, target_xy) = match hint_spec {
                GoalIsOneUpFrom => (
                    "up",
                    goal_xy.y.checked_add_one().map(|y| tile::XY {
                        y,
                        ..goal_xy
                    })
                ),
                GoalIsOneDownFrom => (
                    "down",
                    goal_xy.y.checked_sub_one().map(|y| tile::XY {
                        y,
                        ..goal_xy
                    })
                ),
                GoalIsOneLeftFrom => (
                    "left",
                    goal_xy.x.checked_add_one().map(|x| tile::XY {
                        x,
                        ..goal_xy
                    })
                ),
                GoalIsOneRightFrom => (
                    "right",
                    goal_xy.x.checked_sub_one().map(|x| tile::XY {
                        x,
                        ..goal_xy
                    })
                ),
            };

            let description = if let Some(target_xy) = target_xy {
                tile::kind_description(get_tile(tiles, target_xy).data.kind)
            } else {
                "the edge of the grid"
            };

            let hint_string = format!(
                "The goal is one tile {} from {}.",
                direction,
                description
            );

            
            let mut hint_sprites = [
                Some(SpriteKind::QuestionMark);
                HINT_TILES_COUNT
            ];

            const CENTER_INDEX: usize = HINT_TILES_COUNT / 2;

            hint_sprites[CENTER_INDEX] = Some(goal_sprite);

            const UP_INDEX: usize = CENTER_INDEX - HINT_TILES_PER_ROW;
            const DOWN_INDEX: usize = CENTER_INDEX + HINT_TILES_PER_ROW;
            const LEFT_INDEX: usize = CENTER_INDEX - 1;
            const RIGHT_INDEX: usize = CENTER_INDEX + 1;

            macro_rules! target_sprite {
                (edge => $edge: ident) => {
                    if let Some(target_xy) = target_xy {
                        use tile::Kind::*;
                        match get_tile(tiles, target_xy).data.kind {
                            Empty => None,
                            Red(_, _) => Some(SpriteKind::Red),
                            RedStar(_) => Some(SpriteKind::RedStar),
                            Green(_, _) => Some(SpriteKind::Green),
                            GreenStar(_) => Some(SpriteKind::GreenStar),
                            Blue(_, _) => Some(SpriteKind::Blue),
                            BlueStar(_) => Some(SpriteKind::BlueStar),
                            Goal(_) => Some(goal_sprite),
                            Hint(_, _) => Some(SpriteKind::Hint),
                        }
                    } else {
                        Some(SpriteKind::$edge)
                    };
                }
            }

            match hint_spec {
                GoalIsOneUpFrom => {
                    hint_sprites[DOWN_INDEX] = target_sprite!(edge => EdgeUp);
                },
                GoalIsOneDownFrom => {
                    hint_sprites[UP_INDEX] = target_sprite!(edge => EdgeDown);
                },
                GoalIsOneLeftFrom => {
                    hint_sprites[RIGHT_INDEX] = target_sprite!(edge => EdgeLeft);
                },
                GoalIsOneRightFrom => {
                    hint_sprites[LEFT_INDEX] = target_sprite!(edge => EdgeRight);
                },
            };

            Some((
                hint_string,
                hint_sprites,
            ))
        } else {
            None
        }
    };

    let board_xywh = &state.sizes.board_xywh;
    let left_text_x = state.sizes.play_xywh.x + MARGIN;
    let right_text_x = board_xywh.x + board_xywh.w + MARGIN;

    const MARGIN: f32 = 16.;

    let small_section_h = state.sizes.draw_wh.h / 8. - MARGIN;
    let large_section_h = small_section_h * 3.;
    {
        let mut y = MARGIN;

        commands.push(Text(TextSpec{
            text: format!(
                "Level\n{:#?}",
                state.board.level
            ),
            xy: DrawXY { x: left_text_x, y },
            wh: DrawWH {
                w: board_xywh.x - left_text_x,
                h: small_section_h
            },
            kind: TextKind::Level,
        }));

        y += small_section_h;

        commands.push(Text(TextSpec{
            text: format!(
                "Digs\n{}",
                state.board.digs,
            ),
            xy: DrawXY { x: left_text_x, y },
            wh: DrawWH {
                w: board_xywh.x - left_text_x,
                h: small_section_h
            },
            kind: TextKind::Digs,
        }));

        y += small_section_h;

        // TODO handle the case where there is less than enough room for 
        // tiles at the regular size, better.
        if let Some((hint_string, hint_sprites)) = hint_info {
            commands.push(Text(TextSpec{
                text: hint_string,
                xy: DrawXY { x: left_text_x, y },
                wh: DrawWH {
                    w: board_xywh.x - left_text_x,
                    h: large_section_h
                },
                kind: TextKind::HintString,
            }));

            y += large_section_h;

            let left_hint_tile_x = left_text_x + state.sizes.tile_side_length;

            for column in 0..HINT_TILES_PER_COLUMN {
                y += state.sizes.tile_side_length;

                for row in 0..HINT_TILES_PER_ROW {
                    if let Some(sprite) = hint_sprites[row + HINT_TILES_PER_ROW * column] {
                        commands.push(Sprite(SpriteSpec{
                            sprite,
                            xy: DrawXY {
                                x: left_hint_tile_x + row as DrawX * state.sizes.tile_side_length,
                                y
                            },
                        }));
                    }
                }
            }
        }
    }

    {
        let text = match state.view_mode {
            Clean => None,
            ShowAllDistances => Some("Show All Distances"),
            HideRevealed => Some("Hide Revealed"),
        };

        if let Some(text) = text {
            let y = MARGIN;
            commands.push(Text(TextSpec{
                text: text.to_owned(),
                xy: DrawXY { x: right_text_x, y },
                wh: DrawWH {
                    w: board_xywh.x - left_text_x,
                    h: small_section_h
                },
                kind: TextKind::Fast,
            }));
        }
    }

    {
        match state.tool {
            Selectrum => {},
            Ruler(pos) => {
                let y = (state.sizes.draw_wh.h) / 2.;
                commands.push(Text(TextSpec{
                    text: format!("Ruler: {}", match state.board.ui_pos {
                        Tile(xy) => {
                            tile::manhattan_distance(pos, xy)
                        }
                    }),
                    xy: DrawXY { x: right_text_x, y },
                    wh: DrawWH {
                        w: board_xywh.x - left_text_x,
                        h: small_section_h,
                    },
                    kind: TextKind::Ruler,
                }));
            }
        }
    }

    if let InputSpeed::Fast = state.input_speed {
        let y = state.sizes.draw_wh.h * (MARGIN - 1.) / MARGIN;
        commands.push(Text(TextSpec{
            text: "Fast".to_owned(),
            xy: DrawXY { x: right_text_x, y },
            wh: DrawWH {
                w: board_xywh.x - left_text_x,
                h: small_section_h
            },
            kind: TextKind::Fast,
        }));
    }
}
