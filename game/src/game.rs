// TODO stack string, assuming we realy care about no std
//#![no_std]
#![deny(unused)]

macro_rules! compile_time_assert {
    ($assertion: expr) => {{
        #[allow(unknown_lints, eq_op)]
        // Based on the const_assert macro from static_assertions;
        const _: [(); 0 - !{$assertion} as usize] = [];
    }}
}

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

fn xs_shuffle<A>(rng: &mut Xs, slice: &mut [A]) {
    for i in 1..slice.len() as u32 {
        // This only shuffles the first u32::MAX_VALUE - 1 elements.
        let r = xs_u32(rng, 0, i + 1) as usize;
        let i = i as usize;
        slice.swap(i, r);
    }
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

mod char_utils {
    /// A char from a base 10 digit: that is, a u8 in the range [0, 9]. The result 
    /// for input outside that range will be some other char.
    pub fn from_digit(digit: u8) -> char {
        debug_assert!(digit <= 9);
        ('0' as u8 + digit) as char
    }
}

mod tile {
    use crate::{
        char_utils,
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
            #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
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

                pub(crate) const ZERO: $struct_name = $struct_name(Coord::ZERO);
                #[allow(unused)] // Desired in the tests
                pub(crate) const CENTER: $struct_name = $struct_name(Coord::CENTER);
                pub(crate) const MAX: $struct_name = $struct_name(Coord::MAX);
                
                pub(crate) const COUNT: Count = Coord::COUNT;

                #[allow(unused)]
                pub (crate) const ALL: [$struct_name; Self::COUNT as usize] = {
                    let mut all = [$struct_name(Coord::ALL[0]); Self::COUNT as usize];

                    let mut coord = Coord::ALL[0];
                    while let Some(c) = coord.const_checked_add_one() {
                        all[Coord::const_to_count(c) as usize] = $struct_name(c);

                        coord = c;
                    }

                    all
                };

                #[allow(unused)] // desired in tests
                pub fn from_rng(rng: &mut Xs) -> Self {
                    $struct_name(Coord::from_rng(rng))
                }
            }

            impl From<$struct_name> for usize {
                fn from(thing: $struct_name) -> Self {
                    Self::from(thing.0)
                }
            }
        }
    }

    tuple_new_type!{X}
    tuple_new_type!{Y}

    macro_rules! wrapping_add_delta_def {
        ($coord_name: ident, $delta_name: ident) => {
            impl $coord_name {
                fn wrapping_add_delta(&self, delta: $delta_name) -> Self {
                    let result = core::convert::TryFrom::try_from(
                        (Count::from(self.0) + Count::from(delta)) % Coord::COUNT
                    );

                    debug_assert!(result.is_ok(), "{:?}", result);

                    $coord_name(result.unwrap_or_default())
                }
            }
        }
    }
    wrapping_add_delta_def!{X, WrappingDeltaX}
    wrapping_add_delta_def!{Y, WrappingDeltaY}

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

        pub(crate) fn all_center_spiralish() -> Box<dyn sprialish::IterWithBounds> {
            Box::new(sprialish::OutIter::starting_at(XY {
                x: X(Coord::CENTER),
                y: Y(Coord::CENTER),
            }))
        }

        #[allow(unused)]
        pub(crate) fn upper_left_quadrant_spiralish() -> Box<dyn sprialish::IterWithBounds> {
            Box::new(sprialish::OutIter::starting_at(XY {
                x: X(Coord::QUARTER),
                y: Y(Coord::QUARTER),
            }))
        }

        #[allow(unused)]
        pub(crate) fn upper_right_quadrant_spiralish() -> Box<dyn sprialish::IterWithBounds> {
            Box::new(sprialish::OutIter::starting_at(XY {
                x: X(Coord::THREE_QUARTERS),
                y: Y(Coord::QUARTER),
            }))
        }

        #[allow(unused)]
        pub(crate) fn lower_left_quadrant_spiralish() -> Box<dyn sprialish::IterWithBounds> {
            Box::new(sprialish::OutIter::starting_at(XY {
                x: X(Coord::QUARTER),
                y: Y(Coord::THREE_QUARTERS),
            }))
        }

        #[allow(unused)]
        pub(crate) fn lower_right_quadrant_spiralish() -> Box<dyn sprialish::IterWithBounds> {
            Box::new(sprialish::OutIter::starting_at(XY {
                x: X(Coord::THREE_QUARTERS),
                y: Y(Coord::THREE_QUARTERS),
            }))
        }

        pub(crate) fn upper_left_spiral_in() -> Box<dyn sprialish::IterWithBounds> {
            Box::new(sprialish::InIter::starting_at(XY {
                x: X(Coord::ZERO),
                y: Y(Coord::ZERO),
            }))
        }

        const ZERO: XY = XY { x: X::ZERO, y: Y::ZERO };
        const MAX: XY = XY { x: X::MAX, y: Y::MAX };

        #[allow(unused)] // desired in tests
        pub fn from_rng(rng: &mut Xs) -> Self {
            Self {
                x: X::from_rng(rng),
                y: Y::from_rng(rng),
            }
        }
    }

    pub type LabelChars = [char; 2];

    #[derive(Copy, Clone, Debug)]
    pub enum LabelKind {
        Amount,
        Modulus(DistanceModulus),
    }

    impl Default for LabelKind {
        fn default() -> Self {
            DEFAULT_LABEL.kind
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct Label {
        pub chars: LabelChars,
        pub kind: LabelKind,
    }

    const DEFAULT_LABEL_CHARS: LabelChars = ['?', '?'];
    const DEFAULT_LABEL: Label = Label {
        chars: DEFAULT_LABEL_CHARS,
        kind: LabelKind::Amount,
    };

    impl Label {
        pub fn chars_string(&self) -> String {
            self.chars.iter().filter(|&c| *c != '\0').collect()
        }
    }

    pub(crate) fn label_from_amount_or_default(amount: u8) -> Label {
        match (amount / 10, amount % 10) {
            (tens, ones) if tens <= 9 && ones <= 9 => Label {
                chars: [
                    char_utils::from_digit(tens),
                    char_utils::from_digit(ones),
                ],
                kind: LabelKind::Amount,
            },
            _ => DEFAULT_LABEL,
        }
    }

    pub(crate) fn distance_label(intel: DistanceIntel, distance: Distance) -> Label {
        use DistanceIntel::*;
    
        let mut kind = LabelKind::Amount;
    
        // We could technically avoid this allocation since there 
        // are only finitely many needed strings here.
        let chars = match intel {
            Full => return label_from_amount_or_default(distance),
            PartialAmount(digit) => {
                let digit_distance = Distance::from(digit);
                if distance == digit_distance {
                    return label_from_amount_or_default(distance)
                } else if distance > digit_distance {
                    ['>', char_utils::from_digit(digit_distance)]
                } else /* distance < digit_distance */{
                    ['<', char_utils::from_digit(digit_distance)]
                }
            },
            NModM(modulus) => {
                let modded = distance % (modulus as Distance);
    
                kind = LabelKind::Modulus(modulus);
    
                // The null char is not meant to be used.
                // TODO tighter types?
                [char_utils::from_digit(modded), '\0']
            },
            PrimeOrNot => {
                // If we decide to add a font that supports it, this
                // could be "element of blackboard bold P" instead.
                if PRIMES_BELOW_100.contains(&distance) {
                    [' ', 'p']
                } else {
                    ['!', 'p']
                }.to_owned()
            }
        };
    
        Label {
            chars,
            kind,
        }
    }

    mod sprialish {
        use super::*;

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

        pub(crate) trait IterWithBounds: Iterator<Item = XY> {
            // TODO should be bounding_rects, and return multiple
            fn bounding_rect(&self) -> Rect;
        }

        #[derive(Debug)]
        pub(crate) struct OutIter {
            current: Option<XY>,
            bounding_rect: Rect,
            next_action: NextAction,
        }

        impl OutIter {
            pub(crate) fn starting_at(start: XY) -> Self {
                Self {
                    current: Some(start),
                    bounding_rect: Rect::min_max(start, start),
                    next_action: <_>::default(),
                }
            }
        }

        impl IterWithBounds for OutIter {
            fn bounding_rect(&self) -> Rect {
                self.bounding_rect
            }
        }

        impl Iterator for OutIter {
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

        #[derive(Debug)]
        pub(crate) struct InIter {
            current: Option<XY>,
            shrinking_rect: Rect,
            next_action: NextAction,
            initial_action: NextAction,
        }

        impl InIter {
            pub(crate) fn starting_at(start: XY) -> Self {
                let mut next_action = NextAction::MoveLeft;
                if start.x == X::ZERO {
                    next_action = NextAction::MoveUp;
                }

                if start.y == Y::ZERO {
                    next_action = NextAction::MoveRight;
                }

                if start.x == X::MAX {
                    next_action = NextAction::MoveDown;
                }

                if start.y == Y::MAX {
                    next_action = NextAction::MoveLeft;
                }

                Self {
                    current: Some(start),
                    shrinking_rect: Rect::min_max(
                        XY::ZERO,
                        XY::MAX
                    ),
                    next_action,
                    initial_action: next_action,
                }
            }
        }

        impl IterWithBounds for InIter {
            fn bounding_rect(&self) -> Rect {
                self.shrinking_rect
            }
        }

        impl Iterator for InIter {
            type Item = XY;
    
            fn next(&mut self) -> Option<Self::Item> {
                let last = self.current;
    
                if let Some(current) = last {
                    use NextAction::*;

                    macro_rules! shrink_if_needed {
                        () => {
                            if self.initial_action == self.next_action {
                                match (
                                    self.shrinking_rect.min.x.checked_add_one(),
                                    self.shrinking_rect.min.y.checked_add_one(),
                                    self.shrinking_rect.max.x.checked_sub_one(),
                                    self.shrinking_rect.max.y.checked_sub_one(),
                                ) {
                                    (Some(min_x), Some(min_y), Some(max_x), Some(max_y)) 
                                    if min_x != max_x && min_y != max_y => {
                                        self.shrinking_rect = Rect::xyxy(min_x, min_y, max_x, max_y);
                                    },
                                    _ => {
                                        self.current = None;
                                    }
                                }
                            }
                        }
                    }
                    match self.next_action {
                        MoveDiagonally => {
                            // TODO get rid of this case.
                            self.current = None;
                        },
                        MoveUp => {
                            let new = current.y.checked_sub_one()
                                .map(|y| XY {
                                    y,
                                    ..current
                                });

                            if let Some(new) = new {
                                if new.y == self.shrinking_rect.min.y {
                                    self.next_action = MoveRight;
                                    shrink_if_needed!();
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
                                if new.x == self.shrinking_rect.min.x {
                                    self.next_action = MoveUp;
                                    shrink_if_needed!();
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
                                if new.y == self.shrinking_rect.max.y {
                                    self.next_action = MoveLeft;
                                    shrink_if_needed!();
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
                                if new.x == self.shrinking_rect.max.x {
                                    self.next_action = MoveDown;
                                    shrink_if_needed!();
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

    #[test]
    fn upper_left_spiral_in_produces_the_expected_initial_values() {
        macro_rules! exp {
            ($x: expr, $y: expr) => {
                Some(XY { 
                    x: X(Coord::ALL[$x]),
                    y: Y(Coord::ALL[$y])
                })
            }
        }

        let mut iter = XY::upper_left_spiral_in();

        // This named constant just makes it more obvious what the 1 is for.
        const AT_CORNER: usize = 1;
        // These we might actually change later.
        const PRE_CORNER: usize = 2;
        const POST_CORNER: usize = 2;
        const CORNER_COUNT: usize = PRE_CORNER + AT_CORNER + POST_CORNER;

        const FIRST_CORNER_COUNT: usize = AT_CORNER + POST_CORNER;
        let mut first_corner = [None; FIRST_CORNER_COUNT];

        for i in 0..FIRST_CORNER_COUNT {
            first_corner[i] = iter.next();
        }

        assert_eq!(
            first_corner,
            [    
                exp!(0, 0),
                exp!(1, 0),
                exp!(2, 0),
            ]
        );

        for _ in 0..Coord::COUNT as usize - FIRST_CORNER_COUNT - (PRE_CORNER + AT_CORNER) {
            iter.next();
        }

        let mut second_corner = [None; CORNER_COUNT];

        for i in 0..CORNER_COUNT {
            second_corner[i] = iter.next();
        }

        const MAX_INDEX: usize = Coord::MAX_INDEX as _;

        assert_eq!(
            second_corner,
            [    
                exp!(MAX_INDEX - 2, 0),
                exp!(MAX_INDEX - 1, 0),
                exp!(MAX_INDEX    , 0),
                exp!(MAX_INDEX    , 1),
                exp!(MAX_INDEX    , 2),
            ]
        );
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
        assert_ne!(u8::from(rect.max.x.0), 0);
        assert_ne!(u8::from(rect.max.y.0), 0);
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
        xy_to_i_usize((usize::from(xy.x.0), usize::from(xy.y.0)))
    }

    pub fn xy_to_i_usize((x, y): (usize, usize)) -> usize {
        y * Coord::COUNT as usize + x
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

    /// We assume that Distance can only be at most 99, since that is part of
    // `Coord`'s design. If this ever becomes a problem we can enforce this.
    pub type Distance = u8;
    pub(crate) fn manhattan_distance(a: XY, b: XY) -> Distance {
        manhattan_distance_count_tuples(
            (Count::from(a.x.0), Count::from(a.y.0)),
            (Count::from(b.x.0), Count::from(b.y.0)),
        )
    }

    pub(crate) fn manhattan_distance_count_tuples(
        a: (Count, Count),
        b: (Count, Count)
    ) -> Distance {
        ((a.0 as i8 - b.0 as i8).abs() 
        + (a.1 as i8 - b.1 as i8).abs()) as Distance
    }

    pub const PRIMES_BELOW_100: [Distance; 25] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        73, 79, 83, 89, 97
    ];

    #[derive(Clone, Copy, Debug)]
    pub(crate) struct WrappingDeltaXY {
        x: WrappingDeltaX,
        y: WrappingDeltaY,
    }

    impl WrappingDeltaXY {
        pub fn from_rng(rng: &mut Xs) -> Self {
            Self {
                x: WrappingDeltaX::from_rng(rng),
                y: WrappingDeltaY::from_rng(rng),
            }
        }
    }

    pub(crate) fn apply_wrap_around_delta(
        delta: WrappingDeltaXY,
        xy: XY
    ) -> XY {
        XY {
            x: xy.x.wrapping_add_delta(delta.x),
            y: xy.y.wrapping_add_delta(delta.y),
        }
    }

    macro_rules! coord_def {
        (
            ($zero_variant: ident => $zero_number: literal),
            $( ($wrap_variants: ident => $wrap_number: literal) ),+ $(,)?
        ) => {
            #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
            #[repr(u8)]
            /// We only want to handle displaying at most 2 decimal digits for any 
            /// distance from one tile to another. Since we're using Manhattan 
            /// distance, if we keep the value of any coordinate in the range 
            /// [0, 50), then that preseves the desired property.
            pub enum Coord {
                $zero_variant,
                $($wrap_variants,)+
            }

            impl Coord {
                pub const COUNT: Count = {
                    let mut count = 0;
                    
                    count += 1; // $zero_number
                    $(
                        // Some reference to the vars is needed to use 
                        // the repetitions.
                        let _ = $wrap_number;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Coord; Self::COUNT as usize] = [
                    Coord::$zero_variant,
                    $(Coord::$wrap_variants,)+
                ];

                pub const MAX_INDEX: Count = Self::COUNT - 1;

                #[allow(unused)] // desired in tests
                pub fn from_rng(rng: &mut Xs) -> Self {
                    Self::ALL[xs_u32(rng, 0, Self::ALL.len() as u32) as usize]
                }
            }

            impl Default for Coord {
                fn default() -> Self {
                    Self::$zero_variant
                }
            }

            impl From<Coord> for u8 {
                fn from(coord: Coord) -> u8 {
                    coord.const_to_count()
                }
            }

            impl Coord {
                const fn const_to_count(self) -> Count {
                    match self {
                        Coord::$zero_variant => $zero_number,
                        $(Coord::$wrap_variants => $wrap_number,)+
                    }
                }
            }

            impl From<Coord> for usize {
                fn from(coord: Coord) -> Self {
                    Self::from(u8::from(coord))
                }
            }

            impl core::convert::TryFrom<u8> for Coord {
                type Error = ();

                fn try_from(byte: u8) -> Result<Self, Self::Error> {
                    Self::const_try_from(byte)
                }
            }

            impl Coord {
                const fn const_try_from(byte: u8) -> Result<Self, ()> {
                    match byte {
                        $zero_number => Ok(Coord::$zero_variant),
                        $($wrap_number => Ok(Coord::$wrap_variants),)+
                        Self::COUNT..=u8::MAX => Err(()),
                    }
                }
            }

            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            #[repr(u8)]
            pub enum WrappingDeltaX {
                $($wrap_variants,)+
            }

            impl From<WrappingDeltaX> for u8 {
                fn from(dx: WrappingDeltaX) -> u8 {
                    match dx {
                        $(WrappingDeltaX::$wrap_variants => $wrap_number,)+
                    }
                }
            }

            impl WrappingDeltaX {
                pub const COUNT: Count = {
                    let mut count = 0;
                    
                    $(
                        // Some reference to the vars is needed to use 
                        // the repetitions.
                        let _ = $wrap_number;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Self; Self::COUNT as usize] = [
                    $(Self::$wrap_variants,)+
                ];

                pub fn from_rng(rng: &mut Xs) -> Self {
                    Self::ALL[xs_u32(rng, 0, Self::ALL.len() as u32) as usize]
                }
            }

            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            #[repr(u8)]
            pub enum WrappingDeltaY {
                $($wrap_variants,)+
            }

            impl From<WrappingDeltaY> for u8 {
                fn from(dy: WrappingDeltaY) -> u8 {
                    match dy {
                        $(WrappingDeltaY::$wrap_variants => $wrap_number,)+
                    }
                }
            }

            impl WrappingDeltaY {
                pub const COUNT: Count = {
                    let mut count = 0;
                    
                    $(
                        // Some reference to the vars is needed to use 
                        // the repetitions.
                        let _ = $wrap_number;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Self; Self::COUNT as usize] = [
                    $(Self::$wrap_variants,)+
                ];

                pub fn from_rng(rng: &mut Xs) -> Self {
                    Self::ALL[xs_u32(rng, 0, Self::ALL.len() as u32) as usize]
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

    impl AddOne for Coord {
        fn checked_add_one(&self) -> Option<Self> {
            self.const_checked_add_one()
        }
    }

    impl Coord {
        const fn const_checked_add_one(&self) -> Option<Self> {
            match (*self as u8).checked_add(1) {
                Some(byte) => match Self::const_try_from(byte) {
                    Ok(x) => Some(x),
                    Err(_) => None,
                },
                None => None,
            }
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

        const QUARTER_INDEX: usize = Self::CENTER_INDEX / 2;

        const QUARTER: Coord = Coord::ALL[Self::QUARTER_INDEX];

        const THREE_QUARTERS_INDEX: usize = Self::CENTER_INDEX + Self::QUARTER_INDEX;

        const THREE_QUARTERS: Coord = Coord::ALL[Self::THREE_QUARTERS_INDEX];

        const ZERO: Coord = Coord::ALL[0];
        const MAX: Coord = Coord::ALL[Coord::ALL.len() - 1];
    }

    macro_rules! relative_delta_def {
        ($( $variants: ident ),+ $(,)?) => {
            #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
            pub(crate) enum RelativeDelta {
                $($variants,)+
            }

            impl RelativeDelta {
                pub const COUNT: usize = {
                    let mut count = 0;

                    $(
                        // I think some reference to the vars is needed to use
                        // the repetitions.
                        let _ = Self::$variants;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Self; Self::COUNT] = [
                    $(Self::$variants,)+
                ];
            }

            impl Default for RelativeDelta {
                fn default() -> Self {
                    Self::ALL[0]
                }
            }
        }
    }

    relative_delta_def!{
        // Ordered left to right top to bottom, so that the descriptions can easily
        // be put into that order. We want the descriptions to be the opposite order
        // but this order is nicer for coding, IMO, so we currently reverse the 
        // ordering for the descriptions.
        TwoUpTwoLeft,
        TwoUpOneLeft,
        TwoUp,
        TwoUpOneRight,
        TwoUpTwoRight,
        OneUpTwoLeft,
        OneUpOneLeft,
        OneUp,
        OneUpOneRight,
        OneUpTwoRight,
        TwoLeft,
        OneLeft,
        OneRight,
        TwoRight,
        OneDownTwoLeft,
        OneDownOneLeft,
        OneDown,
        OneDownOneRight,
        OneDownTwoRight,
        TwoDownTwoLeft,
        TwoDownOneLeft,
        TwoDown,
        TwoDownOneRight,
        TwoDownTwoRight,
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum BetweenSpec {
        /// The minumum number of tiles matching the visual kind that must be
        /// traversed between this tile and the target tile.
        Minimum(WrappingDeltaXY, VisualKind),
        // TODO
        // Maximum(VisualKind),
        // Less sure about these ones. They seem too computationally intensive 
        // for the player.
        // Mean(VisualKind),
        // Median(VisualKind),
        // Mode(VisualKind),
    }

    /// By default, `GoalIsNot` hints are much less powerful than GoalIs hints. 
    /// This can be demonstrated by counting how many possibilties each eliminates.
    /// As of this writing, while we include multiple `GoalIsNot` hints per tile
    /// when we set one, we do not include enough extra ones to match the power of 
    /// a `GoalIs` hint. We do this for two reasons:
    /// * At this point, we think this might be better for game balance
    /// * It seems like it would be more mentally taxing to integrate more 
    ///   `GoalIsNot` hints. So it might be best if the player is encouraged to do 
    ///   this less.
    pub(crate) const HINTS_PER_GOAL_IS_NOT: usize = 3;
    // The last index in an HINTS_PER_GOAL_IS_NOT + 1 array.
    // AKA (HINTS_PER_GOAL_IS_NOT + 1) - 1;
    pub(crate) const EXTRA_OFFSET_INDEX: usize = HINTS_PER_GOAL_IS_NOT;

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum HintSpec {
        GoalIs(RelativeDelta),
        GoalIsNot(
            [RelativeDelta; HINTS_PER_GOAL_IS_NOT],
            // Plus one as a backup offset in case the current offset is bad. For
            // example if we offset from an off-the-edge tile to another 
            // off-the-edge tile.
            [NonZeroHintTileIndex; HINTS_PER_GOAL_IS_NOT + 1]
        ),
    }

    macro_rules! hybrid_offset_def {
        ($($variants: ident),+ $(,)?) => {
            /// `Red(HybridOffset::Zero, ..)` is plain `Red`, Red(HybridOffset::One, ..)
            /// is rendered as RedGreen, but is fundamentally a Red tile.
            /// `Zero` means not a hybrid.
            #[derive(Clone, Copy, Debug)]
            pub(crate) enum HybridOffset {
                $($variants),+
            }

            impl HybridOffset {
                pub const COUNT: Count = {
                    let mut count = 0;
                    
                    $(
                        // I think some reference to the vars is needed to use 
                        // the repetitions.
                        let _ = HybridOffset::$variants;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Self; Self::COUNT as usize] = [
                    $(Self::$variants,)+
                ];

                pub fn from_rng(rng: &mut Xs) -> Self {
                    Self::ALL[xs_u32(rng, 0, Self::ALL.len() as u32) as usize]
                }

                pub(crate) const DEFAULT: HybridOffset = Self::ALL[0];
            }
        
            impl Default for HybridOffset {
                fn default() -> Self {
                    Self::DEFAULT
                }
            }
        }
    }

    hybrid_offset_def! {
        Zero,
        One,
        Two,
        Three,
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Kind {
        Empty,
        Red(HybridOffset, Visibility, DistanceIntel),
        RedStar(Visibility),
        Green(HybridOffset, Visibility, DistanceIntel),
        GreenStar(Visibility),
        Blue(HybridOffset, Visibility, DistanceIntel),
        BlueStar(Visibility),
        Goal(Visibility),
        Hint(Visibility, HintSpec),
        GoalDistance(HybridOffset, Visibility, DistanceIntel),
        Between(Visibility, BetweenSpec),
    }

    impl Default for Kind {
        fn default() -> Self {
            Self::Empty
        }
    }

    /// Should always be lower than HintTile::COUNT.
    pub type HintTileIndex = usize;
    /// Should always be lower than HintTile::COUNT.
    pub type NonZeroHintTileIndex = core::num::NonZeroUsize;

    macro_rules! hint_tile_def {
        (
            VisualKind { $( $kind_variants: ident ),+ $(,)? }
            WentOff{ $( $went_off_variants: ident ),+ $(,)? }
        ) => {
            #[derive(Copy, Clone, Debug, PartialEq, Eq)]
            pub(crate) enum VisualKind {
                $( $kind_variants,)*
            }

            impl VisualKind {
                pub const COUNT: usize = {
                    let mut count = 0;

                    $(
                        // I think some reference to the vars is needed to use
                        // the repetitions.
                        let _ = Self::$kind_variants;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Self; Self::COUNT] = [
                    $(Self::$kind_variants,)+
                ];

                #[allow(unused)] // desired in tests
                pub fn from_rng(rng: &mut Xs) -> Self {
                    Self::ALL[xs_u32(rng, 0, Self::ALL.len() as u32) as usize]
                }
            }

            #[derive(Copy, Clone, Debug)]
            pub(crate) enum WentOff {
                $($went_off_variants,)+
            }

            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub(crate) enum HintTile {
                $( $kind_variants,)*
                $( $went_off_variants,)*
            }

            impl HintTile {
                pub const COUNT: usize = {
                    let mut count = 0;

                    $(
                        // I think some reference to the vars is needed to use
                        // the repetitions.
                        let _ = Self::$kind_variants;

                        count += 1;
                    )+

                    $(
                        // Otherwise how would this tell the difference?
                        let _ = Self::$went_off_variants;

                        count += 1;
                    )+

                    count
                };

                pub const ALL: [Self; Self::COUNT] = [
                    $(Self::$kind_variants,)+
                    $(Self::$went_off_variants,)+
                ];

                // SAFTEY: The `+` instead of `*` in the macro definition should prevent this 
                // from being zero.
                pub const NON_ZERO_MAX_INDEX: NonZeroHintTileIndex = unsafe {
                    NonZeroHintTileIndex::new_unchecked(Self::COUNT - 1)
                };
            }

            impl From<VisualKind> for HintTile {
                fn from(visual_kind: VisualKind) -> Self {
                    match visual_kind {
                        $(VisualKind::$kind_variants => Self::$kind_variants,)+
                    }
                }
            }

        }
    }

    hint_tile_def!{
        VisualKind {
            Empty,
            Red,
            RedStar,
            RedGreen,
            Green,
            GreenStar,
            GreenBlue,
            Blue,
            BlueStar,
            BlueRed,
            Goal,
            Hint,
            GoalDistance,
            RedGoal,
            GreenGoal,
            BlueGoal,
            Between,
        }
        WentOff {
            UpAndLeftEdges,
            UpEdge,
            UpAndRightEdges,
            LeftEdge,
            RightEdge,
            DownAndLeftEdges,
            DownEdge,
            DownAndRightEdges,
        }
    }

    impl From<Kind> for VisualKind {
        fn from(kind: Kind) -> Self {
            use VisualKind::*;
            use HybridOffset::*;
            match kind {
                Kind::Empty => Empty,
                Kind::Red(Zero, _, _) => Red,
                Kind::Red(One, _, _) => RedGreen,
                Kind::Red(Two, _, _) => BlueRed,
                Kind::Red(Three, _, _) => RedGoal,
                Kind::RedStar(_) => RedStar,
                Kind::Green(Zero, _, _) => Green,
                Kind::Green(One, _, _) => GreenBlue,
                Kind::Green(Two, _, _) => GreenGoal,
                Kind::Green(Three, _, _) => RedGreen,
                Kind::GreenStar(_) => GreenStar,
                Kind::Blue(Zero, _, _) => Blue,
                Kind::Blue(One, _, _) => BlueGoal,
                Kind::Blue(Two, _, _) => BlueRed,
                Kind::Blue(Three, _, _) => GreenBlue,
                Kind::BlueStar(_) => BlueStar,
                Kind::Goal(_) => Goal,
                Kind::Hint(_, _) => Hint,
                Kind::GoalDistance(Zero, _, _) => GoalDistance,
                Kind::GoalDistance(One, _, _) => RedGoal,
                Kind::GoalDistance(Two, _, _) => GreenGoal,
                Kind::GoalDistance(Three, _, _) => BlueGoal,
                Kind::Between(_, _) => Between,
            }
        }
    }

    pub(crate) fn hint_tile_from_kind(kind: Kind) -> HintTile {
        use HintTile::*;
        use HybridOffset::*;
        match kind {
            Kind::Empty => Empty,
            Kind::Red(Zero, _, _) => Red,
            Kind::Red(One, _, _) => RedGreen,
            Kind::Red(Two, _, _) => BlueRed,
            Kind::Red(Three, _, _) => RedGoal,
            Kind::RedStar(_) => RedStar,
            Kind::Green(Zero, _, _) => Green,
            Kind::Green(One, _, _) => GreenBlue,
            Kind::Green(Two, _, _) => GreenGoal,
            Kind::Green(Three, _, _) => RedGreen,
            Kind::GreenStar(_) => GreenStar,
            Kind::Blue(Zero, _, _) => Blue,
            Kind::Blue(One, _, _) => BlueGoal,
            Kind::Blue(Two, _, _) => BlueRed,
            Kind::Blue(Three, _, _) => GreenBlue,
            Kind::BlueStar(_) => BlueStar,
            Kind::Goal(_) => Goal,
            Kind::Hint(_, _) => Hint,
            Kind::GoalDistance(Zero, _, _) => GoalDistance,
            Kind::GoalDistance(One, _, _) => RedGoal,
            Kind::GoalDistance(Two, _, _) => GreenGoal,
            Kind::GoalDistance(Three, _, _) => BlueGoal,
            Kind::Between(_, _) => Between,
        }
    }

    pub(crate) fn hint_tile_from_went_off(went_off: WentOff) -> HintTile {
        use HintTile::*;
        match went_off {
            WentOff::UpAndLeftEdges => UpAndLeftEdges,
            WentOff::UpEdge => UpEdge,
            WentOff::UpAndRightEdges => UpAndRightEdges,
            WentOff::LeftEdge => LeftEdge,
            WentOff::RightEdge => RightEdge,
            WentOff::DownAndLeftEdges => DownAndLeftEdges,
            WentOff::DownEdge => DownEdge,
            WentOff::DownAndRightEdges => DownAndRightEdges,
        }
    }

    pub(crate) fn hint_tile_from_kind_went_off_result(
        result: Result<Kind, WentOff>
    ) -> HintTile {
        match result {
            Ok(kind) => hint_tile_from_kind(kind),
            Err(went_off) => hint_tile_from_went_off(went_off),
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    pub(crate) enum Category {
        OffTheEdge,
        Misc,
    }

    pub(crate) fn hint_tile_category(hint_tile: HintTile) -> Category {
        use HintTile::*;
        match hint_tile {
            UpAndLeftEdges
            | UpEdge
            | UpAndRightEdges
            | LeftEdge
            | RightEdge
            | DownAndLeftEdges
            | DownEdge
            | DownAndRightEdges => Category::OffTheEdge,
            Empty 
            | Red
            | RedStar
            | RedGreen
            | Green
            | GreenStar
            | GreenBlue
            | Blue
            | BlueStar
            | BlueRed
            | Goal
            | Hint
            | GoalDistance
            | RedGoal
            | GreenGoal
            | BlueGoal
            | Between => Category::Misc,
        }
    }

    pub(crate) fn get_visibility(kind: Kind) -> Option<Visibility> {
        use Kind::*;
        match kind {
            Empty => None,
            Red(_, vis, _)
            | RedStar(vis)
            | Green(_, vis, _)
            | GreenStar(vis)
            | Blue(_, vis, _)
            | BlueStar(vis)
            | Goal(vis)
            | Hint(vis, _)
            | GoalDistance(_, vis, _)
            | Between(vis, _) => Some(vis),
        }
    }

    pub(crate) fn set_visibility(kind: Kind, vis: Visibility) -> Kind {
        use Kind::*;
        match kind {
            Empty => Empty,
            Red(offset, _, intel) => Red(offset, vis, intel),
            RedStar(_) => RedStar(vis),
            Green(offset, _, intel) => Green(offset, vis, intel),
            GreenStar(_) => GreenStar(vis),
            Blue(offset, _, intel) => Blue(offset, vis, intel),
            BlueStar(_) => BlueStar(vis),
            Goal(_) => Goal(vis),
            Hint(_, hint_spec) => Hint(vis, hint_spec),
            GoalDistance(offset, _, intel) => GoalDistance(offset, vis, intel),
            Between(_, between_spec) => Between(vis, between_spec),
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
        hint_tile_description(hint_tile_from_kind(kind))
    }

    pub(crate) fn hint_tile_description(hint_tile: HintTile) -> &'static str {
        use HintTile::*;
        match hint_tile {
            Empty => "an empty space",
            RedGreen => "a red/green tile",
            Red => "a red tile",
            RedStar => "the red star tile",
            GreenBlue => "a green/blue tile",
            Green => "a green tile",
            GreenStar => "the green star tile",
            BlueRed => "a blue/red tile",
            Blue => "a blue tile",
            BlueStar => "the blue star tile",
            Goal => "the goal tile",
            Hint => "a hint tile",
            GoalDistance => "a goal distance hint tile",
            RedGoal => "a red/goal distance hint tile",
            GreenGoal => "a green/goal distance hint tile",
            BlueGoal => "a blue/goal distance hint tile",
            // TODO distinct descriptions for these?
            UpAndLeftEdges
            | UpEdge
            | UpAndRightEdges
            | LeftEdge
            | RightEdge
            | DownAndLeftEdges
            | DownEdge
            | DownAndRightEdges => "the edge of the grid",
            Between => "a between hint tile",
        }
    }

    pub(crate) fn hint_tile_adjective(hint_tile: HintTile) -> &'static str {
        use HintTile::*;
        match hint_tile {
            Empty => "empty space",
            RedGreen => "red/green",
            Red => "red",
            RedStar => "red star",
            GreenBlue => "green/blue",
            Green => "green",
            GreenStar => "green star",
            BlueRed => "blue/red",
            Blue => "blue",
            BlueStar => "blue star",
            Goal => "goal",
            Hint => "hint",
            GoalDistance => "goal distance hint",
            RedGoal => "red/goal distance hint",
            GreenGoal => "green/goal distance hint",
            BlueGoal => "blue/goal distance hint",
            // TODO distinct descriptions for these?
            UpAndLeftEdges
            | UpEdge
            | UpAndRightEdges
            | LeftEdge
            | RightEdge
            | DownAndLeftEdges
            | DownEdge
            | DownAndRightEdges => "off of the grid",
            Between => "between hint",
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Visibility {
        Hidden,
        Shown
    }

    impl Visibility {
        pub(crate) const DEFAULT: Visibility = Self::Hidden;
    }

    impl Default for Visibility {
        fn default() -> Self {
            Self::DEFAULT
        }
    }

    macro_rules! intel_digit_def {
        ($( ($variants: ident => $number: literal) ),+ $(,)?) => {
            #[derive(Clone, Copy, Debug)]
            pub(crate) enum IntelDigit {
                $($variants),+
            }

            impl IntelDigit {
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

                pub const ALL: [Self; Self::COUNT as usize] = [
                    $(Self::$variants,)+
                ];
            }

            impl From<IntelDigit> for Distance {
                fn from(digit: IntelDigit) -> Self {
                    match digit {
                        $(IntelDigit::$variants => $number,)+
                    }
                }
            }

            impl IntelDigit {
                pub fn from_rng(rng: &mut Xs) -> Self {
                    Self::ALL[xs_u32(rng, 0, Self::ALL.len() as u32) as usize]
                }
            }

            impl Default for IntelDigit {
                fn default() -> Self {
                    Self::ALL[0]
                }
            }
        }
    }

    intel_digit_def!{
        // One would only ever tell you almost nothing (>1) or exactly how far (=1)
        (Two => 2),
        (Three => 3),
        (Four => 4),
        (Five => 5),
        (Six => 6),
        (Seven => 7),
        (Eight => 8),
        (Nine => 9),
    }

    pub(crate) type DistanceModulus = u8;

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum DistanceIntel {
        Full,
        PartialAmount(IntelDigit),
        NModM(DistanceModulus),
        PrimeOrNot,
    }

    impl Default for DistanceIntel {
        fn default() -> Self {
            Self::DEFAULT
        }
    }

    impl DistanceIntel {
        pub(crate) const DEFAULT: DistanceIntel = Self::Full;

        pub(crate) fn from_rng(rng: &mut Xs) -> Self {
            use DistanceIntel::*;
            match xs_u32(rng, 0, 4) {
                0 => Full,
                1 => PartialAmount(IntelDigit::from_rng(rng)),
                2 => NModM(xs_u32(rng, 2, 5) as DistanceModulus),
                _ => PrimeOrNot
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

type TileDataArray = [TileData; TILES_LENGTH as _];

#[derive(Clone, Debug)]
pub struct Tiles {
    tiles: TileDataArray,
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

        use tile::{
            Kind::*,
            Visibility::*,
            HintSpec::*,
            RelativeDelta::{self, *},
            DistanceIntel::{self, *},
            HybridOffset::{self, *},
        };

        const SCALE_FACTOR: usize = 512;

        let mut tiles_remaining = match level {
            Level::One => 49 * 4,//SCALE_FACTOR * 1,
            Level::Two => SCALE_FACTOR * 2,
            Level::Three => SCALE_FACTOR * 3,
        };


        let mut xy_iter = match 1 {//xs_u32(rng, 0, 5) {
            0 => tile::XY::all_center_spiralish(),
            /*
            1 => tile::XY::upper_left_quadrant_spiralish(),
            2 => tile::XY::upper_right_quadrant_spiralish(),
            3 => tile::XY::lower_left_quadrant_spiralish(),
            _ => tile::XY::lower_right_quadrant_spiralish(),
            */
            _ => tile::XY::upper_left_spiral_in(),
        };
        for xy in &mut xy_iter {
            let kind = match xs_u32(rng, 0, 15) {
                1 => Red(HybridOffset::from_rng(rng), Hidden, Full),
                2 => Green(HybridOffset::from_rng(rng), Hidden, Full),
                3 => Blue(HybridOffset::from_rng(rng), Hidden, Full),
                4|7|10 => Red(HybridOffset::from_rng(rng), Hidden, DistanceIntel::from_rng(rng)),
                5|8|11 => Green(HybridOffset::from_rng(rng), Hidden, DistanceIntel::from_rng(rng)),
                6|9|12 => Blue(HybridOffset::from_rng(rng), Hidden,  DistanceIntel::from_rng(rng)),
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
        // TODO: If we add multiple islands of tiles, then we'll need to pull from 
        // each bounding rect uniformly at random.
        let bounding_rect = xy_iter.bounding_rect();

        macro_rules! set_random_tile {
            ($vis: ident, $($from: pat)|+ => $to: expr) => {{
                set_random_tile!(_offset, $vis, _intel, $($from)|+ => $to)
            }};
            ($offset: ident, $vis: ident, $intel: ident, $($from: pat)|+ => $to: expr) => {{
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

                // To avoid not having a selected xy, if we don't have one yet,
                // it seems reasonable to fallback to a less restrictive pattern
                // TODO: Maybe allow disabling this?
                if selected_xy.is_none() {
                    index = tile::xy_to_i(start_xy);
                    for _ in 0..TILES_LENGTH as usize {
                        match tiles[index].kind {
                            RedStar(..)
                            | GreenStar(..)
                            | BlueStar(..)
                            | Goal(..)
                            | Empty => {
                                // Don't overwrite these,  even if you can't find
                                // any room!
                            },
                            Red(offset, vis, intel)
                            | Green(offset, vis, intel)
                            | Blue(offset, vis, intel)
                            | GoalDistance(offset, vis, intel) => {
                                let $offset = offset;
                                let $vis = vis;
                                let $intel = intel;
                                tiles[index].kind = $to;
                                selected_xy = Some(tile::i_to_xy(index));
                                break
                            },
                            Hint(vis, _)
                            | Between(vis, _) => {
                                // This _very_ slightly tips the prevalence of
                                // these offsets and intels but what would a
                                // player do with this information?
                                let $offset = Zero;
                                let $intel = Full;
                                let $vis = vis;
                                tiles[index].kind = $to;
                                selected_xy = Some(tile::i_to_xy(index));
                                break
                            }
                        }

                        index += 1;
                        index %= TILES_LENGTH as usize;
                    }
                }

                if cfg!(debug_assertions) {
                    selected_xy.expect("set_random_tile found no tile!")
                } else {
                    selected_xy.unwrap_or_default()
                }
            }}
        }

        let red_star_xy = set_random_tile!(vis, Red(Zero, vis, _) => RedStar(vis));
        let green_star_xy = set_random_tile!(vis, Green(Zero, vis, _) => GreenStar(vis));
        let blue_star_xy = set_random_tile!(vis, Blue(Zero, vis, _) => BlueStar(vis));

        let goal_xy = set_random_tile!(
            vis,
            Red(Zero, vis, _)|Green(Zero, vis, _)|Blue(Zero, vis, _) => Goal(vis)
        );

        set_random_tile!(
            offset,
            vis,
            intel,
            Red(offset, vis, intel) => GoalDistance(
                offset,
                vis,
                intel
            )
        );

        set_random_tile!(
            offset,
            vis,
            intel,
            Green(offset, vis, intel) => GoalDistance(
                offset,
                vis,
                intel
            )
        );

        set_random_tile!(
            offset,
            vis,
            intel,
            Blue(offset, vis, intel) => GoalDistance(
                offset,
                vis,
                intel
            )
        );

        macro_rules! set_hint {
            ($hint_spec: expr) => {
                let _ = set_random_tile!(
                    vis,
                    Red(Zero, vis, _)
                    |Green(Zero, vis, _)
                    |Blue(Zero, vis, _) => Hint(vis, $hint_spec)
                );
            }
        }

        // TODO Only add these sometimes? Maybe based on difficulty?
        set_hint!(GoalIs(OneUpOneLeft));
        set_hint!(GoalIs(OneUp));
        set_hint!(GoalIs(OneUpOneRight));
        set_hint!(GoalIs(OneLeft));
        set_hint!(GoalIs(OneRight));
        set_hint!(GoalIs(OneDownOneLeft));
        set_hint!(GoalIs(OneDown));
        set_hint!(GoalIs(OneDownOneRight));

        set_hint!(GoalIs(TwoUpTwoLeft));
        set_hint!(GoalIs(TwoUpOneLeft));
        set_hint!(GoalIs(TwoUp));
        set_hint!(GoalIs(TwoUpOneRight));
        set_hint!(GoalIs(TwoUpTwoRight));
        set_hint!(GoalIs(OneUpTwoLeft));
        set_hint!(GoalIs(OneUpTwoRight));
        set_hint!(GoalIs(TwoLeft));
        set_hint!(GoalIs(TwoRight));
        set_hint!(GoalIs(OneDownTwoRight));
        set_hint!(GoalIs(OneDownTwoLeft));
        set_hint!(GoalIs(TwoDownTwoLeft));
        set_hint!(GoalIs(TwoDownOneLeft));
        set_hint!(GoalIs(TwoDown));
        set_hint!(GoalIs(TwoDownOneRight));
        set_hint!(GoalIs(TwoDownTwoRight));

        // GoalIsNot section
        {
            use tile::{
                HintTileIndex,
                NonZeroHintTileIndex,
                HintTile,
                HINTS_PER_GOAL_IS_NOT
            };

            let mut deck = RelativeDelta::ALL.clone();
            xs_shuffle(rng, &mut deck);

            let mut deck_unused_i = 0;
            while RelativeDelta::ALL.len() - deck_unused_i >= HINTS_PER_GOAL_IS_NOT {
                let mut deltas = [RelativeDelta::default(); HINTS_PER_GOAL_IS_NOT];

                for i in 0..HINTS_PER_GOAL_IS_NOT {
                    deltas[i] = deck[deck_unused_i];
                    deck_unused_i += 1;
                }
                // Reverse order so hints are read top left to bottom right.
                deltas.sort_by(|a, b| a.cmp(b).reverse());

                macro_rules! gen_offset {
                    () => {{
                        let offset = xs_u32(rng, 1, HintTile::COUNT as u32);
                        let offset = NonZeroHintTileIndex::new(offset as HintTileIndex)
                            .unwrap_or(HintTile::NON_ZERO_MAX_INDEX);

                        offset
                    }}
                }

                let offsets = [
                    gen_offset!(),
                    gen_offset!(),
                    gen_offset!(),
                    // Extra in case of bad offsets.
                    gen_offset!(), 
                ];

                set_hint!(GoalIsNot(deltas, offsets));
            }
        }

        {
            use tile::{
                BetweenSpec::*,
                VisualKind,
            };
            macro_rules! set_between_hint {
                ($between_spec: expr) => {
                    let _ = set_random_tile!(
                        vis,
                        Red(Zero, vis, _)
                        |Green(Zero, vis, _)
                        |Blue(Zero, vis, _) => Between(vis, $between_spec)
                    );
                }
            }
    
            for visual_kind in VisualKind::ALL {
                set_between_hint!(Minimum(
                    tile::WrappingDeltaXY::from_rng(rng),
                    visual_kind
                ));
            }
        }

        Self {
            tiles,
            red_star_xy,
            green_star_xy,
            blue_star_xy,
            goal_xy,
        }
    }
}

fn get_tile_from_array(tile_array: &TileDataArray, xy: tile::XY) -> Tile {
    Tile {
        xy,
        data: tile_array[tile::xy_to_i(xy)]
    }
}

fn get_tile(tiles: &Tiles, xy: tile::XY) -> Tile {
    get_tile_from_array(&tiles.tiles, xy)
}

fn get_tile_kind_from_array(tile_array: &TileDataArray, xy: tile::XY) -> tile::Kind {
    tile_array[tile::xy_to_i(xy)].kind
}

fn get_tile_visual_kind(tiles: &Tiles, xy: tile::XY) -> tile::VisualKind {
    tile::VisualKind::from(get_tile_kind_from_array(&tiles.tiles, xy))
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MinimumOutcome {
    NoMatchingTiles,
    Count(tile::Count)
}

impl MinimumOutcome {
    fn unwrap_or_default(&self) -> tile::Count {
        match self {
            Self::NoMatchingTiles => <_>::default(),
            Self::Count(count) => *count,
        }
    }
}

#[cfg(test)]
mod between_tests;

#[derive(Copy, Clone, Debug)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right
}

pub(crate) fn apply_dir(dir: Dir, xy: tile::XY) -> Option<tile::XY> {
    use tile::XY;
    match dir {
        Dir::Up => xy.y.checked_sub_one().map(|y| XY {
            y,
            ..xy
        }),
        Dir::Down => xy.y.checked_add_one().map(|y| XY {
            y,
            ..xy
        }),
        Dir::Left => xy.x.checked_sub_one().map(|x| XY {
            x,
            ..xy
        }),
        Dir::Right => xy.x.checked_add_one().map(|x| XY {
            x,
            ..xy
        }),
    }
}

pub(crate) fn get_long_and_short_dir(
    from: tile::XY,
    to: tile::XY
) -> (Dir, Dir) {
    let from_x = usize::from(from.x);
    let from_y = usize::from(from.y);
    let to_x = usize::from(to.x);
    let to_y = usize::from(to.y);

    let x_distance = (from_x as isize - to_x as isize).abs();
    let y_distance = (from_y as isize - to_y as isize).abs();
    let x_dir = if from_x > to_x {
        Dir::Left
    } else {
        Dir::Right
    };
    let y_dir = if from_y > to_y {
        Dir::Up
    } else {
        Dir::Down
    };
    
    if x_distance > y_distance {
        (x_dir, y_dir)
    } else {
        (y_dir, x_dir)
    }
}

/// Used for an algorithm resembling https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm.
#[derive(Clone, Copy, Debug)]
struct DykstrasTileData {
    tentative_count: tile::Count,
    visited: bool,
}

impl Default for DykstrasTileData {
    fn default() -> Self {
        Self {
            tentative_count: tile::Count::max_value(),
            visited: false,
        }
    }
}

type DykstrasTileSet = [DykstrasTileData; TILES_LENGTH as usize];

fn minimum_between_of_visual_kind(
    tiles: &Tiles,
    from: tile::XY,
    to: tile::XY,
    visual_kind: tile::VisualKind
) -> MinimumOutcome {
    if from == to {
        return MinimumOutcome::NoMatchingTiles;
    }

    let (long_dir, short_dir) = get_long_and_short_dir(from, to);

    let mut set: DykstrasTileSet = [DykstrasTileData::default(); TILES_LENGTH as usize];


    set[tile::xy_to_i(from)].tentative_count = 0;

    let max_xy = tile::XY {x: core::cmp::max(from.x, to.x), y: core::cmp::max(from.y, to.y)};
    let min_xy = tile::XY {x: core::cmp::min(from.x, to.x), y: core::cmp::min(from.y, to.y)};

    let max_x = usize::from(max_xy.x);
    let max_y = usize::from(max_xy.y);

    let min_x = usize::from(min_xy.x);
    let min_y = usize::from(min_xy.y);

    let mut minimum = tile::Count::max_value();
    let mut current_xy = from;

    // TODO Capacity bound could be tighter.
    let mut next_xys = std::collections::VecDeque::with_capacity(max_x * max_y);

    loop {
        let current_index = tile::xy_to_i(current_xy);

        for &dir in [long_dir, short_dir].iter() {
            let mut current_count: tile::Count = set[current_index].tentative_count;

            let xy_opt = apply_dir(dir, current_xy);
            let new_xy = match xy_opt {
                Some(new_xy) => {
                    if new_xy.x > max_xy.x
                    || new_xy.y > max_xy.y
                    || new_xy.x < min_xy.x
                    || new_xy.y < min_xy.y {
                        continue;
                    }
                    new_xy
                },
                None => {
                    continue;
                }
            };

            let i = tile::xy_to_i(new_xy);

            if set[i].visited {
                continue;
            }
            next_xys.push_back(new_xy);

            if visual_kind == get_tile_visual_kind(tiles, new_xy) {
                current_count += 1;
            }

            if current_count < set[i].tentative_count {
                set[i].tentative_count = current_count;
            }
        }

        set[current_index].visited = true;

        let target = set[tile::xy_to_i(to)];

        if target.visited {
            minimum = target.tentative_count;
            break;
        }

        // find the next current xy: an unvisited node with smallest count.
        let mut next_xy = None;

        while let Some(xy) = next_xys.pop_front() {
            let i = tile::xy_to_i(xy);

            if set[i].visited {
                continue;
            }

            // TODO Is this ever false?
            if set[i].tentative_count < tile::Count::max_value() {
                next_xy = Some(xy);
                break;
            }
        }

        if let Some(next_xy) = next_xy {
            current_xy = next_xy;
        } else {
            break;
        }
    }

    compile_time_assert!(tile::Count::max_value() > tile::Coord::COUNT);
    // Given the compile-time assert above, we know that if we got 
    // `tile::Count::max_value()` here, then it is because something is very wrong.
    debug_assert!(minimum != tile::Count::max_value());

    // We are getting the `minimum_between`, so we don't want to count the 
    // end tile, so decrement it if it was incremented.
    if visual_kind == get_tile_visual_kind(tiles, to) {
        debug_assert!(minimum != 0);
        minimum = minimum.saturating_sub(1);
    }

    if minimum == 0 {
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let xy = tile::XY{x: tile::X::ALL[x], y: tile::Y::ALL[y]};

                if visual_kind == get_tile_visual_kind(tiles, xy) {
                    return MinimumOutcome::Count(minimum);
                }
            }
        }

        return MinimumOutcome::NoMatchingTiles;
    }

    MinimumOutcome::Count(minimum)
}

type DistanceInfo = (tile::XY, tile::DistanceIntel);

fn distance_info_from_kind(tiles: &Tiles, kind: tile::Kind) -> Option<DistanceInfo> {
    use tile::{Kind::*, Visibility::*};

    match kind {
        Red(_, Shown, intel) => Some((
            get_star_xy(tiles, tile::Colour::Red),
            intel,
        )),
        Green(_, Shown, intel) => Some((
            get_star_xy(tiles, tile::Colour::Green),
            intel,
        )),
        Blue(_, Shown, intel) => Some((
            get_star_xy(tiles, tile::Colour::Blue),
            intel,
        )),
        GoalDistance(_, Shown, goal_intel) => Some((
            tiles.goal_xy,
            tile::DistanceIntel::from(goal_intel),
        )),
        Empty 
        | RedStar(_) | GreenStar(_) | BlueStar(_)
        | Goal(_) | Hint(_, _) | Between(_, _)
        | Red(_, Hidden, _)
        | Green(_, Hidden, _)
        | Blue(_, Hidden, _)
        | GoalDistance(_, Hidden, _) => None,
    }
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

const MAX_RULER_COUNT: usize = 8;

type Rulers = [Option<tile::XY>; MAX_RULER_COUNT];

fn push_ruler_saturating(rulers: &mut Rulers, element: tile::XY) {
    for i in 0..MAX_RULER_COUNT {
        if rulers[i].is_none() {
            rulers[i] = Some(element);
            break;
        }
    }
}

fn ruler_pos_iter(rulers: &Rulers) -> impl Iterator<Item = tile::XY> + '_ {
    rulers.iter().scan((), |(), item: &Option<tile::XY>| *item)
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Tool {
    Selectrum,
    Ruler(Rulers)
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

/// 64k animation frames ought to be enough for anybody!
type AnimationTimer = u16;

/// We use this because it has a lot more varied factors than 65536.
const ANIMATION_TIMER_LENGTH: AnimationTimer = 60 * 60 * 18;

#[derive(Debug, Default)]
pub struct State {
    sizes: draw::Sizes,
    board: Board,
    input_speed: InputSpeed,
    tool: Tool,
    view_mode: ViewMode,
    animation_timer: AnimationTimer
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

fn is_last_level(board: &Board) -> bool {
    next_level(board.level) == board.level
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

fn render_goal_sprite(board: &Board) -> SpriteKind {
    if is_last_level(board) {
        SpriteKind::TerminalGoal
    } else {
        SpriteKind::InstrumentalGoal
    }
}

mod hint {
    pub(crate) const TILES_PER_ROW: usize = 5;
    pub(crate) const TILES_PER_COLUMN: usize = 5;
    pub(crate) const TILES_COUNT: usize = TILES_PER_ROW * TILES_PER_COLUMN;
    
    pub(crate) const CENTER_INDEX: usize = TILES_COUNT / 2;
    pub(crate) const UP_INDEX: usize = CENTER_INDEX - TILES_PER_ROW;
    pub(crate) const DOWN_INDEX: usize = CENTER_INDEX + TILES_PER_ROW;
    pub(crate) const LEFT_INDEX: usize = CENTER_INDEX - 1;
    pub(crate) const RIGHT_INDEX: usize = CENTER_INDEX + 1;
    pub(crate) const UP_LEFT_INDEX: usize = UP_INDEX - 1;
    pub(crate) const UP_RIGHT_INDEX: usize = UP_INDEX + 1;
    pub(crate) const DOWN_LEFT_INDEX: usize = DOWN_INDEX - 1;
    pub(crate) const DOWN_RIGHT_INDEX: usize = DOWN_INDEX + 1;

    pub(crate) const TWO_UP_TWO_LEFT_INDEX: usize = TWO_UP_INDEX - 2;
    pub(crate) const TWO_UP_ONE_LEFT_INDEX: usize = TWO_UP_INDEX - 1;
    pub(crate) const TWO_UP_INDEX: usize = CENTER_INDEX - (2 * TILES_PER_ROW);
    pub(crate) const TWO_UP_ONE_RIGHT_INDEX: usize = TWO_UP_INDEX + 1;
    pub(crate) const TWO_UP_TWO_RIGHT_INDEX: usize = TWO_UP_INDEX + 2;
    pub(crate) const UP_TWO_LEFT_INDEX: usize = TWO_LEFT_INDEX - TILES_PER_ROW;
    pub(crate) const UP_TWO_RIGHT_INDEX: usize = TWO_RIGHT_INDEX - TILES_PER_ROW;
    pub(crate) const TWO_LEFT_INDEX: usize = LEFT_INDEX - 1;
    pub(crate) const TWO_RIGHT_INDEX: usize = RIGHT_INDEX + 1;
    pub(crate) const DOWN_TWO_LEFT_INDEX: usize = TWO_LEFT_INDEX + TILES_PER_ROW;
    pub(crate) const DOWN_TWO_RIGHT_INDEX: usize = TWO_RIGHT_INDEX + TILES_PER_ROW;
    pub(crate) const TWO_DOWN_TWO_LEFT_INDEX: usize = TWO_DOWN_INDEX - 2;
    pub(crate) const TWO_DOWN_ONE_LEFT_INDEX: usize = TWO_DOWN_INDEX - 1;
    pub(crate) const TWO_DOWN_INDEX: usize = CENTER_INDEX + (2 * TILES_PER_ROW);
    pub(crate) const TWO_DOWN_ONE_RIGHT_INDEX: usize = TWO_DOWN_INDEX + 1;
    pub(crate) const TWO_DOWN_TWO_RIGHT_INDEX: usize = TWO_DOWN_INDEX + 2;
}

#[derive(Clone, Copy, Debug)]
enum HintOverlay {
    NoOverlay,
    Sprite(SpriteKind),
    Label(tile::Label),
}

#[derive(Clone, Copy, Debug)]
struct HintSprite {
    sprite: SpriteKind,
    overlay: HintOverlay,
}

impl HintSprite {
    fn no_overlay(sprite: SpriteKind) -> Self {
        HintSprite {
            sprite,
            overlay: HintOverlay::NoOverlay,
        }
    }

    fn not_symbol_overlay(sprite: SpriteKind) -> Self {
        HintSprite {
            sprite,
            overlay: HintOverlay::Sprite(SpriteKind::NotSymbol),
        }
    }

    fn label_overlay(sprite: SpriteKind, label: tile::Label) -> Self {
        HintSprite {
            sprite,
            overlay: HintOverlay::Label(label),
        }
    }
}

impl From<HintSprite> for SpriteKind {
    fn from(hint_sprite: HintSprite) -> Self {
        hint_sprite.sprite
    }
}

type HintInfo = (String, [Option<HintSprite>; hint::TILES_COUNT]);

struct HintTarget {
    direction: &'static str,
    xy: Result<tile::XY, tile::WentOff>,
}

fn render_hint_target(
    relative_delta: tile::RelativeDelta,
    goal_xy: tile::XY,
) -> HintTarget {
    use tile::{RelativeDelta::*, WentOff::{self, *}};

    fn merge(a: WentOff, b: WentOff) -> WentOff {
        match (a, b) {
            (UpEdge, LeftEdge)
            | (LeftEdge, UpEdge)
            | (UpAndLeftEdges, UpEdge)
            | (UpAndLeftEdges, LeftEdge) => UpAndLeftEdges,
            (UpEdge, RightEdge)
            | (RightEdge, UpEdge)
            | (UpAndRightEdges, UpEdge)
            | (UpAndRightEdges, RightEdge) => UpAndRightEdges,
            (DownEdge, LeftEdge)
            | (LeftEdge, DownEdge)
            | (DownAndLeftEdges, DownEdge)
            | (DownAndLeftEdges, LeftEdge) => DownAndLeftEdges,
            (DownEdge, RightEdge)
            | (RightEdge, DownEdge)
            | (DownAndRightEdges, DownEdge)
            | (DownAndRightEdges, RightEdge) => DownAndRightEdges,
            // If they match this does the right thing. If they are an
            // unexpected pair, say, (UpEdge, DownEdge), then we return 
            // something that is at least partially right.
            _ => b
        }
    }

    // These macros assume that they will only chained in such a way that 
    // the edge lines are only crossed once in each direction. At the upper
    // left corner, just up, or up then left should both work. But up then 
    // down, etc. will likely have undesired behaviour.
    macro_rules! inc_x {
        ($xy: expr) => {{
            match $xy {
                Ok(xy) => xy.x.checked_add_one().map(|x| tile::XY {
                    x,
                    ..xy
                }).ok_or(RightEdge),
                Err(went_off) => Err(merge(went_off, RightEdge))
            }
        }}
    }

    macro_rules! dec_x {
        ($xy: expr) => {{
            match $xy {
                Ok(xy) => xy.x.checked_sub_one().map(|x| tile::XY {
                    x,
                    ..xy
                }).ok_or(LeftEdge),
                Err(went_off) => Err(merge(went_off, LeftEdge))
            }
        }}
    }

    macro_rules! inc_y {
        ($xy: expr) => {{
            match $xy {
                Ok(xy) => xy.y.checked_add_one().map(|y| tile::XY {
                    y,
                    ..xy
                }).ok_or(DownEdge),
                Err(went_off) => Err(merge(went_off, DownEdge))
            }
        }}
    }

    macro_rules! dec_y {
        ($xy: expr) => {{
            match $xy {
                Ok(xy) => xy.y.checked_sub_one().map(|y| tile::XY {
                    y,
                    ..xy
                }).ok_or(UpEdge),
                Err(went_off) => Err(merge(went_off, UpEdge))
            }
        }}
    }

    macro_rules! use_most_diagonal_err {
        ($start_expr: expr => $op1: ident, $op2: ident $(,)? ) => {{
            match ($op1!($op2!($start_expr)), $op2!($op1!($start_expr))) {
                (Ok(a), _) | (_, Ok(a)) => Ok(a),
                (Err(a), Err(b)) => Err(merge(a, b)),
            }
        }};
        ($start_expr: expr => $op1: ident * 2, $op2: ident $(,)? ) => {{
            match (
                $op1!($op1!($op2!($start_expr))),
                $op2!($op1!($op1!($start_expr)))
            ) {
                (Ok(a), _) | (_, Ok(a)) => Ok(a),
                (Err(a), Err(b)) => Err(merge(a, b)),
            }
        }};
        ($start_expr: expr => $op1: ident, $op2: ident * 2 $(,)? ) => {{
            match (
                $op1!($op2!($op2!($start_expr))),
                $op2!($op2!($op1!($start_expr)))
            ) {
                (Ok(a), _) | (_, Ok(a)) => Ok(a),
                (Err(a), Err(b)) => Err(merge(a, b)),
            }
        }};
        ($start_expr: expr => $op1: ident * 2, $op2: ident * 2 $(,)? ) => {{
            match (
                $op1!($op1!($op2!($op2!($start_expr)))),
                $op2!($op2!($op1!($op1!($start_expr))))
            ) {
                (Ok(a), _) | (_, Ok(a)) => Ok(a),
                (Err(a), Err(b)) => Err(merge(a, b)),
            }
        }};
    }

    let (direction, xy) = match relative_delta {
        OneUpOneLeft => (
            "one up and left",
            // go one down, one right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x, inc_y
            )
        ),
        // go one down from goal
        OneUp => ("one up", inc_y!(Ok(goal_xy))),
        OneUpOneRight => (
            "one up and right",
            // go one down, one left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x, inc_y
            )
        ),
        // go one right from goal
        OneLeft => ("one left", inc_x!(Ok(goal_xy))),
        // go one left from goal
        OneRight => ("one right", dec_x!(Ok(goal_xy))),
        OneDownOneLeft => (
            "one down and left",
            // go one up, one right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x, dec_y
            )
        ),
        // go one up from goal
        OneDown => ("one down", dec_y!(Ok(goal_xy))),
        
        OneDownOneRight => (
            "one down and right",
            // go one up, one left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x, dec_y
            )
        ),

        TwoUpTwoLeft => (
            "two up and two left",
            // go two down, two right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x * 2, inc_y * 2
            )
        ),
        TwoUpOneLeft => (
            "two up and one left",
            // go two down, one right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x, inc_y * 2
            )
        ),
        // go two down from goal
        TwoUp => ("two up", inc_y!(inc_y!(Ok(goal_xy)))),
        TwoUpOneRight => (
            "two up and one right",
            // go two down, one left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x, inc_y * 2
            )
        ),
        TwoUpTwoRight => (
            "two up and two right",
            // go two down, two left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x * 2, inc_y * 2
            )
        ),
        OneUpTwoLeft => (
            "one up and two left",
            // go one down, two right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x * 2, inc_y
            )
        ),
        OneUpTwoRight => (
            "one up and two right",
            // go two down, two left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x * 2, inc_y
            )
        ),
        // go two right from goal
        TwoLeft => ("two left", inc_x!(inc_x!(Ok(goal_xy)))),
        // go two left from goal
        TwoRight => ("two right", dec_x!(dec_x!(Ok(goal_xy)))),
        OneDownTwoLeft => (
            "one down and two left",
            // go one up, two right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x * 2, dec_y
            )
        ),
        OneDownTwoRight => (
            "one down and two right",
            // go one up, two left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x * 2, dec_y
            )
        ),
        TwoDownTwoLeft => (
            "two down and two left",
            // go two up, two right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x * 2, dec_y * 2
            )
        ),
        TwoDownOneLeft => (
            "two down and one left",
            // go two up, one right from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => inc_x, dec_y * 2
            )
        ),
        // go two up from goal
        TwoDown => ("two down", dec_y!(dec_y!(Ok(goal_xy)))),
        TwoDownOneRight => (
            "two down and one right",
            // go two up, one left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x, dec_y * 2
            )
        ),        
        TwoDownTwoRight => (
            "two down and two right",
            // go two up, two left from goal
            use_most_diagonal_err!(
                Ok(goal_xy) => dec_x * 2, dec_y * 2
            )
        ),
    };

    HintTarget{
        direction, 
        xy
    }
}

fn render_hint_spec(
    tile_array: &TileDataArray,
    hint_spec: tile::HintSpec,
    goal_sprite: SpriteKind,
    goal_xy: tile::XY,
) -> HintInfo {
    use SpriteKind::*;
    use tile::{HintSpec::*, RelativeDelta::*, WentOff::*};

    let mut hint_string = String::new();

    let mut hint_sprites = [
        Some(HintSprite::no_overlay(SpriteKind::QuestionMark));
        hint::TILES_COUNT
    ];

    fn target_index_from_relative_delta(
        relative_delta: tile::RelativeDelta
    ) -> usize {
        match relative_delta {
            OneUpOneLeft => hint::DOWN_RIGHT_INDEX,
            OneUp => hint::DOWN_INDEX,
            OneUpOneRight => hint::DOWN_LEFT_INDEX,
            OneLeft => hint::RIGHT_INDEX,
            OneRight => hint::LEFT_INDEX,
            OneDownOneLeft => hint::UP_RIGHT_INDEX,
            OneDown => hint::UP_INDEX,
            OneDownOneRight => hint::UP_LEFT_INDEX,
    
            TwoUpTwoLeft => hint::TWO_DOWN_TWO_RIGHT_INDEX,
            TwoUpOneLeft => hint::TWO_DOWN_ONE_RIGHT_INDEX,
            TwoUp => hint::TWO_DOWN_INDEX,
            TwoUpOneRight => hint::TWO_DOWN_ONE_LEFT_INDEX,
            TwoUpTwoRight => hint::TWO_DOWN_TWO_LEFT_INDEX,
            OneUpTwoLeft => hint::DOWN_TWO_RIGHT_INDEX,
            OneUpTwoRight => hint::DOWN_TWO_LEFT_INDEX,
            TwoLeft => hint::TWO_RIGHT_INDEX,
            TwoRight => hint::TWO_LEFT_INDEX,
            OneDownTwoRight => hint::UP_TWO_LEFT_INDEX,
            OneDownTwoLeft => hint::UP_TWO_RIGHT_INDEX,
            TwoDownTwoLeft => hint::TWO_UP_TWO_RIGHT_INDEX,
            TwoDownOneLeft => hint::TWO_UP_ONE_RIGHT_INDEX,
            TwoDown => hint::TWO_UP_INDEX,
            TwoDownOneRight => hint::TWO_UP_ONE_LEFT_INDEX,
            TwoDownTwoRight => hint::TWO_UP_TWO_LEFT_INDEX,
        }
    }

    match hint_spec {
        GoalIs(relative_delta) => {
            let HintTarget{ direction, xy: target_xy } = render_hint_target(
                relative_delta,
                goal_xy,
            );
        
            let description = if let Ok(target_xy) = target_xy {
                tile::kind_description(get_tile_kind_from_array(tile_array, target_xy))
            } else {
                "the edge of the grid"
            };
        
            hint_string.push_str(&format!(
                "The goal is {} from {}.",
                direction,
                description
            ));
        
            hint_sprites[hint::CENTER_INDEX] = Some(
                HintSprite::no_overlay(goal_sprite)
            );
        
            let target_sprite: Option<SpriteKind> = match target_xy {
                Ok(target_xy) => {
                    draw::sprite_kind_from_hint_tile(
                        tile::hint_tile_from_kind(
                            get_tile_kind_from_array(tile_array, target_xy)
                        ),
                        goal_sprite,
                    )
                },
                Err(went_off) => {
                    // When talking about which edges were went off of, it's most
                    // natural to talk about the edge from the perspective of the
                    // tile. When talking about the sprite it is most natural to
                    // talk about the direction the arrow is pointing.
                    let sprite = match went_off {
                        UpAndLeftEdges => EdgeDownRight,
                        UpEdge => EdgeDown,
                        UpAndRightEdges => EdgeDownLeft,
                        LeftEdge => EdgeRight,
                        RightEdge => EdgeLeft,
                        DownAndLeftEdges => EdgeUpRight,
                        DownEdge => EdgeUp,
                        DownAndRightEdges => EdgeUpLeft,
                    };
                    Some(sprite)
                }
            };
        
            let target_index = target_index_from_relative_delta(
                relative_delta
            );
        
            hint_sprites[target_index] = target_sprite.map(
                HintSprite::no_overlay
            );
        },
        GoalIsNot(relative_deltas, hint_tile_offsets) => {
            use tile::{
                Category,
                hint_tile_category,
                EXTRA_OFFSET_INDEX,
                HINTS_PER_GOAL_IS_NOT
            };

            hint_string.push_str("The goal is not ");

            for i in 0..HINTS_PER_GOAL_IS_NOT {
                let relative_delta = relative_deltas[i];
                let hint_tile_offset = hint_tile_offsets[i];

                let HintTarget{ direction, xy: target_xy } = render_hint_target(
                    relative_delta,
                    goal_xy,
                );

                let different_hint_tile = {
                    let actual_hint_tile = tile::hint_tile_from_kind_went_off_result(
                        target_xy.map(|target_xy| 
                            get_tile_kind_from_array(
                                tile_array,
                                target_xy
                            )
                        )
                    );
                    let mut hint_tile_index: tile::HintTileIndex = 0;
                    for &hint_tile in tile::HintTile::ALL.iter() {
                        if actual_hint_tile == hint_tile {
                            break;
                        }

                        hint_tile_index += 1;
                    }

                    let different_hint_tile_index = 
                        (hint_tile_index + hint_tile_offset.get())
                        % tile::HintTile::ALL.len()
                    ;

                    let different_hint_tile = tile::HintTile::ALL[
                        different_hint_tile_index
                    ];

                    match (
                        hint_tile_category(actual_hint_tile),
                        hint_tile_category(different_hint_tile),
                    ) {
                        (Category::OffTheEdge, Category::OffTheEdge) => {
                            let mut corrected_offset = different_hint_tile_index;
                            while 
                                hint_tile_category(
                                    tile::HintTile::ALL[corrected_offset]
                                ) == Category::OffTheEdge {
                                corrected_offset += 1;
                                corrected_offset %= tile::HintTile::COUNT;
                            }

                            let mut extra_offset_counter = hint_tile_offsets[
                                EXTRA_OFFSET_INDEX
                            ].get();

                            while extra_offset_counter > 0 {
                                corrected_offset += 1;
                                corrected_offset %= tile::HintTile::COUNT;

                                if hint_tile_category(
                                    tile::HintTile::ALL[corrected_offset]
                                ) != Category::OffTheEdge {
                                    extra_offset_counter -= 1;
                                }
                            }

                            tile::HintTile::ALL[corrected_offset]
                        }
                        (Category::OffTheEdge, Category::Misc)
                        | (Category::Misc, Category::OffTheEdge)
                        // Might make sense to handle (Misc, Misc) differently 
                        // when/if a third category is added.
                        | (Category::Misc, Category::Misc) => {
                            // All good.
                            different_hint_tile
                        }
                    }
                };
            
                let description = tile::hint_tile_description(
                    different_hint_tile
                );
            
                hint_string.push_str(&format!(
                    "{} from {}{}",
                    direction,
                    description,
                    if i == HINTS_PER_GOAL_IS_NOT - 1 {
                        "."
                    } else if i == HINTS_PER_GOAL_IS_NOT - 2 {
                        ", or "
                    } else {
                        ", "
                    }
                ));
            
                hint_sprites[hint::CENTER_INDEX] = Some(
                    HintSprite::no_overlay(goal_sprite)
                );
            
                let target_sprite: Option<SpriteKind> = 
                    draw::sprite_kind_from_hint_tile(
                        different_hint_tile,
                        goal_sprite,
                    );
            
                let target_index = target_index_from_relative_delta(
                    relative_delta
                );
            
                hint_sprites[target_index] = target_sprite.map(
                    HintSprite::not_symbol_overlay
                );
            }
        },
    };

    (
        hint_string,
        hint_sprites,
    )
}

#[cfg(test)]
mod hint_tests;

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
                Tile(xy) => {
                    let mut rulers = [None; MAX_RULER_COUNT];

                    push_ruler_saturating(&mut rulers, xy);

                    Ruler(rulers)
                }
            },
            Ruler(_) => Selectrum,
        };
    }

    if INPUT_TOOL_RIGHT_PRESSED & input_flags != 0 {
        state.tool = match state.tool {
            Selectrum => match state.board.ui_pos {
                Tile(xy) => {
                    let mut rulers = [None; MAX_RULER_COUNT];
                    rulers[0] = Some(xy);
                    Ruler(rulers)
                }
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
                                    if let $variant(offset, Hidden, intel) = state.board.tiles.tiles[index].kind {
                                        state.board.tiles.tiles[index].kind = $variant(offset, Shown, intel);
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
                        if is_last_level(&state.board) {
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
                Ruler(ref mut rulers) => {
                    let mut saw_match = false;
                    for i in 0..MAX_RULER_COUNT {
                        if let Some(ref mut pos) = rulers[i] {
                            if *pos == *xy {
                                saw_match = true;
                            }
                        } else {
                            break;
                        }
                    }

                    if saw_match {
                        // We find that we want to be able to press `Interact` after
                        // measuring with the ruler, and have a dig happen. But we
                        // don't want to make mistakes easy to make, so we make it
                        // require multiple presses to do that.
                        do_ui_reset!();
                        // We considered making the reset happen if there are no rulers
                        // left as well, but if all the rulers are used, accidentally
                        // removing all of them because there aren't any left seems worse
                        // than needing to switch tool modes manually.
                    } else {
                        push_ruler_saturating(rulers, *xy);
                    }
                }
            }
        },
    }

    let goal_sprite = render_goal_sprite(&state.board);

    macro_rules! push_tile_label {
        ($label: expr, $draw_xy: expr) => {{
            let label: tile::Label = $label;
            let text: String = label.chars_string();
            commands.push(Text(TextSpec {
                text,
                xy: $draw_xy,
                wh: DrawWH {
                    w: state.sizes.tile_side_length,
                    h: state.sizes.tile_side_length,
                },
                kind: TextKind::from(label.kind),
            }));
        }}
    }

    for txy in tile::XY::all() {
        let tiles = &state.board.tiles;
        let tile = get_tile(tiles, txy);

        let xy = draw::tile_xy_to_draw(&state.sizes, txy);

        let sprite = if let Some(sprite) = draw::sprite_kind_from_tile_kind(
            tile.data.kind,
            goal_sprite
        ) {
            sprite
        } else {
            continue
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
            let distance_info = distance_info_from_kind(tiles, tile.data.kind);

            if let Some((target_xy, intel)) = distance_info {
                let should_draw_distance = if matches!(state.view_mode, Clean) {
                    tile::is_hidden(get_tile_kind_from_array(&tiles.tiles, target_xy))
                } else {
                    // We already checked for HideRevealed above
                    true
                };

                if should_draw_distance {
                    let distance: tile::Distance = tile::manhattan_distance(
                        txy,
                        target_xy,
                    );

                    let label = tile::distance_label(
                        intel,
                        distance,
                    );

                    push_tile_label!(label, xy);
                }
            }
        }
    }

    match state.tool {
        Selectrum => {},
        Ruler(ref rulers) => {
            for pos in ruler_pos_iter(rulers) {
                commands.push(Sprite(SpriteSpec{
                    sprite: SpriteKind::RulerEnd,
                    xy: draw::tile_xy_to_draw(&state.sizes, pos),
                }));
            }
        }
    }

    if !interacted {
        commands.push(Sprite(SpriteSpec{
            sprite: SpriteKind::Selectrum,
            xy: state.board.ui_pos.xy(&state.sizes),
        }));
    }

    let hint_info = match state.board.ui_pos {
        Tile(txy) => {
            let board = &state.board;
            let tiles = &board.tiles;
            let tile = get_tile(tiles, txy);

            use tile::{Kind::*, Visibility::*};
            match tile.data.kind {
                Hint(Shown, hint_spec) => Some(render_hint_spec(
                    &board.tiles.tiles,
                    hint_spec,
                    render_goal_sprite(board),
                    tiles.goal_xy,
                )),
                Between(Shown, spec) => {
                    use tile::{HintTile, BetweenSpec::*};
                    let (between_count, description, target_hint_tile) = match spec {
                        Minimum(delta, bewteen_visual_kind) => {
                            let target_xy = tile::apply_wrap_around_delta(delta, txy);

                            let distance = tile::manhattan_distance(txy, target_xy);

                            let distance_visual_kind = get_tile_visual_kind(tiles, target_xy);

                            let minimum = minimum_between_of_visual_kind(
                                tiles,
                                txy,
                                target_xy,
                                bewteen_visual_kind
                            );

                            (
                                minimum.unwrap_or_default(),
                                match minimum {
                                    MinimumOutcome::Count(minimum) => format!(
                                        "There is a {} tile {} tiles away from here and the minimum number of {} tiles between here and there is {}.",
                                        tile::hint_tile_adjective(distance_visual_kind.into()),
                                        distance,
                                        tile::hint_tile_adjective(bewteen_visual_kind.into()),
                                        minimum
                                    ),
                                    MinimumOutcome::NoMatchingTiles => format!(
                                        "There is a {} tile {} tiles away from here and there are no {} tiles between here and there!",
                                        tile::hint_tile_adjective(distance_visual_kind.into()),
                                        distance,
                                        tile::hint_tile_adjective(bewteen_visual_kind.into()),                                        
                                    )
                                },
                                HintTile::from(bewteen_visual_kind),
                            )
                        }
                    };

                    const HOLD_FRAMES: AnimationTimer = 8;
                    let frame_number = (
                        state.animation_timer % (16 * HOLD_FRAMES)
                    ) / HOLD_FRAMES;

                    // These indexes rely on the value of hint::TILES_COUNT being 25.
                    compile_time_assert!(hint::TILES_COUNT == 25);
                    let (mut hint_index, mut target_index) = match frame_number {
                         0 | 8 => (        0, 4 * 5 + 4),
                        1 |  9 => (        1, 4 * 5 + 3),
                        2 | 10 => (        2, 4 * 5 + 2),
                        3 | 11 => (        3, 4 * 5 + 1),
                        4 | 12 => (        4, 4 * 5),
                        5 | 13 => (    5 + 4, 3 * 5),
                        6 | 14 => (2 * 5 + 4, 2 * 5),
                             _ => (3 * 5 + 4, 5),
                    };
                    if frame_number >= 8 {
                        core::mem::swap(&mut hint_index, &mut target_index);
                    }

                    let mut sprites = [
                        Some(HintSprite::no_overlay(SpriteKind::QuestionMark));
                        hint::TILES_COUNT
                    ];
                    sprites[hint_index] = Some(HintSprite::no_overlay(
                        SpriteKind::Between
                    ));
                    sprites[target_index] = draw::sprite_kind_from_hint_tile(
                        target_hint_tile,
                        goal_sprite
                    ).map(HintSprite::no_overlay);

                    sprites[hint::CENTER_INDEX] = Some(HintSprite::label_overlay(
                        SpriteKind::Hidden,
                        tile::label_from_amount_or_default(between_count),
                    ));

                    Some((
                        description,
                        sprites
                    ))
                },
                Empty
                | Red(_, _, _)
                | RedStar(_)
                | Green(_, _, _)
                | GreenStar(_)
                | Blue(_, _, _)
                | BlueStar(_)
                | Goal(_)
                | GoalDistance(_, _, _)
                | Hint(Hidden, _)
                | Between(Hidden, _) => None
            }
        }
    };

    let board_xywh = &state.sizes.board_xywh;
    let left_text_x = state.sizes.play_xywh.x + MARGIN;
    let right_text_x = board_xywh.x + board_xywh.w + MARGIN;

    const MARGIN: f32 = 16.;

    let small_section_h = state.sizes.draw_wh.h / 8. - MARGIN;

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

        if let Some((hint_string, hint_sprites)) = hint_info {
            // TODO handle the case where there is less than enough room for
            // tiles at the regular size, better.
            let left_hint_tile_x = left_text_x + state.sizes.tile_side_length;

            for column in 0..hint::TILES_PER_COLUMN {
                y += state.sizes.tile_side_length;

                for row in 0..hint::TILES_PER_ROW {
                    if let Some(HintSprite{sprite, overlay}) = hint_sprites[row + hint::TILES_PER_ROW * column] {
                        let xy = DrawXY {
                            x: left_hint_tile_x + row as DrawX * state.sizes.tile_side_length,
                            y
                        };

                        commands.push(Sprite(SpriteSpec{
                            sprite,
                            xy,
                        }));

                        match overlay {
                            HintOverlay::NoOverlay => {},
                            HintOverlay::Sprite(overlay_sprite) => {
                                commands.push(Sprite(SpriteSpec{
                                    sprite: overlay_sprite,
                                    xy,
                                }));
                            },
                            HintOverlay::Label(label) => {
                                push_tile_label!(label, xy);
                            },
                        }
                    }
                }
            }

            y += state.sizes.tile_side_length * 2.;

            commands.push(Text(TextSpec{
                text: hint_string,
                xy: DrawXY { x: left_text_x, y },
                wh: DrawWH {
                    w: board_xywh.x - left_text_x,
                    h: board_xywh.h - y
                },
                kind: TextKind::HintString,
            }));
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
            Ruler(ref rulers) => {
                let ruler_section_h = state.sizes.draw_wh.h / 12. - MARGIN;
                let mut y = state.sizes.board_xywh.y + MARGIN + ruler_section_h;

                let tiles = &state.board.tiles;

                for pos in ruler_pos_iter(rulers) {
                    let tile_kind = get_tile_kind_from_array(
                        &tiles.tiles,
                        pos
                    );
    
                    let draw_xy = DrawXY { x: right_text_x, y };

                    if let Some(sprite) = draw::sprite_kind_from_tile_kind(
                        tile_kind,
                        goal_sprite
                    ) {
                        commands.push(Sprite(SpriteSpec{
                            sprite,
                            xy: draw_xy,
                        }));
                    }

                    if let Some((target_xy, intel)) = distance_info_from_kind(
                        tiles,
                        tile_kind
                    ) {
                        let label = tile::distance_label(
                            intel,
                            // This needs to be the distance to the tile's target, so we
                            // always show the same thing as on the grid.
                            tile::manhattan_distance(pos, target_xy),
                        );

                        push_tile_label!(label, draw_xy);
                    }

                    y += state.sizes.tile_side_length;
                    y += MARGIN;

                    commands.push(Text(TextSpec{
                        text: format!("Ruler: {}", match state.board.ui_pos {
                            Tile(xy) => {
                                tile::manhattan_distance(pos, xy)
                            }
                        }),
                        xy: DrawXY { x: right_text_x, y },
                        wh: DrawWH {
                            w: board_xywh.x - left_text_x,
                            h: ruler_section_h - (
                                state.sizes.tile_side_length
                            ),
                        },
                        kind: TextKind::Ruler,
                    }));

                    y += ruler_section_h;
                }
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

    state.animation_timer += 1;
    if state.animation_timer >= ANIMATION_TIMER_LENGTH {
        state.animation_timer = 0;
    }
}
