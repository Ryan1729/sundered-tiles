#![no_std]
#![deny(unused)]

pub trait ClearableStorage<A> {
    fn clear(&mut self);

    fn push(&mut self, a: A);
}

use floats;

pub use floats::{f32_is, const_assert_valid_bi_unit, const_assert_valid_unit};

pub use floats::bi_unit::{self, X, Y, x, y, x_lt, x_gt};

pub use floats::unit::{self, W, H, w, h, proportion, Proportion};

pub const COORD_COUNT: tile::Count = tile::Coord::COUNT;

#[derive(Clone, Copy, Debug, Default)]
pub struct XY {
    pub x: X,
    pub y: Y,
}

macro_rules! xy_minimum {
    ($xy_a: expr, $xy_b: expr) => {{
        let a: XY = $xy_a;
        let b: XY = $xy_b;
        if x_lt!(a.x, b.x) && x_lt!(a.x, b.x) {
            a
        } else {
            b
        }
    }}
}

macro_rules! xy_maximum {
    ($xy_a: expr, $xy_b: expr) => {{
        let a: XY = $xy_a;
        let b: XY = $xy_b;
        if x_gt!(a.x, b.x) && x_gt!(a.x, b.x) {
            a
        } else {
            b
        }
    }}
}

impl XY {
    fn minimum(a: Self, b: Self) -> Self {
        xy_minimum!(a, b)
    }

    fn maximum(a: Self, b: Self) -> Self {
        xy_maximum!(a, b)
    }
}

/// A min/max Rect. This way of defining a rectangle has nicer behaviour when 
/// clamping the rectangle within a rectangular area, than say an x,y,w,h version.
/// The fields aren't public so we can maintain the min/max relationship internally.
pub struct Rect {
    min: XY,
    max: XY,
}

impl Rect {
    pub fn new_xyxy(x1: X, y1: Y, x2: X, y2: Y) -> Self {
        Self::new(XY{x: x1, y: y1}, XY{x: x2, y: y2})
    }

    pub fn new(a: XY, b: XY) -> Self {
        Self {
            min: XY::minimum(a, b),
            max: XY::maximum(a, b),
        }
    }

    pub const fn min(&self) -> XY {
        self.min
    }

    pub const fn max(&self) -> XY {
        self.max
    }

    pub fn wh(&self) -> (W, H) {
        (
            w!(f32::from(self.max.x) - f32::from(self.min.x)),
            h!(f32::from(self.max.y) - f32::from(self.min.y)),
        )
    }
}

macro_rules! rect_xyxy {
    (
        $min_x: literal,
        $min_y: literal,
        $max_x: literal,
        $max_y: literal $(,)?
    ) => {{
        let a = XY{x: x!($min_x), y: y!($min_y)};
        let b = XY{x: x!($max_x), y: y!($max_y)};
        Rect {
            min: xy_minimum!(a, b),
            max: xy_maximum!(a, b),
        }
    }}
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

#[test]
fn this_same_distance_from_0_xy_case_produces_a_positive_wh_rect() {
    let (w, h) = rect_xyxy!(
        0.25,
        0.5,
        0.5,
        0.25,
    );

    assert!(w > unit::w!(0.0));
    assert!(h > unit::h!(0.0));
}

#[test]
fn this_same_distance_from_0_xy_case_produces_the_expected_normalized_rect() {
    let r = rect_xyxy!(
        0.25,
        0.5,
        0.5,
        0.25,
    );

    assert_eq!(r.min, XY{ x: bi_unit::x!(0.25), y: bi_unit::y!(0.25) });
    assert_eq!(r.max, XY{ x: bi_unit::x!(0.5), y: bi_unit::y!(0.5) });
}

pub const TILES_RECT: Rect = rect_xyxy!(
    -1.0,
    -1.0,
    1.0,
    1.0,
);

type PlayX = f32;
type PlayY = f32;
type PlayW = f32;
type PlayH = f32;

struct PlayXYWH {
    x: PlayX,
    y: PlayY,
    w: PlayW,
    h: PlayH,
}

pub type TileSideLength = f32;

struct Sizes {
    draw_wh: DrawWH,
    play_xywh: PlayXYWH,
    tile_side_length: TileSideLength,
}

const LEFT_UI_WIDTH_TILES: tile::Count = 9;
const RIGHT_UI_WIDTH_TILES: tile::Count = 9;
const DRAW_WIDTH_TILES: tile::Count = LEFT_UI_WIDTH + COORD_COUNT + RIGHT_UI_WIDTH;

fn fresh_sizes(wh: DrawWH) -> Sizes {
    screen_size tile = DrawW::min(
        wh.w / DRAW_WIDTH_TILES as DrawW,
        wh.h / COORD_COUNT as DrawH
    );

    screen_size play_area_w = tile * DRAW_WIDTH_TILES;
    screen_size play_area_h = tile * COORD_COUNT;
    screen_size play_area_x = (w - play_area_w) / 2;
    screen_size play_area_y = (h - play_area_h) / 2;

    struct sizes output = {
        .play_area_x = play_area_x,
        .play_area_y = play_area_y,
        .play_area_w = play_area_w,
        .play_area_h = play_area_h,
        .tile = tile,
    };

    return output;
}

fn tile_xy_to_draw(txy: tile::XY, draw_wh: DrawWH) -> DrawXY {
    let (w, h) = TILES_RECT.wh();
    let min = TILES_RECT.min();

    XY {
        x: min.x + w * txy.x.proportion(),
        y: min.y + h * txy.y.proportion(),
    }
}


#[derive(Clone, Copy, Debug)]
enum UiPos {
    Tile(tile::XY),
}

impl UiPos {
    fn xy(&self) -> XY {
        use UiPos::*;

        match self {
            Tile(txy) => {
                tile_xy_to_bi_unit(*txy)
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
    };

    // An amount of tiles, which are usually arranged in a line.
    pub type Count = u8;

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

    #[derive(Copy, Clone, Default, Debug)]
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
    }

    #[test]
    fn xy_all_looks_right() {
        let xy_all = XY::all();

        assert!(false, "{:?}", xy_all);
    }

    pub fn xy_to_i(xy: XY) -> usize {
        u8::from(xy.y.0) as usize * Coord::COUNT as usize
        + u8::from(xy.x.0) as usize
    }

    macro_rules! coord_def {
        ($( ($variants: ident => $number: literal) ),+ $(,)?) => {
            #[derive(Clone, Copy, Debug)]
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
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SpriteKind {
    Blank,
    Red,
    Green,
    Blue,
    Selectrum,
}

impl Default for SpriteKind {
    fn default() -> Self {
        Self::Blank
    }
}

/// A Tile should always be at a particular position, but that position should be 
/// derivable from the tiles location in the tiles array, so it doesn't need to be
/// stored. But, we often want to get the tile's data and it's location as a single
/// thing. This is why we have both `Tile` and `TileData`
type TileData = SpriteKind;

struct Tile {
    #[allow(unused)]
    xy: tile::XY,
    data: TileData
}

const TILES_LENGTH: usize = COORD_COUNT as usize * COORD_COUNT as usize;

#[derive(Clone, Debug)]
pub struct Tiles {
    tiles: [TileData; TILES_LENGTH],
}

fn get_tile(tiles: &Tiles, xy: tile::XY) -> Tile {
    Tile {
        xy,
        data: tiles.tiles[tile::xy_to_i(xy)]
    }
}

impl Default for Tiles {
    fn default() -> Self {
        Self {
            tiles: [TileData::default(); TILES_LENGTH]
        }
    }
}

#[derive(Debug, Default)]
struct Board {
    ui_pos: UiPos,
    tiles: Tiles
}

pub type DrawX = DrawLength;
pub type DrawY = DrawLength;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct DrawXY {
    pub x: DrawX,
    pub y: DrawY,
}

pub type DrawLength = f32;
pub type DrawW = DrawLength;
pub type DrawH = DrawLength;

#[derive(Debug, Default, PartialEq, Eq)]
pub struct DrawWH {
    pub w: DrawW,
    pub h: DrawH,
}

#[derive(Debug, Default)]
pub struct State {
    draw_wh: DrawWH,
    board: Board,
}

pub enum Command {
    Sprite(SpriteSpec),
}

pub struct SpriteSpec {
    pub sprite: SpriteKind,
    pub xy: DrawXY,
}

#[derive(Clone, Copy, Debug)]
pub enum Input {
    NoChange,
    Up,
    Down,
    Left,
    Right,
    Interact,
}

pub fn update(
    state: &mut State,
    commands: &mut dyn ClearableStorage<Command>,
    input: Input,
    draw_wh: DrawWH,
) {
    use Input::*;
    use UiPos::*;
    use Command::*;

    commands.clear();

    let mut interacted = false;

    match (input, &mut state.ui_pos) {
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
        (Interact, _) => {
            interacted = true;
        },
    }

    for xy in tile::XY::all() {
        let tile = get_tile(&state.tiles, xy);

        commands.push(Sprite(SpriteSpec{
            sprite: tile.data,
            xy: tile_xy_to_draw(xy)
        }));
    }

    if !interacted {
        commands.push(Sprite(SpriteSpec{
            sprite: SpriteKind::Selectrum,
            xy: state.ui_pos.xy(),
        }));
    }
}