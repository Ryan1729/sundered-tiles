// TODO stack string, assuming we realy care about no std
//#![no_std]
#![deny(unused)]

// In case we decide that we care about no_std/not allocating
type StrBuf = String;

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

pub use draw::{DrawLength, DrawX, DrawY, DrawXY, DrawW, DrawH, DrawWH};

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

    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Kind {
        Empty,
        Red(Visibility),
        RedStar(Visibility),
        Green(Visibility),
        GreenStar(Visibility),
        Blue(Visibility),
        BlueStar(Visibility),
        Goal(Visibility),
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
            Red(vis)
            | RedStar(vis)
            | Green(vis)
            | GreenStar(vis)
            | Blue(vis)
            | BlueStar(vis)
            | Goal(vis) => Some(vis),
        }
    }

    pub(crate) fn set_visibility(kind: Kind, vis: Visibility) -> Kind {
        use Kind::*;
        match kind {
            Empty => Empty,
            Red(_) => Red(vis),
            RedStar(_) => RedStar(vis),
            Green(_) => Green(vis),
            GreenStar(_) => GreenStar(vis),
            Blue(_) => Blue(vis),
            BlueStar(_) => Blue(vis),
            Goal(_) => Goal(vis),
        }
    }

    pub(crate) fn is_goal(kind: Kind) -> bool {
        matches!(kind, Kind::Goal(_))
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
}

#[derive(Clone, Copy, Debug)]
pub enum SpriteKind {
    Hidden,
    Red,
    Green,
    Blue,
    RedStar,
    GreenStar,
    BlueStar,
    InstrumentalGoal,
    TerminalGoal,
    Selectrum,
}

impl Default for SpriteKind {
    fn default() -> Self {
        Self::Hidden
    }
}

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

const TILES_LENGTH: u32 = COORD_COUNT as u32 * COORD_COUNT as u32;

#[derive(Clone, Debug)]
pub struct Tiles {
    tiles: [TileData; TILES_LENGTH as _],
}

impl Tiles {
    fn from_rng(rng: &mut Xs, level: Level) -> Self {
        let mut tiles = [TileData::default(); TILES_LENGTH as _];

        use tile::{Kind::*, Visibility::*};
        use Level::*;
        for i in 0..TILES_LENGTH as usize {
            let kind_max = match level {
                One => 200,
                Two => 100,
                Three => 50,
            };

            let kind = match xs_u32(rng, 0, kind_max) {
                1 => Red(Hidden),
                2 => Green(Hidden),
                3 => Blue(Hidden),
                _ => Empty,
            };

            tiles[i] = TileData {
                kind,
                ..<_>::default()
            };
        }

        macro_rules! set_random_tile {
            ($from: ident => $to: ident) => {
                let mut index = xs_u32(rng, 0, TILES_LENGTH) as usize;
                for _ in 0..TILES_LENGTH as usize {
                    if let $from(vis) = tiles[index].kind {
                        tiles[index].kind = $to(vis);
                        break
                    }

                    index += 1;
                    index %= TILES_LENGTH as usize;
                }
            }
        }

        set_random_tile!(Red => RedStar);
        set_random_tile!(Green => GreenStar);
        set_random_tile!(Blue => BlueStar);

        // TODO remove this slight non-uniformity?
        set_random_tile!(Red => Goal);

        Self {
            tiles
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

impl Default for Tiles {
    fn default() -> Self {
        Self {
            tiles: [TileData::default(); TILES_LENGTH as _]
        }
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

#[derive(Debug, Default)]
struct Board {
    ui_pos: UiPos,
    tiles: Tiles,
    rng: Xs,
    level: Level
}

impl Board {
    fn from_seed(seed: Seed, level: Level) -> Self {
        let mut rng = xs_from_seed(seed);

        let tiles = Tiles::from_rng(&mut rng, level);

        Self {
            rng,
            tiles,
            ..<_>::default()
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum InputMode {
    Standard,
    FastMovement
}

impl Default for InputMode {
    fn default() -> Self {
        Self::Standard
    }
}

#[derive(Debug, Default)]
pub struct State {
    sizes: draw::Sizes,
    board: Board,
    input_mode: InputMode,
}

impl State {
    pub fn from_seed(seed: Seed) -> Self {
        Self {
            board: Board::from_seed(seed, <_>::default()),
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

pub enum Command {
    Sprite(SpriteSpec),
    Text(TextSpec),
}

pub struct SpriteSpec {
    pub sprite: SpriteKind,
    pub xy: DrawXY,
}

pub struct TextSpec {
    pub text: StrBuf,
    pub xy: DrawXY,
}

pub type InputFlags = u16;

pub const INPUT_UP_PRESSED: InputFlags        = 0b00_0000_0001;
pub const INPUT_DOWN_PRESSED: InputFlags      = 0b00_0000_0010;
pub const INPUT_LEFT_PRESSED: InputFlags      = 0b00_0000_0100;
pub const INPUT_RIGHT_PRESSED: InputFlags     = 0b00_0000_1000;

pub const INPUT_UP_DOWN: InputFlags           = 0b00_0001_0000;
pub const INPUT_DOWN_DOWN: InputFlags         = 0b00_0010_0000;
pub const INPUT_LEFT_DOWN: InputFlags         = 0b00_0100_0000;
pub const INPUT_RIGHT_DOWN: InputFlags        = 0b00_1000_0000;

pub const INPUT_INTERACT_PRESSED: InputFlags  = 0b01_0000_0000;
pub const INPUT_FAST_PRESSED: InputFlags      = 0b10_0000_0000;

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
    fn from_flags(flags: InputFlags, input_mode: InputMode) -> Self {
        use Input::*;
        use InputMode::*;
        match input_mode {
            // We disallow Interact during FastMovement to prevent non-undoable 
            // mistakes
            FastMovement => if INPUT_UP_DOWN & flags != 0 {
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
    commands: &mut dyn ClearableStorage<Command>,
    input_flags: InputFlags,
    draw_wh: DrawWH,
) {
    use Input::*;
    use UiPos::*;
    use Command::*;

    if draw_wh != state.sizes.draw_wh {
        state.sizes = draw::fresh_sizes(draw_wh);
    }

    commands.clear();

    if INPUT_FAST_PRESSED & input_flags != 0 {
        state.input_mode = if let InputMode::FastMovement = state.input_mode {
            InputMode::Standard
        } else {
            InputMode::FastMovement
        };
    }

    let input = Input::from_flags(input_flags, state.input_mode);

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
            let mut tile = get_tile(&state.board.tiles, *xy);

            let started_visible = matches!(
                tile::get_visibility(tile.data.kind),
                Some(tile::Visibility::Shown)
            );

            tile.data.kind = tile::set_visibility(
                tile.data.kind,
                tile::Visibility::Shown
            );

            set_tile(&mut state.board.tiles, tile);

            if started_visible && tile::is_goal(tile.data.kind) {
                if is_last_level(state) {
                    for xy in tile::XY::all() {
                        set_tile(&mut state.board.tiles, crate::Tile {
                            xy,
                            data: TileData {
                                kind: tile::Kind::Goal(tile::Visibility::Shown),
                                ..<_>::default()
                            }
                        });
                    }
                } else {
                    let level = state.board.level;
                    state.board = Board::from_seed(
                        new_seed(&mut state.board.rng),
                        state.board.level,
                    );
                    state.board.level = next_level(level);
                }
            }
            
            interacted = true;
        },
    }

    for txy in tile::XY::all() {
        let tile = get_tile(&state.board.tiles, txy);

        let xy = draw::tile_xy_to_draw(&state.sizes, txy);

        use tile::{Kind::*, Visibility::*};

        let sprite = match tile.data.kind {
            Empty => continue,
            Red(Hidden)
            | RedStar(Hidden)
            | Green(Hidden)
            | GreenStar(Hidden)
            | Blue(Hidden)
            | BlueStar(Hidden)
            | Goal(Hidden) => SpriteKind::Hidden,
            Red(Shown) => SpriteKind::Red,
            RedStar(Shown) => SpriteKind::RedStar,
            Green(Shown) => SpriteKind::Green,
            GreenStar(Shown) => SpriteKind::GreenStar,
            Blue(Shown) => SpriteKind::Blue,
            BlueStar(Shown) => SpriteKind::BlueStar,
            Goal(Shown) => if is_last_level(state) {
                SpriteKind::TerminalGoal
            } else {
                SpriteKind::InstrumentalGoal
            },
        };

        commands.push(Sprite(SpriteSpec{
            sprite,
            xy
        }));
    }

    if !interacted {
        commands.push(Sprite(SpriteSpec{
            sprite: SpriteKind::Selectrum,
            xy: state.board.ui_pos.xy(&state.sizes),
        }));
    }

    commands.push(Text(TextSpec{
        text: format!("{:#?}", state.board.level),//format!("{:#?}", state.sizes),
        xy: DrawXY { x: 16., y: 16. },
    }));

    if let InputMode::FastMovement = state.input_mode {
        commands.push(Text(TextSpec{
            text: "Fast".to_owned(),
            xy: DrawXY { x: 16., y: state.sizes.draw_wh.h * 15. / 16. },
        }));
    }
}