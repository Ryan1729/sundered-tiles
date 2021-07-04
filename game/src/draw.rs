#![deny(unused)]

use crate::tile;

// In case we decide that we care about no_std/not allocating
type StrBuf = String;

type PlayX = DrawLength;
type PlayY = DrawLength;
type PlayW = DrawLength;
type PlayH = DrawLength;

#[derive(Clone, Debug, Default)]
pub struct PlayXYWH {
    pub x: PlayX,
    pub y: PlayY,
    pub w: PlayW,
    pub h: PlayH,
}

type BoardX = DrawLength;
type BoardY = DrawLength;
type BoardW = DrawLength;
type BoardH = DrawLength;

#[derive(Clone, Debug, Default)]
pub struct BoardXYWH {
    pub x: BoardX,
    pub y: BoardY,
    pub w: BoardW,
    pub h: BoardH,
}

pub type DrawX = DrawLength;
pub type DrawY = DrawLength;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DrawXY {
    pub x: DrawX,
    pub y: DrawY,
}

impl core::ops::Add for DrawXY {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl core::ops::AddAssign for DrawXY {
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
        };
    }
}

pub type DrawLength = f32;
pub type DrawW = DrawLength;
pub type DrawH = DrawLength;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct DrawWH {
    pub w: DrawW,
    pub h: DrawH,
}

pub type TileSideLength = DrawLength;

#[derive(Clone, Debug, Default)]
pub struct Sizes {
    pub draw_wh: DrawWH,
    pub play_xywh: PlayXYWH,
    pub board_xywh: BoardXYWH,
    pub tile_side_length: TileSideLength,
}

const LEFT_UI_WIDTH_TILES: tile::Count = 9;
const RIGHT_UI_WIDTH_TILES: tile::Count = 9;
const DRAW_WIDTH_TILES: tile::Count = LEFT_UI_WIDTH_TILES 
    + tile::Coord::COUNT 
    + RIGHT_UI_WIDTH_TILES;

pub fn fresh_sizes(wh: DrawWH) -> Sizes {
    let w_length_bound = wh.w / DRAW_WIDTH_TILES as DrawW;
    let h_length_bound = wh.h / tile::Coord::COUNT as DrawH;

    let (raw_bound, tile_side_length, board_x_offset, board_y_offset) = {
        if w_length_bound == h_length_bound {
            (h_length_bound, h_length_bound.trunc(), h_length_bound.fract() / 2., h_length_bound.fract() / 2.)
        } else if w_length_bound > h_length_bound {
            (h_length_bound, h_length_bound.trunc(), 0., h_length_bound.fract() / 2.)
        } else if w_length_bound < h_length_bound {
            (w_length_bound, w_length_bound.trunc(), w_length_bound.fract() / 2., 0.)
        } else {
            // NaN ends up here
            // TODO return a Result? Panic? Take only known non-NaN values?
            (100., 100., 0., 0.)
        }
    };

    let play_area_w = raw_bound * DRAW_WIDTH_TILES as PlayW;
    let play_area_h = raw_bound * tile::Coord::COUNT as PlayH;
    let play_area_x = (wh.w - play_area_w) / 2.;
    let play_area_y = (wh.h - play_area_h) / 2.;

    let board_area_w = tile_side_length * tile::Coord::COUNT as BoardW;
    let board_area_h = tile_side_length * tile::Coord::COUNT as BoardH;
    let board_area_x = play_area_x + board_x_offset + (play_area_w - board_area_w) / 2.;
    let board_area_y = play_area_y + board_y_offset + (play_area_h - board_area_h) / 2.;

    Sizes {
        draw_wh: wh,
        play_xywh: PlayXYWH {
            x: play_area_x,
            y: play_area_y,
            w: play_area_w,
            h: play_area_h,
        },
        board_xywh: BoardXYWH {
            x: board_area_x,
            y: board_area_y,
            w: board_area_w,
            h: board_area_h,
        },
        tile_side_length,
    }
}

#[test]
fn fresh_sizes_produces_the_expected_tile_size_in_these_symmetric_cases() {
    assert_eq!(
        fresh_sizes(DrawWH{w: 1366., h: 768.}).tile_side_length,
        // AKA the largest integer tile length that will fit
        750. / tile::Coord::COUNT as DrawLength
    );

    assert_eq!(
        fresh_sizes(DrawWH{w: 768., h: 1366.}).tile_side_length,
        // AKA the largest integer tile length that will fit
        (11. * DRAW_WIDTH_TILES as DrawLength) / DRAW_WIDTH_TILES as DrawLength 
    );
}

pub fn tile_xy_to_draw(sizes: &Sizes, txy: tile::XY) -> DrawXY {
    DrawXY {
        x: sizes.board_xywh.x + sizes.board_xywh.w * txy.x.proportion(),
        y: sizes.board_xywh.y + sizes.board_xywh.h * txy.y.proportion(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    RulerEnd,
    Hint,
    EdgeUp,
    EdgeDown,
    EdgeLeft,
    EdgeRight,
    QuestionMark,
    RedGreen,
    GreenBlue,
    BlueRed,
    GoalDistanceHint,
    EdgeUpLeft,
    EdgeUpRight,
    EdgeDownLeft,
    EdgeDownRight,
}

impl Default for SpriteKind {
    fn default() -> Self {
        Self::Hidden
    }
}

pub(crate) fn sprite_kind_from_tile_kind(
    kind: tile::Kind,
    goal_sprite: SpriteKind,
) -> Option<SpriteKind> {
    use tile::{Kind::*, Visibility::*, DistanceIntel::*, PrevNext::*};
    let sprite_kind = match kind {
        Empty => return None,
        Red(Hidden, _)
        | RedStar(Hidden)
        | Green(Hidden, _)
        | GreenStar(Hidden)
        | Blue(Hidden, _)
        | BlueStar(Hidden)
        | Goal(Hidden)
        | Hint(Hidden, _)
        | GoalDistance(Hidden, _) => SpriteKind::Hidden,
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
        GoalDistance(Shown, _) => SpriteKind::GoalDistanceHint,
    };

    Some(sprite_kind)
}

pub(crate) fn shown_sprite_kind_from_tile_kind(
    kind: tile::Kind,
    goal_sprite: SpriteKind,
) -> Option<SpriteKind> {
    use tile::{Kind::*, DistanceIntel::*, PrevNext::*};
    let sprite_kind = match kind {
        Empty => return None,
        Red(_, PartialColour(Prev)) => SpriteKind::BlueRed,
        Red(_, PartialColour(Next)) => SpriteKind::RedGreen,
        Red(_, _) => SpriteKind::Red,
        RedStar(_) => SpriteKind::RedStar,
        Green(_, PartialColour(Prev)) => SpriteKind::RedGreen,
        Green(_, PartialColour(Next)) => SpriteKind::GreenBlue,
        Green(_, _) => SpriteKind::Green,
        GreenStar(_) => SpriteKind::GreenStar,
        Blue(_, PartialColour(Prev)) => SpriteKind::GreenBlue,
        Blue(_, PartialColour(Next)) => SpriteKind::BlueRed,
        Blue(_, _) => SpriteKind::Blue,
        BlueStar(_) => SpriteKind::BlueStar,
        Goal(_) => goal_sprite,
        Hint(_, _) => SpriteKind::Hint,
        GoalDistance(_, _) => SpriteKind::GoalDistanceHint,
    };

    Some(sprite_kind)
}

#[derive(Debug)]
pub enum Command {
    Sprite(SpriteSpec),
    Text(TextSpec),
}

#[derive(Debug)]
pub struct SpriteSpec {
    pub sprite: SpriteKind,
    pub xy: DrawXY,
}

/// This is provided to make font selection etc easier for platform layers.
#[derive(Debug)]
pub enum TextKind {
    DistanceMarker,
    Level,
    Digs,
    HintString,
    Fast,
    Ruler,
}

#[derive(Debug)]
pub struct TextSpec {
    pub text: StrBuf,
    pub xy: DrawXY,
    /// We'd rather define a rectangle for the text to (hopefully) lie inside than
    /// a font size directly.
    pub wh: DrawWH,
    pub kind: TextKind,
}
