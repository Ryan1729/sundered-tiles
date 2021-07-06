use super::*;

#[test]
fn goal_is_one_down_one_right_produces_the_expected_hints() {
    use SpriteKind::*;
    use tile::{HintSpec::*, RelativeDelta::*};

    let tile_array = [TileData::default(); TILES_LENGTH as _];

    let (_, sprites) = render_hint_spec(
        &tile_array,
        GoalIs(OneDownOneRight),
        InstrumentalGoal,
        <_>::default(),
    );

    assert_eq!(EdgeDownRight, sprites[hint::UP_LEFT_INDEX].expect("UP_LEFT_INDEX"));
}

#[test]
fn goal_is_one_down_one_left_produces_the_expected_hint_spec() {
    use SpriteKind::*;
    use tile::{HintSpec::*, RelativeDelta::*};

    let tile_array = [TileData::default(); TILES_LENGTH as _];
    let goal_xy = tile::XY{
        x: tile::X::MAX,
        y: tile::Y::ZERO,
    };

    let (_, sprites) = render_hint_spec(
        &tile_array,
        GoalIs(OneDownOneLeft),
        InstrumentalGoal,
        goal_xy,
    );

    assert_eq!(EdgeDownLeft, sprites[hint::UP_RIGHT_INDEX].expect("UP_RIGHT_INDEX"));
}

#[test]
fn goal_is_one_up_one_left_produces_the_expected_hint_spec_in_the_max_corner() {
    use SpriteKind::*;
    use tile::{HintSpec::*, RelativeDelta::*};

    let tile_array = [TileData::default(); TILES_LENGTH as _];
    let goal_xy = tile::XY{
        x: tile::X::MAX,
        y: tile::Y::MAX,
    };

    let (_, sprites) = render_hint_spec(
        &tile_array,
        GoalIs(OneUpOneLeft),
        InstrumentalGoal,
        goal_xy,
    );

    assert_eq!(EdgeUpLeft, sprites[hint::DOWN_RIGHT_INDEX].expect("DOWN_RIGHT_INDEX"));
}

#[test]
fn goal_is_one_up_one_left_produces_the_expected_hint_spec_on_the_x_max_edge() {
    use SpriteKind::*;
    use tile::{HintSpec::*, RelativeDelta::*};

    let tile_array = [TileData::default(); TILES_LENGTH as _];
    let goal_xy = tile::XY {
        x: tile::X::MAX,
        y: tile::Y::CENTER,
    };

    let (_, sprites) = render_hint_spec(
        &tile_array,
        GoalIs(OneUpOneLeft),
        InstrumentalGoal,
        goal_xy,
    );

    assert_eq!(EdgeUpLeft, sprites[hint::DOWN_RIGHT_INDEX].expect("DOWN_RIGHT_INDEX"));
}

#[test]
fn goal_is_two_up_one_left_produces_the_expected_hint_spec() {
    use SpriteKind::*;
    use tile::{HintSpec::*, RelativeDelta::*};

    let tile_array = [TileData::default(); TILES_LENGTH as _];
    let goal_xy = tile::XY{
        x: tile::X::MAX,
        y: tile::Y::MAX,
    };

    let (_, sprites) = render_hint_spec(
        &tile_array,
        GoalIs(TwoUpOneLeft),
        InstrumentalGoal,
        goal_xy,
    );

    assert_eq!(EdgeUpLeft, sprites[hint::TWO_DOWN_ONE_RIGHT_INDEX].expect("TWO_DOWN_ONE_RIGHT_INDEX"));
}

#[test]
fn goal_is_one_down_two_left_produces_the_expected_hint_spec() {
    use SpriteKind::*;
    use tile::{HintSpec::*, RelativeDelta::*};

    let tile_array = [TileData::default(); TILES_LENGTH as _];
    let goal_xy = tile::XY{
        x: tile::X::MAX,
        y: tile::Y::ZERO,
    };

    let (_, sprites) = render_hint_spec(
        &tile_array,
        GoalIs(OneDownTwoLeft),
        InstrumentalGoal,
        goal_xy,
    );

    assert_eq!(EdgeDownLeft, sprites[hint::UP_TWO_RIGHT_INDEX].expect("UP_TWO_RIGHT_INDEX"));
}