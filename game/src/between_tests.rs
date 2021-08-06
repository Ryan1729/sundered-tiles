use super::*;
use tile::{VisualKind, Dir, XY};

fn get_long_and_short_dir(
    from: XY,
    to: XY
) -> (Dir, Dir) {
    //FIXME account for xs and ys.
    let from_x = usize::from(from.x);
    let from_y = usize::from(from.y);
    let to_x = usize::from(to.x);
    let to_y = usize::from(to.y);

    let x_distance = ((from_x as isize) - (to_x as isize)).abs() as usize;
    let y_distance = ((from_y as isize) - (to_y as isize)).abs() as usize;
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

fn generate_all_paths(
    from: tile::XY,
    to: tile::XY,
) -> Vec<Vec<tile::Dir>> {
    let (long_dir, short_dir) = get_long_and_short_dir(from, to);
    let distance = tile::manhattan_distance(from, to);
    // If we don't add a limit somewhere this will allocate way too much memory.
    // Also, 64k paths ought to be enough for anybody!
    assert!(distance <= 16, "distance: {}", distance);
    let two_to_the_distance = 1 << (distance as u64);
    // Yes this is O(2^n). Yes we will all but certainly need to replace this.
    
    let mut output = Vec::with_capacity(two_to_the_distance as usize);
    'outer: for possibility in 0..two_to_the_distance {
        let mut path = Vec::with_capacity(distance as usize);
        let mut xy = from;
        for bit_index in 0..distance {
            let bit = (possibility >> bit_index) & 1;

            let dir = match bit {
                0 => long_dir,
                _ => short_dir
            };

            match tile::apply_dir(dir, xy) {
                Some(new_xy) => {
                    xy = new_xy;
                },
                None => {
                    continue 'outer;
                }
            }
            path.push(dir);
        }

        if xy == to {
            output.push(path);
        }   
    }

    output
}

fn minimum_between_of_visual_kind_slow(
    tiles: &Tiles,
    xy_a: tile::XY,
    xy_b: tile::XY,
    visual_kind: tile::VisualKind
) -> MinimumOutcome {
    if !&tiles.tiles.iter().map(|tile_data| {
        tile::VisualKind::from(tile_data.kind)
    }).any(|v_k| v_k == visual_kind) {
        return MinimumOutcome::NoMatchingTiles;
    }

    let mut minimum = tile::Count::max_value();

    let all_paths = generate_all_paths(
        xy_a,
        xy_b,
    );

    'outer: for path in all_paths {
        let mut current_count = 0;
        let mut xy = xy_a;
        for dir in path {
            let xy_opt = tile::apply_dir(dir, xy);
            match xy_opt {
                Some(new_xy) => {
                    xy = new_xy
                },
                None => {
                    continue 'outer;
                }
            }

            if visual_kind == get_tile_visual_kind(tiles, xy) {
                current_count += 1;
            }
        }

        if current_count < minimum {
            minimum = current_count;
        }
    }

    MinimumOutcome::Count(minimum)
}

mod minimum_between_of_visual_kind_matches_slow_version {
    use super::*;

    // Short for assert. We can be this brief becasue this is local to this module
    macro_rules! a {
        ($tiles: expr, $from: expr, $to: expr, $visual_kind: expr) => {{
            use std::time::Instant;

            let slow_start = Instant::now();
            let slow = minimum_between_of_visual_kind_slow(
                $tiles,
                $from,
                $to,
                $visual_kind,
            );
            let slow_end = Instant::now();

            let fast_start = Instant::now();
            let fast = minimum_between_of_visual_kind(
                $tiles,
                $from,
                $to,
                $visual_kind,
            );
            let fast_end = Instant::now();

            assert_eq!(fast, slow, "mismatch when looking for {:?}", $visual_kind);

            let fast_duration = fast_end.duration_since(fast_start);
            let slow_duration = slow_end.duration_since(slow_start);

            // Without the margin added to the slow duration, this fails sometimes,
            // even with identical implementations. I guess this is due to cache 
            // effects, and/or OS task switching or something?
            assert!(
                fast_duration <= (slow_duration + (slow_duration/20)),
                "{} > {}",
                fast_duration.as_nanos(),
                slow_duration.as_nanos()
            );
        }}
    }

    macro_rules! xy {
        ($x: literal, $y: literal) => {{
            let mut x = tile::X::default();
            for _ in 0..$x {
                x = x.checked_add_one().unwrap();
            }

            let mut y = tile::Y::default();
            for _ in 0..$y {
                y = y.checked_add_one().unwrap();
            }

            tile::XY {
                x,
                y,
            }
        }};
    }

    #[test]
    fn on_this_random_example() {
        let mut rng = xs_from_seed([
            0xb, 0xee, 0xfa, 0xce,
            0xb, 0xee, 0xfa, 0xce,
            0xb, 0xee, 0xfa, 0xce,
            0xb, 0xee, 0xfa, 0xce,
        ]);

        let tiles = Tiles::from_rng(&mut rng, <_>::default());

        let from = tile::XY::from_rng(&mut rng);
        let to = tile::XY::from_rng(&mut rng);

        a!(&tiles, from, to, VisualKind::Empty);
        a!(&tiles, from, to, VisualKind::ALL[1]);
    }

    #[test]
    fn on_this_instructive_set_of_examples() {
        let mut tiles = Tiles::default();

        let wanted_tile_data = TileData {
            kind: tile::Kind::Red(<_>::default(), <_>::default(), <_>::default()),
        };

        tiles.tiles[tile::xy_to_i(xy!(2, 0))] = wanted_tile_data;
        tiles.tiles[tile::xy_to_i(xy!(1, 1))] = wanted_tile_data;
        tiles.tiles[tile::xy_to_i(xy!(0, 2))] = wanted_tile_data;

        let from = tile::XY::default();
/*
        let to = xy!(2, 2);

        a!(&tiles, from, to, VisualKind::Red);
*/
        let to = xy!(3, 2);

        a!(&tiles, from, to, VisualKind::Red);
/*
        let to = xy!(5, 6);

        a!(&tiles, from, to, VisualKind::Red);*/
    }

    #[test]
    fn on_this_non_down_right_example() {
        let mut tiles = Tiles::default();

        let wanted_tile_data = TileData {
            kind: tile::Kind::Red(<_>::default(), <_>::default(), <_>::default()),
        };

        tiles.tiles[tile::xy_to_i(xy!(1, 1))] = wanted_tile_data;

        let from = xy!(1, 1);
        let to = xy!(0, 0);

        a!(&tiles, from, to, VisualKind::Empty);
    }

    #[test]
    fn on_this_no_matching_tiles_example() {
        let tiles = Tiles::default();

        let from = xy!(0, 0);
        let to = xy!(1, 1);

        a!(&tiles, from, to, VisualKind::ALL[1]);
    }
}