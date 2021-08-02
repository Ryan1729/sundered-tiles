use super::*;
use tile::VisualKind;

fn generate_all_paths(
    from: tile::XY,
    to: tile::XY,
) -> Vec<Vec<tile::Dir>> {
    let (long_dir, short_dir) = tile::get_long_and_short_dir(from, to);

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
    // TODO: Make sure this whole function is not absurdly slow, as the first version
    // almost certainly is.
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
            
            assert_eq!(fast, slow);
        }}
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
        let visual_kind = VisualKind::from_rng(&mut rng);

        a!(&tiles, from, to, visual_kind);
    }
}