use super::{Storage, source_coords, SPRITE_PIXELS_PER_TILE_SIDE, SPRITESHEET_BYTES};

use core::{
    convert::TryInto,
};

use macroquad::{
    Color,
    Rect,
    Image,
    DrawTextureParams,
    KeyCode,
    Texture2D,
    clear_background,
    draw_rectangle_lines,
    draw_text,
    draw_texture_ex,
    load_texture_from_image,
    next_frame,
    screen_height,
    screen_width,
    is_key_pressed,
    Vec2,
    BLACK,
    WHITE,
};

use image::{
    ImageFormat,
    ImageError,
};

#[derive(Debug)]
enum E {
    ImageError(ImageError),
    TryFromIntError(core::num::TryFromIntError),
}

fn load_spritesheet() -> Result<Texture2D, E> {
    let rbga = image::load_from_memory_with_format(
        SPRITESHEET_BYTES,
        ImageFormat::Png
    ).map_err(E::ImageError)?.into_rgba();

    let (width, height) = rbga.dimensions();

    let img = Image {
        bytes: rbga.into_raw(),
        width: width.try_into().map_err(E::TryFromIntError)?,
        height: height.try_into().map_err(E::TryFromIntError)?,
    };

    Ok(load_texture_from_image(&img))
}

fn draw_wh() -> game::DrawWH {
    game::DrawWH {
        w: screen_width(),
        h: screen_height(),
    }
}

pub async fn inner_main() {
    let spritesheet_texture: Texture2D = load_spritesheet()
        .expect("Embedded spritesheet could not be loaded!");

    let mut state = game::State::default();
    let mut commands = Storage(Vec::with_capacity(1024));

    // generate the commands for the first frame
    game::update(&mut state, &mut commands, game::Input::NoChange, draw_wh());

    const TEXT: Color = WHITE;
    const NO_TINT: Color = WHITE;
    const OUTLINE: Color = WHITE;

    loop {
        let input;
        {
            use game::Input::*;

            input = if is_key_pressed(KeyCode::Space) || is_key_pressed(KeyCode::Enter) {
                Interact
            } else if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
                Up
            } else if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
                Down
            } else if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
                Left
            } else if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
                Right
            } else {
                NoChange
            };
        }

        game::update(&mut state, &mut commands, input, draw_wh());

        clear_background(BLACK);

        let sizes = game::sizes(&state);

        // the -1 and +2 business makes the border lie just outside the actual
        // play area
        draw_rectangle_lines(
            sizes.play_xywh.x - 1.,
            sizes.play_xywh.y - 1.,
            sizes.play_xywh.w + 2.,
            sizes.play_xywh.h + 2.,
            5.,
            OUTLINE
        );

        let tile_dest_size: Vec2 = {
            let side_length = game::sizes(&state).tile_side_length;

            Vec2::new(
                side_length,
                side_length
            )
        };

        let tile_base_source_rect = Rect {
            x: 0.,
            y: 0.,
            w: SPRITE_PIXELS_PER_TILE_SIDE,
            h: SPRITE_PIXELS_PER_TILE_SIDE,
        };

        for cmd in commands.0.iter() {
            use game::Command::*;
            match cmd {
                Sprite(s) => {
                    let (source_x, source_y) = source_coords(s.sprite);

                    draw_texture_ex(
                        spritesheet_texture,
                        s.xy.x,
                        s.xy.y,
                        NO_TINT,
                        DrawTextureParams {
                            dest_size: Some(tile_dest_size),
                            source: Some(Rect {
                                x: source_x,
                                y: source_y,
                                ..tile_base_source_rect
                            }),
                            ..<_>::default()
                        }
                    );
                }
                Text(t) => {
                    draw_text(
                        &t.text,
                        t.xy.x,
                        t.xy.y,
                        40.,
                        TEXT
                    );
                }
            }
        }

        next_frame().await
    }
}