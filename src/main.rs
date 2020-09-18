use core::{
    convert::TryInto,
    error::Error,
};

use alloc::vec::Vec;

use macroquad::{
    Image,
    DrawTextureParams,
    KeyCode,
    Texture2D,
    clear_background,
    draw_texture,
    load_texture_from_image,
    next_frame,
    screen_height,
    screen_width,
    is_key_pressed,
    BLACK,
    WHITE,
};

const SPRITESHEET_BYTES: &[u8] = include_bytes!("../assets/spritesheet.png");

use image::{
    ImageFormat
};

type Res<A> = Result<A, Box<dyn Error>>;

fn load_spritesheet() -> Res<Texture2D> {
    let rbga = image::load_from_memory_with_format(
        SPRITESHEET_BYTES,
        ImageFormat::Png
    )?.into_rgba();

    let (width, height) = rbga.dimensions();

    let img = Image {
        bytes: rbga.into_raw(),
        width: width.try_into()?,
        height: height.try_into()?,
    };

    Ok(load_texture_from_image(&img))
}

// TODO: make these a function of the screen size later?
const PIXELS_PER_TILE: f32 = 16.0;
const TILES_PER_PIXEL: f32 = 1.0 / PIXELS_PER_TILE;

#[macroquad::main("Sundered Tiles")]
async fn main() {
    let spritesheet_texture: Texture2D = load_spritesheet()
        .expect("Embedded spritesheet could not be loaded!");

    let mut state = game::State::default();
    let mut commands = Vec::with_cacpacity(1024);

    loop {
        let input;
        {
            use game::Input::*;

            input = if is_key_pressed(KeyCode::Space) || is_key_pressed(KeyCode::Enter) {
                Some(Interact)
            } else if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
                Some(Up)
            } else if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
                Some(Down)
            } else if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
                Some(Left)
            } else if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
                Some(Right)
            } else {
                None
            };
        }

        if let Some(input) = input {
            game::update_and_render(&mut state, &mut commands, input);
        }

        clear_background(BLACK);

        let s_width = screen_width();
        let s_height = screen_height();
        let tile_dest_size: Vec2 = Vec2::new(
            TILES_PER_SCREEN_PIXEL * s_width,
            TILES_PER_SCREEN_PIXEL * s_height,
        );

        let tile_base_source_rect = Rect {
            w: SPRITE_PIXELS_PER_TILE,
            h: SPRITE_PIXELS_PER_TILE,
            ..<_>::default()
        };

        for cmd in commands.iter() {
            use game::Command::*;
            match cmd {
                Sprite(s) => {
                    let (x, y) = (
                        s_width * (f32::from(s.x) + 1.0) / 2.0,
                        s_height * (f32::from(s.y) + 1.0) / 2.0,
                    );

                    draw_texture_ex(
                        spritesheet_texture,
                        x,
                        y,
                        WHITE,
                        DrawTextureParams {
                            dest_size: Some(tile_dest_size),
                            source: Some(Rect {
                                x: s.sprite * SPRITE_PIXELS_PER_TILE,
                                y: 0.0,
                                ..<_>::default()
                            })
                            ..<_>::default()
                        }
                    );
                }
                // Later we'll want Text at the very least.
            }
        }

        next_frame().await
    }
}
