#![deny(unused)]

use core::{
    convert::TryInto,
};

extern crate alloc;
use alloc::vec::Vec;

struct Storage<A>(Vec<A>);

impl <A> game::ClearableStorage<A> for Storage<A> {
    fn clear(&mut self) {
        self.0.clear();
    }

    fn push(&mut self, a: A) {
        self.0.push(a);
    }
}

use macroquad::{
    Rect,
    Image,
    DrawTextureParams,
    KeyCode,
    Texture2D,
    clear_background,
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

const SPRITESHEET_BYTES: &[u8] = include_bytes!("../assets/spritesheet.png");

use image::{
    ImageFormat,
    ImageError,
};

use game::SpriteKind;

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

// TODO: make these a function of the screen size later?
const SCREEN_PIXELS_PER_TILE: f32 = 16.0;
const TILES_PER_SCREEN_PIXEL: f32 = 1.0 / SCREEN_PIXELS_PER_TILE;
const SPRITE_PIXELS_PER_TILE_SIDE: f32 = 128.0;


#[macroquad::main("Sundered Tiles")]
async fn main() {
    let spritesheet_texture: Texture2D = load_spritesheet()
        .expect("Embedded spritesheet could not be loaded!");

    let mut state = game::State::default();
    let mut commands = Storage(Vec::with_capacity(1024));

    // generate the commands for the first frame
    game::update(&mut state, &mut commands, game::Input::NoChange);

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
            game::update(&mut state, &mut commands, input);
        }

        clear_background(BLACK);

        let s_width = screen_width();
        let s_height = screen_height();
        let tile_dest_size: Vec2 = Vec2::new(
            TILES_PER_SCREEN_PIXEL * s_width,
            TILES_PER_SCREEN_PIXEL * s_height,
        );

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
                    let (x, y) = (
                        s_width * (f32::from(s.xy.x) + 1.0) / 2.0,
                        s_height * (f32::from(s.xy.y) + 1.0) / 2.0,
                    );

                    let (source_x, source_y) = source_coords(s.sprite);

                    draw_texture_ex(
                        spritesheet_texture,
                        x,
                        y,
                        WHITE,
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
                // Later we'll want Text at the very least.
            }
        }

        next_frame().await
    }
}

fn source_coords(sprite: SpriteKind) -> (f32, f32) {
    use SpriteKind::*;

    let sx = match sprite {
        Blank => 0.0,
        Red => 1.0,
        Green => 2.0,
        Blue => 3.0,
        Selectrum => 15.0,
    };

    (
        sx * SPRITE_PIXELS_PER_TILE_SIDE,
        0.0,
    )
}
