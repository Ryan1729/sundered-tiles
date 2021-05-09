//#![deny(unused)]

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

const SAMPLING_SHADER: &str = include_str!("../assets/sampling.fs");

const SPRITESHEET_BYTES: &[u8] = include_bytes!("../assets/spritesheet.png");

const SPRITE_PIXELS_PER_TILE_SIDE: f32 = 128.0;

use game::SpriteKind;

fn source_coords(sprite: SpriteKind) -> (f32, f32) {
    use SpriteKind::*;

    let sx = match sprite {
        Hidden => 0.0,
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

#[cfg(not(any(feature = "platform-macroquad", feature = "platform-raylib-rs")))]
core::compile_error!("You must specify one of \"platform-macroquad\" or \"platform-raylib-rs\"");

#[cfg(all(feature = "platform-macroquad", feature = "platform-raylib-rs"))]
core::compile_error!("You must specify only one of \"platform-macroquad\" or \"platform-raylib-rs\"");

/// If we do the obvious thing,
/// ```rust
/// #[cfg(feature = "platform-macroquad")]
/// #[macroquad::main("Sundered Tiles")]
/// async fn main() {
///     //...
/// }
/// ``` 
/// then we get an error saying "custom attribute panicked". 
#[cfg(feature = "platform-macroquad")]
macro_rules! macroquad_wrapper_macro {
    () => {
        #[macroquad::main("Sundered Tiles")]
        async fn main() {
            macroquad_platform::inner_main().await;
        }
    }
}

#[cfg(feature = "platform-macroquad")]
macroquad_wrapper_macro!();

#[cfg(feature = "platform-macroquad")]
mod macroquad_platform {
    use super::{Storage, source_coords, SPRITE_PIXELS_PER_TILE_SIDE, SPRITESHEET_BYTES};

    use core::{
        convert::TryInto,
    };
    
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
}

#[cfg(feature = "platform-raylib-rs")]
fn main() {
    raylib_rs_platform::inner_main();
}

#[cfg(feature = "platform-raylib-rs")]
mod raylib_rs_platform {
    use super::{
        Storage,
        source_coords,
        SPRITE_PIXELS_PER_TILE_SIDE,
        SPRITESHEET_BYTES,
        SAMPLING_SHADER
    };
    use raylib::prelude::{
        *,
        KeyboardKey::*,
        ffi::LoadImageFromMemory,
        core::{
            drawing::{RaylibTextureModeExt, RaylibShaderModeExt},
            logging
        }
    };

    use ::core::{
        convert::TryInto,
    };

    fn draw_wh(rl: &RaylibHandle) -> game::DrawWH {
        game::DrawWH {
            w: rl.get_screen_width() as game::DrawW,
            h: rl.get_screen_height() as game::DrawH,
        }
    }

    pub fn inner_main() {
        let (mut rl, thread) = raylib::init()
        .size(0, 0)
        .resizable()
        .title("Sundered Tiles")
        .build();

        if cfg!(debug_assertions) {
            logging::set_trace_log_exit(TraceLogType::LOG_WARNING);
        }

        rl.set_target_fps(60);
        rl.toggle_fullscreen();

        let spritesheet_img = {
            let byte_count: i32 = SPRITESHEET_BYTES.len()
                .try_into()
                .expect("(2^31)-1 bytes ought to be enough for anybody!");

            let bytes = SPRITESHEET_BYTES.as_ptr();

            let file_type = b"PNG\0" as *const u8 as *const i8;

            unsafe {
                Image::from_raw(LoadImageFromMemory(
                    file_type,
                    bytes,
                    byte_count
                ))
            }
        };

        let spritesheet = rl.load_texture_from_image(
            &thread,
            &spritesheet_img
        ).expect(
            "Embedded spritesheet could not be loaded!"
        );

        let grid_shader = rl.load_shader_code(
            &thread,
            None,
            Some(SAMPLING_SHADER)
        );

        // This seems like a safe texture size, with wide GPU support.
        // TODO What we should do is query GL_MAX_TEXTURE_SIZE and figure
        // out what to do if we get a smaller value than this.
//        const RENDER_TARGET_SIZE: u32 = 8192;
        // On the other hand, 8192 makes my old intergrated graphics laptop overheat
        const RENDER_TARGET_SIZE: u32 = 2048;

        // We'll let the OS reclaim the memory when the game closes.
        let mut render_target = rl.load_render_texture(
            &thread,
            RENDER_TARGET_SIZE,
            RENDER_TARGET_SIZE
        ).unwrap();

        let seed: u128 = {
            use std::time::SystemTime;
    
            let duration = match 
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
            {
                Ok(d) => d,
                Err(err) => err.duration(),
            };
    
            duration.as_nanos()
        };
    
        println!("{}", seed);

        let mut state = game::State::from_seed(seed.to_le_bytes());
        let mut commands = Storage(Vec::with_capacity(1024));

        // generate the commands for the first frame
        game::update(&mut state, &mut commands, game::Input::NoChange, draw_wh(&rl));

        const BACKGROUND: Color = Color{ r: 0x22, g: 0x22, b: 0x22, a: 255 };
        const WHITE: Color = Color{ r: 0xee, g: 0xee, b: 0xee, a: 255 };
        const TEXT: Color = WHITE;
        const NO_TINT: Color = WHITE;
        const OUTLINE: Color = WHITE;

        const sprite_border: f32 = 4.;

        while !rl.window_should_close() {
            if rl.is_key_pressed(KEY_F11) {
                rl.toggle_fullscreen();
            }

            let input;
            {
                use game::Input::*;
    
                input = if rl.is_key_pressed(KEY_SPACE) || rl.is_key_pressed(KEY_ENTER) {
                    Interact
                } else if rl.is_key_pressed(KEY_UP) || rl.is_key_pressed(KEY_W) {
                    Up
                } else if rl.is_key_pressed(KEY_DOWN) || rl.is_key_pressed(KEY_S) {
                    Down
                } else if rl.is_key_pressed(KEY_LEFT) || rl.is_key_pressed(KEY_A) {
                    Left
                } else if rl.is_key_pressed(KEY_RIGHT) || rl.is_key_pressed(KEY_D) {
                    Right
                } else {
                    NoChange
                };
            }

            game::update(
                &mut state,
                &mut commands,
                input,
                draw_wh(&rl)
            );

            let screen_render_rect = Rectangle {
                x: 0.,
                y: 0.,
                width: rl.get_screen_width() as _,
                height: rl.get_screen_height() as _
            };

            let sizes = game::sizes(&state);

            let mut d = rl.begin_drawing(&thread);

            d.clear_background(BACKGROUND);

            {
                let mut texture_d = d.begin_texture_mode(
                    &thread,
                    &mut render_target
                );

                let mut shader_d = texture_d.begin_shader_mode(
                    &grid_shader
                );

                shader_d.clear_background(BACKGROUND);

                // the -1 and +2 business makes the border lie just outside the actual
                // play area
                shader_d.draw_rectangle_lines(
                    sizes.play_xywh.x as i32 - 1,
                    sizes.play_xywh.y as i32 - 1,
                    sizes.play_xywh.w as i32 + 2,
                    sizes.play_xywh.h as i32 + 2,
                    OUTLINE
                );
            
                let tile_base_source_rect = Rectangle {
                    x: 0.,
                    y: 0.,
                    width: SPRITE_PIXELS_PER_TILE_SIDE - sprite_border * 2.,
                    height: SPRITE_PIXELS_PER_TILE_SIDE - sprite_border * 2.,
                };
    
                let tile_base_render_rect = Rectangle {
                    x: 0.,
                    y: 0.,
                    width: sizes.tile_side_length,
                    height: sizes.tile_side_length,
                };
    
                for cmd in commands.0.iter() {
                    use game::Command::*;
                    match cmd {
                        Sprite(s) => {
                            let (source_x, source_y) = source_coords(s.sprite);
        
                            shader_d.draw_texture_pro(
                                &spritesheet,
                                Rectangle {
                                    x: source_x + sprite_border,
                                    y: source_y + sprite_border,
                                    ..tile_base_source_rect
                                },
                                Rectangle {
                                    x: s.xy.x,
                                    y: s.xy.y,
                                    ..tile_base_render_rect
                                },
                                Vector2::default(),
                                0.0,
                                NO_TINT
                            );
                        }
                        Text(t) => {
                            shader_d.draw_text(
                                &t.text,
                                t.xy.x as i32,
                                t.xy.y as i32,
                                40,
                                TEXT
                            );
                        }
                    }
                }
            }

            let render_target_source_rect = Rectangle {
                x: 0.,
                y: (RENDER_TARGET_SIZE as f32) - screen_render_rect.height,
                width: screen_render_rect.width,
                // y flip for openGL
                height: -screen_render_rect.height
            };

            d.draw_texture_pro(
                &render_target,
                render_target_source_rect,
                screen_render_rect,
                Vector2::default(),
                0.0,
                NO_TINT
            );
        }
    }

}