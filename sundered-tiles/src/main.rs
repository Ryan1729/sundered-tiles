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
        RedStar => 4.0,
        GreenStar => 5.0,
        BlueStar => 6.0,
        InstrumentalGoal => 7.0,
        TerminalGoal => 8.0,
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
mod macroquad_platform;

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
        // Maybe it would be faster/less hot to avoiding clearing the whole thing 
        // each frame?
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
        game::update(&mut state, &mut commands, 0, draw_wh(&rl));

        const BACKGROUND: Color = Color{ r: 0x22, g: 0x22, b: 0x22, a: 255 };
        const WHITE: Color = Color{ r: 0xee, g: 0xee, b: 0xee, a: 255 };
        const TEXT: Color = WHITE;
        const NO_TINT: Color = WHITE;
        const OUTLINE: Color = WHITE;

        const SPRITE_BORDER: f32 = 4.;

        while !rl.window_should_close() {
            if rl.is_key_pressed(KEY_F11) {
                rl.toggle_fullscreen();
            }

            let mut input_flags = 0;

            if rl.is_key_pressed(KEY_F) {   
                input_flags |= game::INPUT_FAST_PRESSED;
            }

            if rl.is_key_pressed(KEY_SPACE) || rl.is_key_pressed(KEY_ENTER) {
                input_flags |= game::INPUT_INTERACT_PRESSED;
            }

            if rl.is_key_down(KEY_UP) || rl.is_key_down(KEY_W) {
                input_flags |= game::INPUT_UP_DOWN;
            }

            if rl.is_key_down(KEY_DOWN) || rl.is_key_down(KEY_S) {
                input_flags |= game::INPUT_DOWN_DOWN;
            }

            if rl.is_key_down(KEY_LEFT) || rl.is_key_down(KEY_A) {
                input_flags |= game::INPUT_LEFT_DOWN;
            }

            if rl.is_key_down(KEY_RIGHT) || rl.is_key_down(KEY_D) {
                input_flags |= game::INPUT_RIGHT_DOWN;
            }

            if rl.is_key_pressed(KEY_UP) || rl.is_key_pressed(KEY_W) {
                input_flags |= game::INPUT_UP_PRESSED;
            }
            
            if rl.is_key_pressed(KEY_DOWN) || rl.is_key_pressed(KEY_S) {
                input_flags |= game::INPUT_DOWN_PRESSED;
            }

            if rl.is_key_pressed(KEY_LEFT) || rl.is_key_pressed(KEY_A) {
                input_flags |= game::INPUT_LEFT_PRESSED;
            }

            if rl.is_key_pressed(KEY_RIGHT) || rl.is_key_pressed(KEY_D) {
                input_flags |= game::INPUT_RIGHT_PRESSED;
            }

            game::update(
                &mut state,
                &mut commands,
                input_flags,
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
                    width: SPRITE_PIXELS_PER_TILE_SIDE - SPRITE_BORDER * 2.,
                    height: SPRITE_PIXELS_PER_TILE_SIDE - SPRITE_BORDER * 2.,
                };
    
                let tile_base_render_rect = Rectangle {
                    x: 0.,
                    y: 0.,
                    width: sizes.tile_side_length,
                    height: sizes.tile_side_length,
                };
    
                for cmd in commands.0.iter() {
                    use game::draw::Command::*;
                    match cmd {
                        Sprite(s) => {
                            let (source_x, source_y) = source_coords(s.sprite);
        
                            shader_d.draw_texture_pro(
                                &spritesheet,
                                Rectangle {
                                    x: source_x + SPRITE_BORDER,
                                    y: source_y + SPRITE_BORDER,
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
                            let mut size = i32::MAX;
                            let mut low = 0;
                            let mut high = i32::MAX;

                            let mut width;
                            let mut next_width;
                            let mut height;
                            let mut next_height;

                            macro_rules! set_wh {
                                () => {
                                    width = measure_text(&t.text, size);
                                    if width == i32::MIN {
                                        width = i32::MAX;
                                    }
                                    next_width = measure_text(&t.text, size.saturating_add(1));
                                    if next_width == i32::MIN {
                                        next_width = i32::MAX;
                                    }
    
                                    // TODO really measure height. (9/5 arrived at through trial and error)
                                    height = measure_text("m", size).saturating_mul(9) / 5;
                                    next_height = measure_text("m", size.saturating_add(1)).saturating_mul(9) / 5;
                                }
                            }

                            {
                                #![allow(unused_assignments)]
                                set_wh!();
                            }

                            while low <= high {
                                set_wh!();

                                let width_does_not_fit = width as f32 > t.wh.w;
                                let height_does_not_fit = height as f32 > t.wh.h;
                                let next_width_fits = t.wh.w > next_width as f32;
                                let next_height_fits = t.wh.h > next_height as f32;

                                if width_does_not_fit
                                || height_does_not_fit
                                || next_width_fits
                                || next_height_fits {
                                    size = low.saturating_add(high) / 2;
                                }

                                if width_does_not_fit || height_does_not_fit {
                                    high = size - 1;
                                }

                                if next_width_fits || next_height_fits {
                                    low = size + 1;
                                }
                            }

                            let desired_center_x = t.xy.x + (t.wh.w / 2.);
                            let desired_center_y = t.xy.y + (t.wh.h / 2.);

                            let centered_x = desired_center_x - (width as f32 / 2.);
                            let centered_y = desired_center_y - (height as f32 / 2.);

                            shader_d.draw_text(
                                &t.text,
                                centered_x.round() as i32,
                                centered_y.round() as i32,
                                size,
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