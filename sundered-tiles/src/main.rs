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

struct SourceSpec {
    x: f32,
    y: f32,
    rotation: f32,
}

fn source_spec(sprite: SpriteKind) -> SourceSpec {
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
        Hint => 9.0,
        EdgeUp | EdgeDown | EdgeLeft | EdgeRight => 10.,
        QuestionMark => 11.,
        Selectrum | RulerEnd => 15.0,
    };

    let rotation = match sprite {
        Hidden
        | Red
        | Green
        | Blue
        | RedStar
        | GreenStar
        | BlueStar
        | InstrumentalGoal
        | TerminalGoal
        | Hint
        | Selectrum
        | RulerEnd
        | QuestionMark
        | EdgeDown => 0.,
        EdgeLeft => 90.,
        EdgeUp => 180.,
        EdgeRight => 270.,
    };

    SourceSpec {
        x: sx * SPRITE_PIXELS_PER_TILE_SIDE,
        y: 0.0,
        rotation,
    }
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
        source_spec,
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

        // We need a reference to this so we can use `draw_text_rec`
        let font = rl.get_font_default();

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
    
            duration.as_nanos();
            1622078637208295340
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
        const RULER_TINT: Color = Color{ r: 0xff, g: 0xff, b: 0xff, a: 192 };

        fn tint_from_kind(sprite: game::SpriteKind) -> Color {
            use game::SpriteKind::*;
            match sprite {
                Hidden
                | Red
                | Green
                | Blue
                | RedStar
                | GreenStar
                | BlueStar
                | InstrumentalGoal
                | TerminalGoal
                | Selectrum
                | Hint
                | EdgeUp
                | EdgeDown
                | EdgeLeft
                | EdgeRight
                | QuestionMark => NO_TINT,
                RulerEnd => RULER_TINT
            }
        }

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

            if rl.is_key_pressed(KEY_Q) || rl.is_key_pressed(KEY_X) {
                input_flags |= game::INPUT_TOOL_LEFT_PRESSED;
            }

            if rl.is_key_pressed(KEY_E) || rl.is_key_pressed(KEY_C) {
                input_flags |= game::INPUT_TOOL_RIGHT_PRESSED;
            }

            if rl.is_key_pressed(KEY_R) || rl.is_key_pressed(KEY_Z) {
                input_flags |= game::INPUT_UI_RESET_PRESSED;
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
                            let spec = source_spec(s.sprite);

                            shader_d.draw_texture_pro(
                                &spritesheet,
                                Rectangle {
                                    x: spec.x + SPRITE_BORDER,
                                    y: spec.y + SPRITE_BORDER,
                                    ..tile_base_source_rect
                                },
                                Rectangle {
                                    x: s.xy.x,
                                    y: s.xy.y,
                                    ..tile_base_render_rect
                                },
                                Vector2::default(),
                                spec.rotation,
                                tint_from_kind(s.sprite)
                            );
                        }
                        Text(t) => {
                            // constant arrived at through trial and error.
                            let size = sizes.draw_wh.w * (1./48.);

                            shader_d.draw_text_rec(
                                &font,
                                &t.text,
                                Rectangle {
                                    x: t.xy.x,
                                    y: t.xy.y,
                                    width: t.wh.w,
                                    height: t.wh.h,
                                },
                                size,
                                1., // spacing
                                true, // word_wrap
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