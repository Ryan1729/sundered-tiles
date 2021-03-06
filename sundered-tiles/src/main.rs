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
        RedStar | Between => 4.0,
        GreenStar => 5.0,
        BlueStar => 6.0,
        InstrumentalGoal => 7.0,
        TerminalGoal => 8.0,
        Hint | GoalDistanceHint => 9.0,
        EdgeUp | EdgeDown | EdgeLeft | EdgeRight
        | EdgeUpLeft | EdgeUpRight | EdgeDownLeft | EdgeDownRight => 10.,
        QuestionMark | NotSymbol => 11.,
        RedGreen | RedGoal => 12.,
        GreenBlue | GreenGoal => 13.,
        BlueRed | BlueGoal => 14.,
        Selectrum | RulerEnd => 15.0,
    };

    let sy = match sprite {
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
        | RedGreen
        | GreenBlue
        | BlueRed
        | EdgeUp | EdgeDown | EdgeLeft | EdgeRight => 0.,
        Between
        | GoalDistanceHint
        | EdgeUpLeft | EdgeUpRight | EdgeDownLeft | EdgeDownRight
        | RedGoal | GreenGoal | BlueGoal
        | NotSymbol => 1.,
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
        | Between
        | Selectrum
        | RulerEnd
        | QuestionMark
        | RedGreen
        | GreenBlue
        | BlueRed
        | GoalDistanceHint
        | RedGoal
        | GreenGoal
        | BlueGoal
        | EdgeDown
        | EdgeDownRight
        | NotSymbol => 0.,
        EdgeLeft | EdgeDownLeft => 90.,
        EdgeUp | EdgeUpLeft => 180.,
        EdgeRight | EdgeUpRight => 270.,
    };

    SourceSpec {
        x: sx * SPRITE_PIXELS_PER_TILE_SIDE,
        y: sy * SPRITE_PIXELS_PER_TILE_SIDE,
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
    
            //duration.as_nanos()
            1629775657529055346
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
                | Between
                | EdgeUp
                | EdgeDown
                | EdgeLeft
                | EdgeRight
                | RedGreen
                | GreenBlue
                | BlueRed
                | QuestionMark
                | GoalDistanceHint
                | RedGoal
                | GreenGoal
                | BlueGoal
                | EdgeUpLeft
                | EdgeUpRight
                | EdgeDownLeft
                | EdgeDownRight
                | NotSymbol => NO_TINT,
                RulerEnd => RULER_TINT
            }
        }

        const SPRITE_BORDER: f32 = 4.;

        let mut show_stats = false;
        use std::time::Instant;
        struct TimeSpan {
            start: Instant,
            end: Instant,
        }

        impl Default for TimeSpan {
            fn default() -> Self {
                let start = Instant::now();
                Self {
                    start,
                    end: start,
                }
            }
        }

        impl std::fmt::Display for TimeSpan {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "{: >6.3} ms",
                    (self.end - self.start).as_micros() as f32 / 1000.0
                )
            }
        }

        #[derive(Default)]
        struct FrameStats {
            loop_body: TimeSpan,
            input_gather: TimeSpan,
            update: TimeSpan,
            render: TimeSpan,
        }

        let mut prev_stats = FrameStats::default();

        while !rl.window_should_close() {
            let mut current_stats = FrameStats::default();
            current_stats.loop_body.start = Instant::now();
            current_stats.input_gather.start = current_stats.loop_body.start;

            if rl.is_key_pressed(KEY_F11) {
                rl.toggle_fullscreen();
            }

            if rl.is_key_pressed(KEY_F10) {
                show_stats = !show_stats;
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

            if rl.is_key_pressed(KEY_B) {
                input_flags |= game::INPUT_VIEW_MODE_RIGHT_PRESSED;
            }

            if rl.is_key_pressed(KEY_V) {
                input_flags |= game::INPUT_VIEW_MODE_LEFT_PRESSED;
            }

            if rl.is_key_pressed(KEY_R) || rl.is_key_pressed(KEY_Z) {
                input_flags |= game::INPUT_UI_RESET_PRESSED;
            }
            current_stats.input_gather.end = Instant::now();
            current_stats.update.start = current_stats.input_gather.end;

            game::update(
                &mut state,
                &mut commands,
                input_flags,
                draw_wh(&rl)
            );

            current_stats.update.end = Instant::now();
            current_stats.render.start = current_stats.update.end;

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

                            let origin = Vector2 {
                                x: (tile_base_render_rect.width / 2.).round(),
                                y: (tile_base_render_rect.height / 2.).round(),
                            };

                            let render_rect = Rectangle {
                                x: s.xy.x + origin.x,
                                y: s.xy.y + origin.y,
                                ..tile_base_render_rect
                            };

                            let source_rect = Rectangle {
                                x: spec.x + SPRITE_BORDER,
                                y: spec.y + SPRITE_BORDER,
                                ..tile_base_source_rect
                            };

                            shader_d.draw_texture_pro(
                                &spritesheet,
                                source_rect,
                                render_rect,
                                origin,
                                spec.rotation,
                                tint_from_kind(s.sprite)
                            );
                        }
                        Text(t) => {
                            macro_rules! draw_to_fill_rect {
                                ($rect: expr, $text: expr) => {{
                                    let mut size = i32::MAX;
                                    let mut low = 0;
                                    let mut high = i32::MAX;
    
                                    let mut width;
                                    let mut next_width;
                                    let mut height;
                                    let mut next_height;

                                    let text = $text;
    
                                    macro_rules! set_wh {
                                        () => {
                                            width = measure_text(text, size);
                                            if width == i32::MIN {
                                                width = i32::MAX;
                                            }
                                            next_width = measure_text(text, size.saturating_add(1));
                                            if next_width == i32::MIN {
                                                next_width = i32::MAX;
                                            }
    
                                            // TODO Include line count in height approximation
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
    
                                        let width_does_not_fit = width as f32 > $rect.width;
                                        let height_does_not_fit = height as f32 > $rect.height;
                                        let next_width_fits = $rect.width > next_width as f32;
                                        let next_height_fits = $rect.height > next_height as f32;
    
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
    
                                    {
                                        #![allow(unused_assignments)]
                                        set_wh!();
                                    }
    
                                    let desired_center_x = $rect.x + ($rect.width / 2.);
                                    let desired_center_y = $rect.y + ($rect.height / 2.);
    
                                    let centered_x = desired_center_x - (width as f32 / 2.);
                                    let centered_y = desired_center_y - (height as f32 / 2.);
    
                                    shader_d.draw_text(
                                        $text,
                                        centered_x as i32,
                                        centered_y as i32,
                                        size,
                                        TEXT
                                    );
                                }}
                            }

                            use game::draw::TextKind;
                            match t.kind {
                                TextKind::DistanceMarker => {
                                    let rect = Rectangle {
                                        x: t.xy.x,
                                        y: t.xy.y,
                                        width: t.wh.w,
                                        height: t.wh.h,
                                    };
                                    draw_to_fill_rect!(rect, &t.text);
                                },
                                TextKind::ModMarker(modulus) => {
                                    let width = t.wh.w / 2.;
                                    let height = t.wh.h * 2. / 3.;
                                    let top_left_rect = Rectangle {
                                        x: t.xy.x,
                                        y: t.xy.y,
                                        width,
                                        height,
                                    };
                                    draw_to_fill_rect!(top_left_rect, &t.text);

                                    let top_right_rect = Rectangle {
                                        x: t.xy.x + width,
                                        y: t.xy.y,
                                        width,
                                        height: t.wh.h / 3.,
                                    };
                                    draw_to_fill_rect!(top_right_rect, "%");
                                    let bottom_rect = Rectangle {
                                        x: t.xy.x + width,
                                        y: t.xy.y + (t.wh.h - height),
                                        width,
                                        height,
                                    };
                                    draw_to_fill_rect!(
                                        bottom_rect,
                                        &format!("{}", modulus)
                                    );
                                },
                                TextKind::HintString => {
                                    shader_d.draw_text_rec(
                                        &font,
                                        &t.text,
                                        Rectangle {
                                            x: t.xy.x,
                                            y: t.xy.y,
                                            width: t.wh.w,
                                            height: t.wh.h,
                                        },
                                        // Constant arrived at through trial and error.
                                        sizes.draw_wh.w * (1./72.),
                                        1.,
                                        true, // word_wrap
                                        TEXT
                                    );
                                },
                                TextKind::Level 
                                | TextKind::Digs 
                                | TextKind::Fast 
                                | TextKind::Ruler => {
                                    shader_d.draw_text_rec(
                                        &font,
                                        &t.text,
                                        Rectangle {
                                            x: t.xy.x,
                                            y: t.xy.y,
                                            width: t.wh.w,
                                            height: t.wh.h,
                                        },
                                        // Constant arrived at through trial and error.
                                        sizes.draw_wh.w * (1./48.),
                                        1.,
                                        true, // word_wrap
                                        TEXT
                                    );
                                },
                            };
                        }
                    }
                }

                if show_stats {
                    shader_d.draw_text_rec(
                        &font,
                        &format!(
                            "loop {}\ninput {}\nupdate {}\nrender {}",
                            prev_stats.loop_body,
                            prev_stats.input_gather,
                            prev_stats.update,
                            prev_stats.render,
                        ),
                        Rectangle {
                            x: 0.,
                            y: 0.,
                            width: sizes.play_xywh.x,
                            height: sizes.play_xywh.h,
                        },
                        // Constant arrived at through trial and error.
                        sizes.draw_wh.w * (1./96.),
                        1.,
                        true, // word_wrap
                        TEXT
                    );
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

            current_stats.render.end = Instant::now();
            current_stats.loop_body.end = current_stats.render.end;

            prev_stats = current_stats;
        }
    }

}
