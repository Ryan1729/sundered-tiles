use std::{
    convert::TryInto,
    error::Error,
};

use macroquad::{
    drawing::Image,
    Texture2D,
    clear_background,
    draw_texture,
    load_texture_from_image,
    next_frame,
    screen_height,
    screen_width,
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

#[macroquad::main("Sundered Tiles")]
async fn main() {
    let spritesheet_texture: Texture2D = load_spritesheet()
        .expect("Embedded spritesheet could not be loaded!");

    loop {
        clear_background(BLACK);
        draw_texture(
            spritesheet_texture,
            screen_width() / 2. - spritesheet_texture.width() / 2.,
            screen_height() / 2. - spritesheet_texture.height() / 2.,
            WHITE,
        );
        next_frame().await
    }
}
