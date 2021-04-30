#![no_std]

pub use f32_is;

#[macro_use]
pub mod unit;
pub use unit::{W, H, w, h};

#[macro_use]
pub mod bi_unit;
pub use bi_unit::{X, Y, x, y};

use core::ops::Add;

impl Add<X> for W {
    type Output = X;

    fn add(self, rhs: X) -> Self::Output {
        let w: f32 = self.into();
        let x: f32 = rhs.into();

        bi_unit::x!(w + x)
    }
}

impl Add<W> for X {
    type Output = X;

    fn add(self, rhs: W) -> Self::Output {
        let x: f32 = self.into();
        let w: f32 = rhs.into();

        bi_unit::x!(x + w)
    }
}

impl Add<Y> for H {
    type Output = Y;

    fn add(self, rhs: Y) -> Self::Output {
        let h: f32 = self.into();
        let y: f32 = rhs.into();

        bi_unit::y!(h + y)
    }
}

impl Add<H> for Y {
    type Output = Y;

    fn add(self, rhs: H) -> Self::Output {
        let y: f32 = self.into();
        let h: f32 = rhs.into();

        bi_unit::y!(y + h)
    }
}
