#[allow(unused_imports)]
use proconio::*;
#[allow(unused_imports)]
use rand::prelude::*;

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Input {
    n: usize,
    height: usize,
    width: usize,
    a: Vec<u8>,
}

impl Input {
    fn read() -> Self {
        input! {
            n: usize,
            height: usize,
            width: usize,
            a: [u8; n],
        }

        Self {
            n,
            height,
            width,
            a,
        }
    }
}

fn main() {
    let input = Input::read();

    if input.height >= 6 && input.width >= 4 {
        for _ in 0..input.n {
            println!("-1 -1 1");
        }
    } else {
        panic!();
    }
}
