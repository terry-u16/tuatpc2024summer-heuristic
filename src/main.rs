use std::fmt::Display;

use grid::{Coord, Map2d};
use itertools::Itertools;
use marker::Usize1;
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
    a: Vec<usize>,
    chain_coefs: Vec<u32>,
}

impl Input {
    const COLOR_COUNT: usize = 4;
    const CONNNECT_COUNT: u32 = 3;

    fn read() -> Self {
        input! {
            n: usize,
            height: usize,
            width: usize,
            a: [Usize1; n],
        }

        // 縦横入れ替え
        let (height, width) = (width, height);

        // 連鎖係数の計算
        let mut chain_coefs = vec![0];

        for i in 1..=21 {
            let c = (512.0 * (1.0 - 0.99f64.powi(2i32.pow(i)))).round() as u32;
            chain_coefs.push(c);
        }

        Self {
            n,
            height,
            width,
            a,
            chain_coefs,
        }
    }
}

fn main() {
    let input = Input::read();
    let ops = solve(&input);

    for op in ops {
        println!("{}", op);
    }
}

fn solve(input: &Input) -> Vec<Op> {
    // 全消しを前提として、DPで最適な連鎖列を求める
    let mut dp = vec![0; input.n + 1];
    let mut history = (0..=input.n)
        .map::<(Box<dyn Chain>, usize), _>(|_| (Box::new(TrushChain::new()), 0))
        .collect_vec();

    for from in 0..input.n {
        let mut counts = [0; Input::COLOR_COUNT];

        for to in from + 1..=input.n {
            counts[input.a[to - 1]] += 1;

            if counts.iter().sum::<u32>() >= (input.height * input.width) as u32 {
                break;
            }

            let snake = SnakeChain::new(input, counts);
            let score = dp[from] + snake.execute(&input.a[from..to]).score;

            if dp[to].change_max(score) {
                history[to] = (Box::new(snake), from);
            }
        }
    }

    eprintln!("Score = {}", dp[input.n]);

    let mut ops = vec![];
    let mut current = input.n;

    while current > 0 {
        let (chain, prev) = &history[current];
        let mut restored_ops = chain.restore(&input.a[*prev..current]);

        while let Some(op) = restored_ops.pop() {
            ops.push(op);
        }

        current = *prev;
    }

    ops.reverse();
    ops
}

#[derive(Debug, Clone, Copy)]
struct Op {
    pos: Option<Coord>,
    fire: bool,
}

impl Op {
    fn new(pos: Option<Coord>, fire: bool) -> Self {
        Self { pos, fire }
    }
}

impl Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.pos {
            Some(pos) => write!(f, "{} {}", pos.col() + 1, pos.row() + 1),
            None => write!(f, "-1 -1"),
        }?;
        write!(f, " {}", if self.fire { 1 } else { 0 })
    }
}

// いらない気がしてきた
/*
struct Board {
    board: Map2d<Option<usize>>,
    height: usize,
    width: usize,
    chain: u32,
}

impl Board {
    fn new(board: Map2d<Option<usize>>, height: usize, width: usize, chain: u32) -> Self {
        Self {
            board,
            height,
            width,
            chain,
        }
    }

    fn progress(&mut self) -> u32 {
        self.chain += 1;
        self.fall();
        self.erase()
    }

    fn fall(&mut self) {
        for row in 0..self.height {
            let row = &mut self.board[row];
            let mut dst = 0;

            for src in 0..self.width {
                if let Some(v) = row[src] {
                    row[src] = None;
                    row[dst] = Some(v);
                    dst += 1;
                }
            }
        }
    }

    fn erase(&mut self) -> u32 {


        todo!();
    }
}
 */

struct EraseInfo {
    chain: u32,
    count: u32,
    kinds: u32,
}

impl EraseInfo {
    fn new(chain: u32, count: u32, kinds: u32) -> Self {
        Self {
            chain,
            count,
            kinds,
        }
    }

    fn calc_score(&self, input: &Input) -> u32 {
        self.count * self.count * (input.chain_coefs[self.chain as usize] + self.kinds * self.kinds)
    }
}

trait Chain {
    fn positions(&self) -> &[Vec<Coord>; Input::COLOR_COUNT];

    fn base_score(&self) -> u32;

    fn execute(&self, colors: &[usize]) -> ExecuteResult {
        let mut disposed = 0;
        let mut completed = self.positions().iter().filter(|&p| p.is_empty()).count();
        let mut counts = [0; Input::COLOR_COUNT];

        for (i, &c) in colors.iter().enumerate() {
            let needed = self.positions()[c].len();

            if counts[c] < needed {
                counts[c] += 1;

                if counts[c] == needed {
                    completed += 1;

                    if completed == Input::COLOR_COUNT {
                        let score = self.base_score() + disposed * 100;
                        return ExecuteResult::new(score, i + 1, true);
                    }
                }
            } else {
                disposed += 1;
            }
        }

        ExecuteResult::new(disposed * 100, colors.len(), false)
    }

    fn restore(&self, colors: &[usize]) -> Vec<Op> {
        let mut ops = vec![];
        let mut counts = [0; Input::COLOR_COUNT];

        for (i, &c) in colors.iter().enumerate() {
            let fire = i == colors.len() - 1;

            match self.positions()[c].get(counts[c]) {
                Some(&pos) => ops.push(Op::new(Some(pos), fire)),
                None => ops.push(Op::new(None, fire)),
            }

            counts[c] += 1;
        }

        let completed = counts
            .iter()
            .zip(self.positions().iter())
            .all(|(&c, p)| c >= p.len());
        if completed {
            ops
        } else {
            vec![Op::new(None, false); colors.len()]
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ExecuteResult {
    score: u32,
    len: usize,
    succeeded: bool,
}

impl ExecuteResult {
    fn new(score: u32, len: usize, succeeded: bool) -> Self {
        Self {
            score,
            len,
            succeeded,
        }
    }
}

#[derive(Debug, Clone)]
struct TrushChain {
    positions: [Vec<Coord>; Input::COLOR_COUNT],
}

impl TrushChain {
    fn new() -> Self {
        Self {
            positions: [vec![], vec![], vec![], vec![]],
        }
    }
}

impl Chain for TrushChain {
    fn positions(&self) -> &[Vec<Coord>; Input::COLOR_COUNT] {
        &self.positions
    }

    fn base_score(&self) -> u32 {
        0
    }

    fn execute(&self, colors: &[usize]) -> ExecuteResult {
        ExecuteResult::new(colors.len() as u32 * 100, colors.len(), false)
    }

    fn restore(&self, _colors: &[usize]) -> Vec<Op> {
        vec![Op::new(None, false)]
    }
}

#[derive(Debug, Clone)]
struct SnakeChain {
    positions: [Vec<Coord>; Input::COLOR_COUNT],
    base_score: u32,
}

impl SnakeChain {
    fn new(input: &Input, counts: [u32; Input::COLOR_COUNT]) -> Self {
        assert!(counts.iter().sum::<u32>() <= input.height as u32 * input.width as u32);
        let mut positions = [vec![], vec![], vec![], vec![]];

        let mut row = 0;
        let mut col = 0;
        let mut erase_count = 0;
        let mut erase_kinds = 0;

        for (color, &count) in counts.iter().enumerate() {
            if count < Input::CONNNECT_COUNT {
                continue;
            }

            erase_count += count;
            erase_kinds += 1;

            for _ in 0..count {
                positions[color].push(Coord::new(row, col));

                if col % 2 == 0 {
                    if row + 1 < input.height {
                        row += 1;
                    } else {
                        col += 1;
                    }
                } else {
                    if row > 0 {
                        row -= 1;
                    } else {
                        col += 1;
                    }
                }
            }
        }

        let base_score = EraseInfo::new(1, erase_count, erase_kinds).calc_score(input);

        Self {
            positions,
            base_score,
        }
    }
}

impl Chain for SnakeChain {
    fn positions(&self) -> &[Vec<Coord>; Input::COLOR_COUNT] {
        &self.positions
    }

    fn base_score(&self) -> u32 {
        self.base_score
    }
}

#[allow(dead_code)]
mod grid {
    use std::ops::{Add, AddAssign, Index, IndexMut};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Coord {
        row: u8,
        col: u8,
    }

    impl Coord {
        pub const fn new(row: usize, col: usize) -> Self {
            Self {
                row: row as u8,
                col: col as u8,
            }
        }

        pub const fn row(&self) -> usize {
            self.row as usize
        }

        pub const fn col(&self) -> usize {
            self.col as usize
        }

        pub fn in_map(&self, size: usize) -> bool {
            self.row < size as u8 && self.col < size as u8
        }

        pub const fn to_index(&self, size: usize) -> usize {
            self.row as usize * size + self.col as usize
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: u8, x1: u8) -> usize {
            (x0 as i64 - x1 as i64).abs() as usize
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordDiff {
        dr: i8,
        dc: i8,
    }

    impl CoordDiff {
        pub const fn new(dr: i32, dc: i32) -> Self {
            Self {
                dr: dr as i8,
                dc: dc as i8,
            }
        }

        pub const fn invert(&self) -> Self {
            Self {
                dr: -self.dr,
                dc: -self.dc,
            }
        }

        pub const fn dr(&self) -> i32 {
            self.dr as i32
        }

        pub const fn dc(&self) -> i32 {
            self.dc as i32
        }
    }

    impl Add<CoordDiff> for Coord {
        type Output = Coord;

        fn add(self, rhs: CoordDiff) -> Self::Output {
            Coord {
                row: self.row.wrapping_add(rhs.dr as u8),
                col: self.col.wrapping_add(rhs.dc as u8),
            }
        }
    }

    impl AddAssign<CoordDiff> for Coord {
        fn add_assign(&mut self, rhs: CoordDiff) {
            self.row = self.row.wrapping_add(rhs.dr as u8);
            self.col = self.col.wrapping_add(rhs.dc as u8);
        }
    }

    pub const ADJACENTS: [CoordDiff; 4] = [
        CoordDiff::new(-1, 0),
        CoordDiff::new(0, 1),
        CoordDiff::new(1, 0),
        CoordDiff::new(0, -1),
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        size: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, size: usize) -> Self {
            debug_assert!(size * size == map.len());
            Self { size, map }
        }
    }

    impl<T: Default + Clone> Map2d<T> {
        pub fn with_default(size: usize) -> Self {
            let map = vec![T::default(); size * size];
            Self { size, map }
        }
    }

    impl<T> Index<Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> IndexMut<Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> Index<&Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> IndexMut<&Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.size)]
        }
    }

    impl<T> Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.size;
            let end = begin + self.size;
            &self.map[begin..end]
        }
    }

    impl<T> IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.size;
            let end = begin + self.size;
            &mut self.map[begin..end]
        }
    }

    #[derive(Debug, Clone)]
    pub struct ConstMap2d<T, const N: usize> {
        map: Vec<T>,
    }

    impl<T, const N: usize> ConstMap2d<T, N> {
        pub fn new(map: Vec<T>) -> Self {
            assert_eq!(map.len(), N * N);
            Self { map }
        }
    }

    impl<T: Default + Clone, const N: usize> ConstMap2d<T, N> {
        pub fn with_default() -> Self {
            let map = vec![T::default(); N * N];
            Self { map }
        }
    }

    impl<T, const N: usize> Index<Coord> for ConstMap2d<T, N> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> IndexMut<Coord> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> Index<&Coord> for ConstMap2d<T, N> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coord) -> &Self::Output {
            &self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> IndexMut<&Coord> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(N)]
        }
    }

    impl<T, const N: usize> Index<usize> for ConstMap2d<T, N> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * N;
            let end = begin + N;
            &self.map[begin..end]
        }
    }

    impl<T, const N: usize> IndexMut<usize> for ConstMap2d<T, N> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * N;
            let end = begin + N;
            &mut self.map[begin..end]
        }
    }

    #[cfg(test)]
    mod test {
        use super::{ConstMap2d, Coord, CoordDiff, Map2d};

        #[test]
        fn coord_add() {
            let c = Coord::new(2, 4);
            let d = CoordDiff::new(-3, 5);
            let actual = c + d;

            let expected = Coord::new(!0, 9);
            assert_eq!(expected, actual);
        }

        #[test]
        fn coord_add_assign() {
            let mut c = Coord::new(2, 4);
            let d = CoordDiff::new(-3, 5);
            c += d;

            let expected = Coord::new(!0, 9);
            assert_eq!(expected, c);
        }

        #[test]
        fn map_new() {
            let map = Map2d::new(vec![0, 1, 2, 3], 2);
            let actual = map[Coord::new(1, 0)];
            let expected = 2;
            assert_eq!(expected, actual);
        }

        #[test]
        fn map_default() {
            let map = Map2d::with_default(2);
            let actual = map[Coord::new(1, 0)];
            let expected = 0;
            assert_eq!(expected, actual);
        }

        #[test]
        fn const_map_new() {
            let map = ConstMap2d::<_, 2>::new(vec![0, 1, 2, 3]);
            let actual = map[Coord::new(1, 0)];
            let expected = 2;
            assert_eq!(expected, actual);
        }

        #[test]
        fn const_map_default() {
            let map = ConstMap2d::<_, 2>::with_default();
            let actual = map[Coord::new(1, 0)];
            let expected = 0;
            assert_eq!(expected, actual);
        }
    }
}
