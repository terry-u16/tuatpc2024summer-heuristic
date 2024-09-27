use std::{collections::HashMap, fmt::Display, iter::Map, ops::Index, u32};

use ac_library::Dsu;
use chain_templates::RENSA_TEMPLATES;
use grid::{Coord, Map2d, ADJACENTS};
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
    let since = std::time::Instant::now();
    let millefeuille_dict = MillefeuilleDict::gen_all(input);
    eprintln!("Elapsed = {:?}", since.elapsed());

    for from in 0..input.n {
        let mut counts = [0; Input::COLOR_COUNT];

        for to in from + 1..=input.n {
            counts[input.a[to - 1]] += 1;

            if counts.iter().sum::<u32>() >= (input.height * input.width) as u32 {
                break;
            }

            // ヘルファイア
            let hellfire = HellFire::new(input, counts);
            let score = dp[from] + hellfire.execute(&input.a[from..to]).score;

            if dp[to].change_max(score) {
                history[to] = (Box::new(hellfire), from);
            }

            // ミルフィーユ
            if let Some(millefeuille) = millefeuille_dict.get(&counts) {
                let score = dp[from] + millefeuille.execute(&input.a[from..to]).score;

                if dp[to].change_max(score) {
                    history[to] = (Box::new(millefeuille.clone()), from);
                }
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
struct HellFire {
    positions: [Vec<Coord>; Input::COLOR_COUNT],
    base_score: u32,
}

impl HellFire {
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

impl Chain for HellFire {
    fn positions(&self) -> &[Vec<Coord>; Input::COLOR_COUNT] {
        &self.positions
    }

    fn base_score(&self) -> u32 {
        self.base_score
    }
}

#[derive(Debug, Clone)]
struct Millefeuille {
    positions: [Vec<Coord>; Input::COLOR_COUNT],
    base_score: u32,
}

impl Millefeuille {
    fn new(positions: [Vec<Coord>; Input::COLOR_COUNT], base_score: u32) -> Self {
        Self {
            positions,
            base_score,
        }
    }
}

impl Chain for Millefeuille {
    fn positions(&self) -> &[Vec<Coord>; Input::COLOR_COUNT] {
        &self.positions
    }

    fn base_score(&self) -> u32 {
        self.base_score
    }
}

struct MillefeuilleDict {
    best_keys: Vec<[u32; Input::COLOR_COUNT]>,
    hashmap: HashMap<[u32; Input::COLOR_COUNT], Millefeuille>,
    len: [u32; Input::COLOR_COUNT],
}

impl MillefeuilleDict {
    fn gen_all(input: &Input) -> Self {
        let mut candidates = HashMap::new();

        for &(s, height, width) in RENSA_TEMPLATES.iter() {
            if input.height >= height && input.width >= width {
                let templates = MillefeuilleTemplate::generate(s, input);

                for template in templates {
                    template.gen_millefeuille_candidates(&mut candidates);
                }
            }
        }

        MillefeuilleDict::new(candidates)
    }

    fn new(hashmap: HashMap<[u32; Input::COLOR_COUNT], Millefeuille>) -> Self {
        let mut len = [0; Input::COLOR_COUNT];

        for key in hashmap.keys() {
            for i in 0..Input::COLOR_COUNT {
                len[i].change_max(key[i]);
            }
        }

        for len in len.iter_mut() {
            *len += 1;
        }

        let mut best_keys = vec![[!0, !0, !0, !0]; (len[0] * len[1] * len[2] * len[3]) as usize];
        let mut best_score = vec![0; (len[0] * len[1] * len[2] * len[3]) as usize];

        for (key, value) in hashmap.iter() {
            let index = Self::to_index(key, &len);
            best_keys[index] = *key;
            best_score[index] = value.base_score;
        }

        // 累積maxを取る
        for i in 0..len[0] - 1 {
            for j in 0..len[1] {
                for k in 0..len[2] {
                    for l in 0..len[3] {
                        let index0 = Self::to_index(&[i, j, k, l], &len);
                        let index1 = Self::to_index(&[i + 1, j, k, l], &len);
                        let new_score = best_score[index0] + 100;

                        if best_score[index1].change_max(new_score) {
                            best_keys[index1] = best_keys[index0];
                        }
                    }
                }
            }
        }

        for j in 0..len[1] - 1 {
            for i in 0..len[0] {
                for k in 0..len[2] {
                    for l in 0..len[3] {
                        let index0 = Self::to_index(&[i, j, k, l], &len);
                        let index1 = Self::to_index(&[i, j + 1, k, l], &len);
                        let new_score = best_score[index0] + 100;

                        if best_score[index1].change_max(new_score) {
                            best_keys[index1] = best_keys[index0];
                        }
                    }
                }
            }
        }

        for k in 0..len[2] - 1 {
            for i in 0..len[0] {
                for j in 0..len[1] {
                    for l in 0..len[3] {
                        let index0 = Self::to_index(&[i, j, k, l], &len);
                        let index1 = Self::to_index(&[i, j, k + 1, l], &len);
                        let new_score = best_score[index0] + 100;

                        if best_score[index1].change_max(new_score) {
                            best_keys[index1] = best_keys[index0];
                        }
                    }
                }
            }
        }

        for l in 0..len[3] - 1 {
            for i in 0..len[0] {
                for j in 0..len[1] {
                    for k in 0..len[2] {
                        let index0 = Self::to_index(&[i, j, k, l], &len);
                        let index1 = Self::to_index(&[i, j, k, l + 1], &len);
                        let new_score = best_score[index0] + 100;

                        if best_score[index1].change_max(new_score) {
                            best_keys[index1] = best_keys[index0];
                        }
                    }
                }
            }
        }

        Self {
            best_keys,
            hashmap,
            len,
        }
    }

    fn to_index(key: &[u32; Input::COLOR_COUNT], len: &[u32; Input::COLOR_COUNT]) -> usize {
        (((key[0] * len[1] + key[1]) * len[2] + key[2]) * len[3] + key[3]) as usize
    }

    fn get(&self, key: &[u32; Input::COLOR_COUNT]) -> Option<&Millefeuille> {
        let mut index = key.clone();

        for (i, j) in index.iter_mut().zip(self.len.iter()) {
            i.change_min(*j - 1);
        }

        let best_key = self.best_keys[MillefeuilleDict::to_index(&index, &self.len)];
        self.hashmap.get(&best_key)
    }
}

struct MillefeuilleTemplate {
    template: Map2d<Option<usize>>,
    prohibited: Vec<Vec<usize>>,
    counts: Vec<u32>,
    placeholder_len: usize,
    base_score: u32,
    color_bonus_target: Vec<usize>,
    last_erase: u32,
}

impl MillefeuilleTemplate {
    fn new(
        template: Map2d<Option<usize>>,
        prohibited: Vec<Vec<usize>>,
        counts: Vec<u32>,
        placeholder_len: usize,
        base_score: u32,
        color_bonus_target: Vec<usize>,
        last_erase: u32,
    ) -> Self {
        Self {
            template,
            prohibited,
            counts,
            placeholder_len,
            base_score,
            color_bonus_target,
            last_erase,
        }
    }

    fn generate(template: &str, input: &Input) -> Vec<Self> {
        let template_str = template.split_whitespace().collect_vec();
        let height = template_str.len();
        let width = template_str.iter().map(|s| s.len()).max().unwrap();
        let mut board = MillefeuilleBoard::new(Map2d::with_default(width, height), height, width);

        for row in 0..height {
            for (col, c) in template_str[row].chars().enumerate() {
                let v = c as usize - 'A' as usize;
                board.board[Coord::new(row, col)] = Some(v);
            }
        }

        let mut result = vec![];

        loop {
            let max_val = board.board.iter().flatten().max();

            let Some(max_val) = max_val else {
                break;
            };

            let placeholder_len = max_val + 1;
            let prohibited = Self::gen_prohibited(board.clone(), placeholder_len);
            let (base_score, last_erase) = Self::calc_base_score(board.clone(), input);

            let mut counts = vec![0; placeholder_len];

            for v in board.board.iter().flatten() {
                counts[*v] += 1;
            }

            let color_bonus_targets = board.board[height - 1]
                .iter()
                .flatten()
                .copied()
                .sorted()
                .dedup()
                .collect_vec();

            result.push(Self::new(
                board.board.clone(),
                prohibited,
                counts.clone(),
                placeholder_len,
                base_score,
                color_bonus_targets,
                last_erase,
            ));

            board.erase();
            board.fall();
        }

        result
    }

    fn gen_millefeuille_candidates(
        &self,
        candidates: &mut HashMap<[u32; Input::COLOR_COUNT], Millefeuille>,
    ) {
        let mut color_counts = [0; Input::COLOR_COUNT];
        let mut assign = vec![None; self.placeholder_len];
        let mut avaliable_flags = vec![(1 << Input::COLOR_COUNT) - 1; self.placeholder_len];
        let mut iter = 0;
        self.collect_dfs(
            &mut color_counts,
            &mut assign,
            &mut avaliable_flags,
            0,
            candidates,
            &mut iter,
        );

        eprintln!("Iter = {}", iter);
    }

    fn collect_dfs(
        &self,
        color_counts: &mut [u32; Input::COLOR_COUNT],
        assign: &mut Vec<Option<usize>>,
        avaliable_flags: &mut Vec<u8>,
        depth: usize,
        candidates: &mut HashMap<[u32; Input::COLOR_COUNT], Millefeuille>,
        iter: &mut u64,
    ) {
        *iter += 1;

        if *iter % 1000000 == 0 {
            eprintln!("{}", *iter);
        }

        if depth == self.placeholder_len {
            let old_score = candidates
                .get(color_counts)
                .map(|f| f.base_score)
                .unwrap_or(0);
            let mut new_score = self.base_score;
            let mut color_flag = 0u32;

            for &target in self.color_bonus_target.iter() {
                color_flag |= 1 << assign[target].unwrap();
            }

            let color_count = color_flag.count_ones();
            new_score += self.last_erase * self.last_erase * color_count * color_count;

            if new_score > old_score {
                let mut positions = [vec![], vec![], vec![], vec![]];

                for row in 0..self.template.height() {
                    for col in 0..self.template.width() {
                        let c = Coord::new(row, col);

                        if let Some(v) = self.template[c] {
                            positions[assign[v].unwrap()].push(c);
                        }
                    }
                }

                candidates.insert(*color_counts, Millefeuille::new(positions, new_score));

                if candidates.len() % 10000 == 0 {
                    eprintln!("{}", candidates.len());
                }
            }

            return;
        }

        // 選択肢が少ないものを優先して探索
        let mut min_avail = u32::MAX;
        let mut target_i = !0;

        for (i, (&assign, &avail)) in assign.iter().zip(avaliable_flags.iter()).enumerate() {
            if assign.is_none() && min_avail.change_min(avail.count_ones()) {
                target_i = i;
            }
        }

        if min_avail == 0 {
            return;
        }

        for color in 0..Input::COLOR_COUNT {
            if (avaliable_flags[target_i] >> color) & 1 == 0 {
                continue;
            }

            color_counts[color] += self.counts[target_i];
            assign[target_i] = Some(color);
            let mut new_available_flags = avaliable_flags.clone();

            for adj in &self.prohibited[target_i] {
                new_available_flags[*adj] &= !(1 << color);
            }

            self.collect_dfs(
                color_counts,
                assign,
                &mut new_available_flags,
                depth + 1,
                candidates,
                iter,
            );

            assign[target_i] = None;
            color_counts[color] -= self.counts[target_i];
        }
    }

    fn gen_prohibited(mut board: MillefeuilleBoard, placeholder_len: usize) -> Vec<Vec<usize>> {
        let mut prohibited = vec![vec![]; placeholder_len];

        loop {
            for row in 0..board.height {
                for col in 0..board.width {
                    let c = Coord::new(row, col);

                    let Some(v) = board.board[c] else {
                        continue;
                    };

                    for adj in ADJACENTS {
                        let next = c + adj;

                        if !next.in_map(board.width, board.height) {
                            continue;
                        }

                        let Some(next_v) = board.board[next] else {
                            continue;
                        };

                        if v != next_v {
                            prohibited[v].push(next_v);
                        }
                    }
                }
            }

            if board.erase() == 0 {
                break;
            }

            board.fall();
        }

        for prohibits in prohibited.iter_mut() {
            prohibits.sort();
            prohibits.dedup();
        }

        prohibited
    }

    fn calc_base_score(mut board: MillefeuilleBoard, input: &Input) -> (u32, u32) {
        let mut chain = 0;
        let mut base_score = 0;
        let mut last_erase = 0;

        loop {
            chain += 1;
            board.fall();
            let erased = board.erase();

            if board.board.iter().flatten().count() > 0 {
                base_score += erased * erased * (input.chain_coefs[chain as usize] + 1);
            } else {
                base_score += erased * erased * input.chain_coefs[chain as usize];
            }

            if erased == 0 {
                break;
            }

            last_erase = erased;
        }

        (base_score, last_erase)
    }
}

#[derive(Debug, Clone)]
struct MillefeuilleBoard {
    board: Map2d<Option<usize>>,
    height: usize,
    width: usize,
}

impl MillefeuilleBoard {
    fn new(board: Map2d<Option<usize>>, height: usize, width: usize) -> Self {
        Self {
            board,
            height,
            width,
        }
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
        let mut dsu = Dsu::new(self.height * self.width);

        for row in 0..self.height {
            for col in 0..self.width {
                let c = Coord::new(row, col);

                for adj in ADJACENTS {
                    let next = c + adj;

                    if !next.in_map(self.width, self.height) {
                        continue;
                    }

                    if self.board[c] == self.board[next] {
                        dsu.merge(c.to_index(self.width), next.to_index(self.width));
                    }
                }
            }
        }

        let mut cnt = 0;

        for row in 0..self.height {
            for col in 0..self.width {
                let c = Coord::new(row, col);

                if let Some(_) = self.board[c] {
                    let size = dsu.size(c.to_index(self.width));

                    if size >= Input::CONNNECT_COUNT as usize {
                        self.board[c] = None;
                        cnt += 1;
                    }
                }
            }
        }

        cnt
    }
}

mod chain_templates {
    pub const RENSA_TEMPLATES: [(&str, usize, usize); 12] = [
        (CHAIN_5X5, 5, 5),
        (CHAIN_6X5, 6, 5),
        (CHAIN_6X5_ALT, 6, 5),
        (CHAIN_6X6, 6, 6),
        (CHAIN_6X6_ALT, 6, 6),
        (CHAIN_7X6, 7, 6),
        (CHAIN_7X7, 7, 7),
        (CHAIN_8X6, 8, 6),
        (CHAIN_8X6_ALT, 8, 6),
        (CHAIN_8X7, 8, 7),
        (CHAIN_8X7_ALT, 8, 7),
        (CHAIN_8X8, 8, 8),
    ];

    const CHAIN_5X5: &str = "
    FGHGF
    FEHHG
    EABCD
    EABCD
    ABCD
    ";

    const CHAIN_6X5: &str = "
    GHHIHG
    GFFII
    FABCDE
    ABCDE
    ABCDE";

    const CHAIN_6X5_ALT: &str = "
    FGGHGF
    FEEHIH
    EABCID
    ABCDI
    ABCD";

    const CHAIN_6X6: &str = "
    HIIJIH
    HGGJKJ
    GFFKK
    FABCDE
    ABCDE
    ABCDE";

    const CHAIN_6X6_ALT: &str = "
    HIIJIH
    HGGJKJ
    GFFKKA
    FABCDE
    ABCDE
    ABCDE";

    const CHAIN_7X6: &str = "
    IJJKJI
    IHHKLK
    HGGLL
    GABCDEF
    ABCDEF
    ABCDEF";

    const CHAIN_7X7: &str = "
    IJJKJI
    IHHLKK
    HGLMMML
    GABCDEF
    GABCDEF
    ABCDEF
    ABCDEF";

    const CHAIN_8X7: &str = "
    JKKLKJ
    JIIMLL
    IHMNNNM
    HABCDEFG
    HABCDEFG
    ABCDEFG
    ABCDEFG";

    const CHAIN_8X6: &str = "
    IJJIMMML
    IHKKLLKJ
    HABCDEFG
    HABCDEFG
    ABCDEFG
    ABCDEFG";

    const CHAIN_8X6_ALT: &str = "
    JKKLMLKJ
    JIIMNML
    IHHNOOON
    HABCDEFG
    ABCDEFG
    ABCDEFG";

    const CHAIN_8X7_ALT: &str = "
    JKKLKJI
    JILMLOO
    IHMNNONM
    HABCDEFG
    HABCDEFG
    ABCDEFG
    ABCDEFG";

    const CHAIN_8X8: &str = "
    KKLMLKJ
    JILNMM
    JINON
    IHOO
    HABCDEFG
    HABCDEFG
    ABCDEFG
    ABCDEFG";
}

#[allow(dead_code)]
mod grid {
    use std::{
        ops::{Add, AddAssign, Index, IndexMut},
        slice::Iter,
    };

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

        pub fn in_map(&self, width: usize, height: usize) -> bool {
            self.row < height as u8 && self.col < width as u8
        }

        pub const fn to_index(&self, width: usize) -> usize {
            self.row as usize * width + self.col as usize
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
        width: usize,
        height: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, width: usize, height: usize) -> Self {
            debug_assert!(width * height == map.len());
            Self { width, height, map }
        }

        pub fn width(&self) -> usize {
            self.width
        }

        pub fn height(&self) -> usize {
            self.height
        }

        pub fn iter(&self) -> Iter<T> {
            self.map.iter()
        }
    }

    impl<T: Default + Clone> Map2d<T> {
        pub fn with_default(width: usize, height: usize) -> Self {
            let map = vec![T::default(); width * height];
            Self::new(map, width, height)
        }
    }

    impl<T> Index<Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.width)]
        }
    }

    impl<T> IndexMut<Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.width)]
        }
    }

    impl<T> Index<&Coord> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coord) -> &Self::Output {
            &self.map[coordinate.to_index(self.width)]
        }
    }

    impl<T> IndexMut<&Coord> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coord) -> &mut Self::Output {
            &mut self.map[coordinate.to_index(self.width)]
        }
    }

    impl<T> Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &self.map[begin..end]
        }
    }

    impl<T> IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
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
}
