use crate::planning::Choices;
use oh_my_vllm_kv_cache::group::RequestId;
use rustc_hash::FxHashMap;

type Contribution = (usize, usize, usize, usize);
type Objective = (usize, usize, usize, std::cmp::Reverse<usize>);

pub(super) fn select_aligned(
    choices: &[Choices],
    budget: usize,
    block: usize,
    slots: usize,
    limited: bool,
) -> Option<FxHashMap<RequestId, usize>> {
    let decodes = choices.iter().filter(|choice| choice.decode).count();
    let suffixes = choices.len() - decodes;
    let blocks = budget / block;
    let admissions = if limited {
        slots
    } else {
        0
    };
    let shape = [decodes + 1, blocks + 1, suffixes + 1, admissions + 1];
    if shape.iter().product::<usize>() > 65536 {
        return None;
    }
    let plan = AlignedPlan {
        budget,
        block,
        limited,
        shape,
    };
    // For fixed decode/block/suffix counts, all token counts between the minimum
    // and the stored maximum are feasible. Keep this capacity, rather than token states.
    let options = options(choices, block);
    let suffix = plan.build_suffix(choices, &options);
    let best = plan.best(suffix.last().unwrap());
    plan.reconstruct(choices, &options, &suffix, best)
}

fn options(choices: &[Choices], block: usize) -> Vec<Vec<Contribution>> {
    choices
        .iter()
        .map(|choice| {
            let maximum = choice.ranges[0].1;
            let mut options = vec![(0, 0, 0, 0)];
            if choice.decode {
                options.push((1, 0, 0, maximum));
            } else {
                for b in 0..=maximum / block {
                    if b > 0 {
                        options.push((0, b, 0, b * block));
                    }
                    let end = maximum.min(b * block + block - 1);
                    if end > b * block {
                        options.push((0, b, 1, end));
                    }
                }
            }
            options
        })
        .collect::<Vec<_>>()
}

struct AlignedPlan {
    budget: usize,
    block: usize,
    limited: bool,
    shape: [usize; 4],
}

impl AlignedPlan {
    fn index(&self, d: usize, b: usize, s: usize, w: usize) -> usize {
        ((d * self.shape[1] + b) * self.shape[2] + s) * self.shape[3] + w
    }

    fn build_suffix(&self, choices: &[Choices], options: &[Vec<Contribution>]) -> Vec<Vec<usize>> {
        let states = self.shape.iter().product::<usize>();
        let [decodes, blocks, suffixes, admissions] = self.shape.map(|count| count - 1);
        let mut suffix = vec![vec![usize::MAX; states]];
        suffix[0][0] = 0;
        for (choice, options) in choices.iter().zip(options).rev() {
            let previous = suffix.last().unwrap();
            let mut next = vec![usize::MAX; states];
            for d in 0..=decodes {
                for b in 0..=blocks {
                    for s in 0..=suffixes {
                        for w in 0..=admissions {
                            let capacity = previous[self.index(d, b, s, w)];
                            if capacity == usize::MAX {
                                continue;
                            }
                            for &(add_d, add_b, add_s, count) in options {
                                let add_w =
                                    usize::from(self.limited && choice.waiting && count > 0);
                                let (d, b, s, w) = (d + add_d, b + add_b, s + add_s, w + add_w);
                                if d > decodes
                                    || b > blocks
                                    || s > suffixes
                                    || w > admissions
                                    || d + b * self.block + s > self.budget
                                {
                                    continue;
                                }
                                let next = &mut next[self.index(d, b, s, w)];
                                let capacity = (capacity
                                    + if add_d > 0 {
                                        0
                                    } else {
                                        count
                                    })
                                .min(self.budget - d);
                                if *next == usize::MAX || capacity > *next {
                                    *next = capacity;
                                }
                            }
                        }
                    }
                }
            }
            suffix.push(next);
        }
        suffix
    }

    fn best(&self, capacities: &[usize]) -> Objective {
        let [decodes, blocks, suffixes, admissions] = self.shape.map(|count| count - 1);
        let mut best = (0, 0, 0, std::cmp::Reverse(0));
        for d in 0..=decodes {
            for b in 0..=blocks {
                for s in 0..=suffixes {
                    for w in 0..=admissions {
                        let capacity = capacities[self.index(d, b, s, w)];
                        if capacity != usize::MAX {
                            best = best.max((d, b, capacity, std::cmp::Reverse(s)));
                        }
                    }
                }
            }
        }
        best
    }

    fn reconstruct(
        &self,
        choices: &[Choices],
        options: &[Vec<Contribution>],
        suffix: &[Vec<usize>],
        best: Objective,
    ) -> Option<FxHashMap<RequestId, usize>> {
        let (mut d, mut b, mut tokens, std::cmp::Reverse(mut s)) = best;
        let mut remaining_budget = self.budget;
        let mut available = self.shape[3] - 1;
        let mut result = FxHashMap::default();
        for (i, (choice, options)) in choices.iter().zip(options).enumerate() {
            let tail = &suffix[choices.len() - i - 1];
            let mut selected: Option<(usize, usize, usize, usize, usize)> = None;
            for &(take_d, take_b, take_s, maximum) in options {
                let charge = usize::from(self.limited && choice.waiting && maximum > 0);
                if take_d > d || take_b > b || take_s > s || charge > available {
                    continue;
                }
                let minimum = take_d + take_b * self.block + take_s;
                let remainder_min = (b - take_b) * self.block + s - take_s;
                if remainder_min > tokens {
                    continue;
                }
                let count = if take_d > 0 {
                    maximum.min(remaining_budget - (d - take_d) - tokens)
                } else {
                    maximum.min(tokens - remainder_min)
                };
                if count < minimum {
                    continue;
                }
                for used in 0..=available - charge {
                    let capacity = tail[self.index(d - take_d, b - take_b, s - take_s, used)];
                    let prefill = if take_d > 0 {
                        0
                    } else {
                        count
                    };
                    if capacity != usize::MAX
                        && tokens - prefill <= capacity
                        && selected.is_none_or(|(old, ..)| count > old)
                    {
                        selected = Some((count, take_d, take_b, take_s, charge));
                    }
                }
            }
            let (count, take_d, take_b, take_s, charge) =
                selected.expect("aligned capacity is feasible");
            result.insert(choice.id, count);
            d -= take_d;
            b -= take_b;
            s -= take_s;
            if take_d == 0 {
                tokens -= count;
            }
            remaining_budget -= count;
            available -= charge;
        }
        Some(result)
    }
}
