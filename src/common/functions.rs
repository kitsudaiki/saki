// Copyright 2026-2026 Tobias Anker <tobias.anker@kitsunemimi.moe>

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::objects::*;

/// Computes a PCG (Permuted Congruential Generator) hash of the given u32 value.
///
/// This is a fast, non-cryptographic hash function suitable for general-purpose use.
/// The function updates the input value in place and returns the computed hash.
#[inline]
pub fn pcg_hash(input: &mut u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    *input = (word >> 22) ^ word;
    *input
}

/// Calculates the position of a neighboring cell in a hexagonal grid.
///
/// Given a source position and a side number (0-11), returns the position of the adjacent cell.
/// The side numbering follows a specific pattern used in hexagonal grid algorithms.
///
/// # Arguments
///
/// * `source_pos` - The position of the source cell
/// * `side` - The side number (0-11) indicating which neighbor to get
///
/// # Panics
///
/// Panics if the side value is out of the valid range (0-11).
pub fn get_neighbor_pos(source_pos: &Position, side: usize) -> Position {
    let mut result = Position { x: 0, y: 0, z: 0 };

    match side {
        0 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z - 1;
        }
        1 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z - 1;
        }
        2 => {
            result.x = source_pos.x;
            result.y = source_pos.y;
            result.z = source_pos.z - 1;
        }
        3 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z;
        }
        4 => {
            result.x = source_pos.x + 1;
            result.y = source_pos.y;
            result.z = source_pos.z;
        }
        5 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z;
        }
        6 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y - 1;
            result.z = source_pos.z;
        }
        7 => {
            result.x = source_pos.x - 1;
            result.y = source_pos.y;
            result.z = source_pos.z;
        }
        8 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z;
        }
        9 => {
            result.x = source_pos.x;
            result.y = source_pos.y;
            result.z = source_pos.z + 1;
        }
        10 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x - 1
            } else {
                source_pos.x
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z + 1;
        }
        11 => {
            result.x = if source_pos.y % 2 == 0 {
                source_pos.x
            } else {
                source_pos.x + 1
            };
            result.y = source_pos.y + 1;
            result.z = source_pos.z + 1;
        }
        _ => panic!("Invalid side value: {side}"),
    }

    result
}
