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

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::common::constants::*;
use crate::common::enums::*;
use crate::common::objects::*;

/// Configuration settings for the neural network model
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub neuron_cooldown: f32,
    pub refractory_time: u32,
    pub max_connection_distance: u32,
}

/// Metadata for connections between neurons (axons)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxonMeta {
    pub from: Position,
    pub to: Position,
}

/// Metadata for hexagonal neurons in the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HexagonMeta {
    pub uuid: Uuid,
    pub positon: Position,
    pub name: String,

    pub is_input: bool,
    pub is_output: bool,

    pub axon_target: Uuid,

    pub possible_hexagon_target_ids: Vec<Uuid>,
    pub neighbors: [Uuid; 12],
}

impl HexagonMeta {
    /// Creates a new HexagonMeta instance with default values
    ///
    /// # Arguments
    /// * `positon` - The initial position of the hexagon
    ///
    /// # Returns
    /// A new HexagonMeta instance with a generated UUID and default values
    pub fn new(positon: Position) -> Self {
        let new_uuid = Uuid::new_v4();
        HexagonMeta {
            uuid: new_uuid,
            positon,
            name: "".to_string(),

            axon_target: new_uuid, // Default to its own UUID

            is_input: false,
            is_output: false,

            neighbors: [Uuid::nil(); 12],
            possible_hexagon_target_ids: vec![Uuid::nil(); NUMBER_OF_POSSIBLE_NEXT],
        }
    }
}

/// Metadata for input connections to the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputMeta {
    pub uuid: Uuid,
    pub hexagon_uuid: Uuid,
    pub name: String,
    pub position: Position,
}

impl InputMeta {
    /// Creates a new InputMeta instance
    ///
    /// # Arguments
    /// * `name` - The name of the input
    /// * `position` - The position of the input in the network
    ///
    /// # Returns
    /// A new InputMeta instance with a generated UUID and nil hexagon UUID
    pub fn new(name: String, position: Position) -> Self {
        InputMeta {
            uuid: Uuid::new_v4(),
            hexagon_uuid: Uuid::nil(), // Not connected to a hexagon by default
            name,
            position,
        }
    }
}

/// Metadata for output connections from the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMeta {
    pub uuid: Uuid,
    pub hexagon_uuid: Uuid,
    pub name: String,
    pub position: Position,
    pub output_type: OutputType,
}

impl OutputMeta {
    /// Creates a new OutputMeta instance
    ///
    /// # Arguments
    /// * `name` - The name of the output
    /// * `position` - The position of the output in the network
    /// * `output_type` - The type of data this output produces
    ///
    /// # Returns
    /// A new OutputMeta instance with a generated UUID and nil hexagon UUID
    pub fn new(name: String, position: Position, output_type: OutputType) -> Self {
        OutputMeta {
            uuid: Uuid::new_v4(),
            hexagon_uuid: Uuid::nil(), // Not connected to a hexagon by default
            name,
            position,
            output_type,
        }
    }
}

/// Metadata for the entire neural network model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMeta {
    pub uuid: Uuid,
    pub name: String,
    pub version: i32,
    pub settings: Settings,
    pub hexagons: HashMap<Uuid, HexagonMeta>,
    pub axons: Vec<AxonMeta>,
    pub inputs: Vec<InputMeta>,
    pub outputs: Vec<OutputMeta>,
}
