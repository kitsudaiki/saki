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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use uuid::Uuid;

use crate::common::error::SakiError;

use crate::core::blocks::block_trait::*;
use crate::core::model_handler::*;
use crate::core::processing::finish_counter::FinishCounter;

use super::processing::output_buffer::*;
use super::processing::worker_queue::*;

/// Represents an interface to interact with a neural network model.
///
/// This struct manages the execution of tasks on a model, including processing
/// inputs and outputs, training, and monitoring task completion.
pub struct ModelInterface {
    pub finish_counter_mutex: Arc<Mutex<FinishCounter>>,
    pub model_uuid: Uuid,
}

impl ModelInterface {
    /// Creates a new ModelInterface instance.
    ///
    /// # Arguments
    ///
    /// * `model_uuid` - Unique identifier for the model
    /// * `finish_counter_mutex` - Shared counter for tracking task completion
    ///
    /// # Returns
    ///
    /// A new ModelInterface instance with a running worker thread
    pub fn new(model_uuid: &Uuid, finish_counter_mutex: &Arc<Mutex<FinishCounter>>) -> Self {
        ModelInterface {
            model_uuid: *model_uuid,
            finish_counter_mutex: finish_counter_mutex.clone(),
        }
    }

    /// Applies plain input data to a model's input block.
    ///
    /// This function takes raw input data and applies it to the specified input block of a model.
    /// It's primarily used for direct input application rather than dataset-based input.
    ///
    /// # Arguments
    ///
    /// * `model_uuid` - Unique identifier for the model
    /// * `hexagon_name` - Name of the hexagon (input block) to apply data to
    /// * `input_ptr` - Pointer to the input data
    /// * `input_size` - Size of the input data
    /// * `pos_counter` - Position counter for the input data
    /// * `task_type` - Type of worker task (Train or Process)
    ///
    /// # Returns
    ///
    /// * `Result<(), SakiError>` - Returns Ok(()) on success, or an SakiError on failure
    pub fn apply_plain_input(
        &self,
        model_uuid: &Uuid,
        hexagon_name: &String,
        input_ptr: &[f32],
        input_size: u64,
        pos_counter: usize,
        task_type: &WorkerTaskType,
    ) -> Result<(), SakiError> {
        let model_handler = MODEL_HANDLER.read().expect("mutex poisoned");
        let input_block_mutex = model_handler.get_input_block(model_uuid, hexagon_name)?;

        let mut input_block = input_block_mutex.lock().expect("mutex poisoned");
        let allow_creation = *task_type == WorkerTaskType::Train;
        input_block.apply_input(input_ptr, input_size as usize, pos_counter, allow_creation);

        let mut worker_queue = WORKER_QUEUE.lock().expect("mutex poisoned");
        let cycle_number = 0;
        let worker_task = WorkerTask {
            task_type: task_type.clone(),
            block: Arc::clone(&input_block_mutex) as Arc<Mutex<dyn Block>>,
            cycle_number,
        };
        worker_queue.add(worker_task);

        Ok(())
    }

    /// Applies expected output data to a model's output buffer.
    ///
    /// This function takes raw output data and applies it to the specified output buffer of a model.
    /// It's primarily used for setting expected outputs for training purposes.
    ///
    /// # Arguments
    ///
    /// * `model_uuid` - Unique identifier for the model
    /// * `hexagon_name` - Name of the hexagon (output buffer) to apply data to
    /// * `input_ptr` - Pointer to the output data
    /// * `input_size` - Size of the output data
    ///
    /// # Returns
    ///
    /// * `Result<(), SakiError>` - Returns Ok(()) on success, or an SakiError on failure
    pub fn apply_expected(
        &self,
        model_uuid: &Uuid,
        hexagon_name: &String,
        input_ptr: &[f32],
        input_size: u64,
    ) -> Result<(), SakiError> {
        let model_handler = MODEL_HANDLER.read().expect("mutex poisoned");
        let output_buffer_mutex = model_handler.get_output_buffer(model_uuid, hexagon_name)?;

        let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
        output_buffer.reset_output();
        convert_buffer_to_expected(&mut output_buffer, input_ptr, input_size);

        Ok(())
    }

    /// Processes inputs through the model and returns outputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input names to their corresponding data
    /// * `outputs` - Map of output names to buffers that will be filled with results
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the operation
    pub fn request(
        &mut self,
        inputs: &HashMap<String, Vec<f32>>,
        outputs: &mut HashMap<String, Vec<f32>>,
    ) -> Result<(), SakiError> {
        let mut counter = self.finish_counter_mutex.lock().expect("mutex poisoned");
        let task_compare = counter.output_compare;
        counter.reset(task_compare, 0);
        drop(counter);

        // reset output-values in the backend
        {
            let model_data_handler = MODEL_HANDLER.read().expect("mutex poisoned");
            for hexagon_name in outputs.keys() {
                let output_buffer_mutex =
                    model_data_handler.get_output_buffer(&self.model_uuid, hexagon_name)?;
                let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
                output_buffer.reset_output();
            }
        }

        for (hexagon_name, data) in inputs {
            self.apply_plain_input(
                &self.model_uuid,
                hexagon_name,
                data.as_slice(),
                data.len() as u64,
                0,
                &WorkerTaskType::Process,
            )?;
        }

        self.run_iteration(&self.model_uuid, &self.finish_counter_mutex)?;

        // get output-values from the backend
        let model_data_handler = MODEL_HANDLER.read().expect("mutex poisoned");
        for (hexagon_name, data) in outputs.iter_mut() {
            let output_buffer_mutex =
                model_data_handler.get_output_buffer(&self.model_uuid, hexagon_name)?;

            let mut output_buffer = output_buffer_mutex.lock().expect("mutex poisoned");
            convert_output_to_buffer(data, &mut output_buffer);
        }

        Ok(())
    }

    /// Trains the model using the provided inputs and expected outputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Map of input names to their corresponding data
    /// * `outputs` - Map of output names to their expected values
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the operation
    pub fn train(
        &mut self,
        inputs: &HashMap<String, Vec<f32>>,
        outputs: &HashMap<String, Vec<f32>>,
    ) -> Result<(), SakiError> {
        let mut counter = self.finish_counter_mutex.lock().expect("mutex poisoned");
        let task_compare = counter.input_compare + counter.output_compare;
        counter.reset(task_compare, 0);
        drop(counter);

        for (hexagon_name, data) in outputs {
            let _ = self.apply_expected(
                &self.model_uuid,
                hexagon_name,
                data.as_slice(),
                data.len() as u64,
            );
        }

        for (hexagon_name, data) in inputs {
            self.apply_plain_input(
                &self.model_uuid,
                hexagon_name,
                data.as_slice(),
                data.len() as u64,
                0,
                &WorkerTaskType::Train,
            )?;
        }

        self.run_iteration(&self.model_uuid, &self.finish_counter_mutex)?;

        Ok(())
    }

    /// Executes a single iteration of model processing.
    ///
    /// This function waits for all tasks to complete or times out after a certain number of iterations.
    ///
    /// # Arguments
    ///
    /// * `model_uuid` - Unique identifier for the model
    /// * `finish_counter_mutex` - Shared counter for tracking task completion
    ///
    /// # Returns
    ///
    /// Result indicating success or failure of the operation
    fn run_iteration(
        &self,
        model_uuid: &Uuid,
        finish_counter_mutex: &Arc<Mutex<FinishCounter>>,
    ) -> Result<(), SakiError> {
        for _ in 0..10000000 {
            let finish_counter = finish_counter_mutex.lock().expect("mutex poisoned");
            if finish_counter.is_finished() {
                return Ok(());
            }
            drop(finish_counter);
            thread::sleep(std::time::Duration::from_micros(1));
        }

        let msg = format!("Timeout while processing model with uuid {model_uuid}");
        Err(SakiError::InternalError(msg))
    }
}

#[cfg(test)]
mod tests {

    // fn run_single_iteration(
    //     model_uuid: &Uuid,
    //     finish_counter_mutex: &Arc<Mutex<FinishCounter>>,
    //     input: &[f32; 4],
    //     expected: &[f32; 4],
    // ) {
    //     let input_name = "test_input".to_string();
    //     let output_name = "test_output".to_string();

    //     let mut counter = finish_counter_mutex.lock().expect("mutex poisoned");
    //     let task_compare = counter.input_compare + counter.output_compare;
    //     counter.reset(task_compare, 0);
    //     drop(counter);

    //     match self.apply_plain_input(
    //         model_uuid,
    //         &input_name,
    //         input,
    //         input.len() as u64,
    //         0,
    //         1,
    //         &WorkerTaskType::Train,
    //     ) {
    //         Ok(()) => {}
    //         Err(e) => {
    //             println!("{e}");
    //             panic!();
    //         }
    //     }

    //     match self.apply_expected(model_uuid, &output_name, expected, expected.len() as u64) {
    //         Ok(()) => {}
    //         Err(e) => {
    //             println!("{e}");
    //             panic!();
    //         }
    //     }

    //     match self.run_iteration(model_uuid, finish_counter_mutex) {
    //         Ok(()) => {}
    //         Err(e) => {
    //             println!("{e}");
    //             panic!();
    //         }
    //     }
    // }

    // #[test]
    // #[serial]
    // fn test_workflow() {
    //     // Initialize processing
    //     let worker_handler = WORKER_HANDLER.lock().expect("mutex poisoned");
    //     drop(worker_handler);
    //     let model_data_handler = MODEL_HANDLER.write().expect("mutex poisoned");
    //     drop(model_data_handler);

    //     // create dummy-model
    //     let model_uuid = Uuid::new_v4();
    //     let model_name = "test_model".to_string();
    //     let input1 = [1.0f32, 2.0f32, -3.0f32, 4.0f32];
    //     let expected1 = [1.0f32, 1.0f32, 0.0f32, 1.0f32];

    //     let input2 = [5.0f32, -1.0f32, 8.0f32, -4.0f32];
    //     let expected2 = [0.0f32, 1.0f32, 1.0f32, 0.0f32];

    //     let template = "version: 1
    //     settings:
    //         neuron_cooldown: 1000000000.0;
    //         refractory_time: 1;
    //         max_connection_distance: 1;
    //     hexagons:
    //         1,1,1;
    //         2,2,2;
    //         3,2,2;
    //     axons:
    //         1,1,1 -> 2,2,2;
    //     inputs:
    //         test_input: 1,1,1;
    //     outputs:
    //         test_output: 3,2,2;"
    //         .to_string();

    //     let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
    //     root_handler.models.clear();
    //     let mut parsed_model = parse_model_template(&model_name, &template).unwrap();
    //     parsed_model.uuid = model_uuid;
    //     let _ = root_handler.init_new_model(&model_uuid, &parsed_model);
    //     let finish_counter_mutex = root_handler.get_finish_counter(&model_uuid).unwrap();
    //     drop(root_handler);

    //     for _ in 0..100 {
    //         run_single_iteration(&model_uuid, &finish_counter_mutex, &input1, &expected1);
    //         run_single_iteration(&model_uuid, &finish_counter_mutex, &input2, &expected2);
    //     }

    //     println!("finished");
    // }
}
