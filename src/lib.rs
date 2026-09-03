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

mod common;
mod core;
mod input_parser;

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::collections::HashMap;
use uuid::Uuid;

use crate::common::error::*;
use crate::core::model_handler::MODEL_HANDLER;
use crate::core::processing::worker_handler::WORKER_HANDLER;
use crate::input_parser::model_parser::parse_model_template;

#[pyclass]
pub struct Saki {}

create_exception!(saki, PySakiError, pyo3::exceptions::PyException);

impl Saki {
    pub fn init_model(&mut self, template: String) -> Result<Uuid, SakiError> {
        let model_uuid = Uuid::new_v4();
        let model_name = "test_model".to_string();

        let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        let mut parsed_model = parse_model_template(&model_name, &template)?;
        parsed_model.uuid = model_uuid;
        let _ = root_handler.init_new_model(&model_uuid, &parsed_model);

        Ok(model_uuid)
    }

    pub fn delete_model(&mut self, uuid_str: &str) -> Result<(), SakiError> {
        let model_uuid = Uuid::parse_str(uuid_str).map_err(|e| {
            SakiError::InvalidInput(format!("Failed to parse uuid with error: {e}"))
        })?;

        let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        let _ = root_handler.delete_model(&model_uuid);

        Ok(())
    }

    pub fn train(
        &mut self,
        uuid_str: &str,
        inputs: &HashMap<String, Vec<f32>>,
        outputs: &HashMap<String, Vec<f32>>,
    ) -> Result<(), SakiError> {
        let model_uuid = Uuid::parse_str(uuid_str).map_err(|e| {
            SakiError::InvalidInput(format!("Failed to parse uuid with error: {e}"))
        })?;

        let root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        let interface_mutex = root_handler.get_model_interface(&model_uuid).unwrap();
        drop(root_handler);

        let mut interface = interface_mutex.lock().expect("mutex poisoned");
        interface.train(inputs, outputs)?;

        Ok(())
    }

    pub fn request(
        &mut self,
        uuid_str: &str,
        inputs: &HashMap<String, Vec<f32>>,
        outputs: &mut HashMap<String, Vec<f32>>,
    ) -> Result<(), SakiError> {
        let model_uuid = Uuid::parse_str(uuid_str).map_err(|e| {
            SakiError::InvalidInput(format!("Failed to parse uuid with error: {e}"))
        })?;

        let root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        root_handler.reset_outputs(&model_uuid)?;
        let interface_mutex = root_handler.get_model_interface(&model_uuid).unwrap();
        drop(root_handler);

        let mut interface = interface_mutex.lock().expect("mutex poisoned");
        interface.request(inputs, outputs)?;

        Ok(())
    }

    pub fn create_checkpoint(
        &mut self,
        uuid_str: &str,
        file_path: &String,
    ) -> Result<(), SakiError> {
        let model_uuid = Uuid::parse_str(uuid_str).map_err(|e| {
            SakiError::InvalidInput(format!("Failed to parse uuid with error: {e}"))
        })?;

        let root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        root_handler
            .create_checkpoint(&model_uuid, file_path)
            .map_err(|e| {
                SakiError::InvalidInput(format!("Failed to restore checkpoint with error: {e}"))
            })?;

        Ok(())
    }

    pub fn restore_checkpoint(
        &mut self,
        uuid_str: &str,
        file_path: &String,
    ) -> Result<(), SakiError> {
        let model_uuid = Uuid::parse_str(uuid_str).map_err(|e| {
            SakiError::InvalidInput(format!("Failed to parse uuid with error: {e}"))
        })?;

        let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        root_handler
            .restore_checkpoint(&model_uuid, file_path)
            .map_err(|e| {
                SakiError::InvalidInput(format!("Failed to restore checkpoint with error: {e}"))
            })?;

        Ok(())
    }
}

#[pymethods]
impl Saki {
    #[new]
    fn new() -> Self {
        let worker_handler = WORKER_HANDLER.lock().expect("mutex poisoned");
        drop(worker_handler);
        let mut model_data_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        model_data_handler.models.clear();
        drop(model_data_handler);

        Saki {}
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[pyo3(name = "init_model")]
    fn py_init_model(&mut self, template: String) -> Result<String, PyErr> {
        let model_uuid = self.init_model(template).map_err(PyErr::from)?;
        Ok(model_uuid.to_string())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[pyo3(name = "delete_model")]
    fn py_delete_model(&mut self, model_uuid: String) -> Result<(), PyErr> {
        self.delete_model(&model_uuid).map_err(PyErr::from)?;
        Ok(())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[pyo3(name = "train")]
    fn py_train(
        &mut self,
        model_uuid: String,
        inputs: HashMap<String, Vec<f32>>,
        outputs: HashMap<String, Vec<f32>>,
    ) -> Result<(), PyErr> {
        self.train(&model_uuid, &inputs, &outputs)
            .map_err(PyErr::from)?;

        Ok(())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[pyo3(name = "request")]
    fn py_request(
        &mut self,
        model_uuid: String,
        inputs: HashMap<String, Vec<f32>>,
        outputs_dict: &Bound<'_, PyDict>,
    ) -> Result<(), PyErr> {
        let mut rust_outputs: HashMap<String, Vec<f32>> = outputs_dict.extract()?;
        self.request(&model_uuid, &inputs, &mut rust_outputs)
            .map_err(PyErr::from)?;

        for (k, v) in rust_outputs {
            outputs_dict.set_item(k, v)?;
        }

        Ok(())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[pyo3(name = "create_checkpoint")]
    fn py_create_checkpoint(&mut self, model_uuid: String, file_path: String) -> Result<(), PyErr> {
        self.create_checkpoint(&model_uuid, &file_path)
            .map_err(PyErr::from)?;

        Ok(())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    #[pyo3(name = "restore_checkpoint")]
    fn py_restore_checkpoint(
        &mut self,
        model_uuid: String,
        file_path: String,
    ) -> Result<(), PyErr> {
        self.restore_checkpoint(&model_uuid, &file_path)
            .map_err(PyErr::from)?;

        Ok(())
    }
}

// Don't forget to add the class to your module!
#[pymodule]
fn saki(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Saki>()?;
    m.add("SakiError", m.py().get_type::<PySakiError>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::input_parser::model_parser::parse_model_template;
    use serial_test::serial;

    use crate::core::model_handler::MODEL_HANDLER;
    use crate::core::processing::worker_handler::WORKER_HANDLER;

    use super::*;

    #[test]
    #[serial]
    fn test_workflow() {
        // Initialize processing
        let worker_handler = WORKER_HANDLER.lock().expect("mutex poisoned");
        drop(worker_handler);
        let model_data_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        drop(model_data_handler);

        // create dummy-model
        let model_uuid = Uuid::new_v4();
        let model_name = "test_model".to_string();

        let input1 = vec![1.0f32, 2.0f32, -3.0f32, 4.0f32];
        let mut input_map1 = HashMap::new();
        input_map1.insert("test_input".to_string(), input1);
        let expected1 = vec![1.0f32, 1.0f32, 0.0f32, 1.0f32];
        let mut expected_map1 = HashMap::new();
        expected_map1.insert("test_output".to_string(), expected1);

        let template = "version: 1 
        settings:
            neuron_cooldown: 1000000000.0;
            refractory_time: 1;
            max_connection_distance: 1;
        hexagons: 
            1,1,1; 
            2,2,2; 
            3,2,2; 
        axons: 
            1,1,1 -> 2,2,2; 
        inputs: 
            test_input: 1,1,1; 
        outputs: 
            test_output: 3,2,2;"
            .to_string();

        let mut root_handler = MODEL_HANDLER.write().expect("mutex poisoned");
        root_handler.models.clear();
        let mut parsed_model = parse_model_template(&model_name, &template).unwrap();
        parsed_model.uuid = model_uuid;
        let _ = root_handler.init_new_model(&model_uuid, &parsed_model);
        let interface_mutex = root_handler.get_model_interface(&model_uuid).unwrap();
        drop(root_handler);

        let mut interface = interface_mutex.lock().expect("mutex poisoned");
        let _ = interface.train(&input_map1, &expected_map1);

        println!("finished");
    }
}
