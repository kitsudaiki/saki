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

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::fmt;

#[derive(Debug)]
pub enum SakiError {
    InvalidInput(String),
    InternalError(String),
}

impl fmt::Display for SakiError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SakiError::InvalidInput(ref msg) => write!(f, "Invalid input: {msg}"),
            SakiError::InternalError(ref msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl PartialEq<&str> for SakiError {
    fn eq(&self, other: &&str) -> bool {
        match self {
            SakiError::InvalidInput(s) | SakiError::InternalError(s) => s == other,
        }
    }
}

impl From<SakiError> for PyErr {
    fn from(err: SakiError) -> PyErr {
        match err {
            SakiError::InvalidInput(msg) => PyValueError::new_err(msg),
            SakiError::InternalError(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

impl std::error::Error for SakiError {}
