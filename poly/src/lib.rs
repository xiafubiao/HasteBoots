#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
//! Multivariant polynomials
mod data_structures;
mod multilinear;

pub use data_structures::{ListOfProductsOfPolynomials, PolynomialInfo};
pub use multilinear::{DenseMultilinearExtension, MultilinearExtension, SparsePolynomial};
