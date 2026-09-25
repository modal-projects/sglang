use sha2::{Digest, Sha256};

use crate::pipeline::{Geometry, ProcessedItem, Tensor, TensorData};

struct Identity(Sha256);

impl Identity {
    fn number(&mut self, value: usize) {
        self.0.update((value as u64).to_le_bytes());
    }

    fn bytes(&mut self, value: &[u8]) {
        self.number(value.len());
        self.0.update(value);
    }

    fn tensor(&mut self, tensor: &Tensor, include_data: bool) {
        self.number(tensor.shape.len());
        for &dimension in &tensor.shape {
            self.number(dimension);
        }
        match &tensor.data {
            TensorData::F32(values) => {
                self.bytes(b"f32-le");
                self.number(values.len());
                if include_data {
                    for value in values {
                        self.0.update(value.to_le_bytes());
                    }
                }
            }
            TensorData::I64(values) => {
                self.bytes(b"i64-le");
                self.number(values.len());
                if include_data {
                    for value in values {
                        self.0.update(value.to_le_bytes());
                    }
                }
            }
        }
    }
}

pub(crate) fn content_identity(
    source: &[u8],
    config: Option<&[u8]>,
    item: &ProcessedItem,
) -> String {
    let mut identity = Identity(Sha256::new());
    identity.bytes(b"sglang-mm-content-v1");
    identity.bytes(source);
    match config {
        Some(config) => {
            identity.bytes(b"deterministic-config");
            identity.bytes(config);
        }
        None => identity.bytes(b"processed-output"),
    }
    // A complete deterministic config and source determine the large feature
    // buffer. Families without that contract hash the output itself.
    identity.tensor(&item.feature, config.is_none());
    let mut auxiliary = item.aux.iter().collect::<Vec<_>>();
    auxiliary.sort_by(|(left, _), (right, _)| left.cmp(right));
    identity.number(auxiliary.len());
    for (name, tensor) in auxiliary {
        identity.bytes(name.as_bytes());
        identity.tensor(tensor, true);
    }
    match item.geometry {
        Geometry::Grid(grid) => {
            identity.bytes(b"grid");
            for dimension in grid {
                identity.0.update(dimension.to_le_bytes());
            }
        }
    }
    format!("sha256:{:x}", identity.0.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item() -> ProcessedItem {
        ProcessedItem {
            feature: Tensor {
                shape: vec![2, 2],
                data: TensorData::F32(vec![1.0, 2.0, 3.0, 4.0]),
            },
            aux: vec![],
            geometry: Geometry::Grid([1, 1, 2]),
        }
    }

    #[test]
    fn frames_source_configuration_and_grid_separately() {
        let mut value = item();
        let original = content_identity(b"image-a", Some(b"config-b"), &value);
        assert!(original.starts_with("sha256:"));
        assert_eq!(original.len(), 71);
        assert_eq!(
            original,
            content_identity(b"image-a", Some(b"config-b"), &value)
        );
        assert_ne!(
            original,
            content_identity(b"image-b", Some(b"config-b"), &value)
        );
        assert_ne!(
            original,
            content_identity(b"image-a", Some(b"config-c"), &value)
        );
        assert_ne!(
            content_identity(b"ab", Some(b"c"), &value),
            content_identity(b"a", Some(b"bc"), &value)
        );
        value.geometry = Geometry::Grid([1, 2, 1]);
        assert_ne!(
            original,
            content_identity(b"image-a", Some(b"config-b"), &value)
        );
    }

    #[test]
    fn fallback_covers_feature_bits_shape_and_auxiliary_tensors() {
        let original = content_identity(b"source", None, &item());
        let mut value = item();
        value.feature.data = TensorData::F32(vec![1.0, 2.0, 3.0, 5.0]);
        assert_ne!(original, content_identity(b"source", None, &value));
        value = item();
        value.feature.shape = vec![1, 4];
        assert_ne!(original, content_identity(b"source", None, &value));
        value = item();
        value.aux.push((
            "image_grid_thw".into(),
            Tensor {
                shape: vec![3],
                data: TensorData::I64(vec![1, 1, 2]),
            },
        ));
        assert_ne!(original, content_identity(b"source", None, &value));
        let with_aux = content_identity(b"source", Some(b"config"), &value);
        value.aux[0].1.data = TensorData::I64(vec![1, 2, 1]);
        assert_ne!(
            with_aux,
            content_identity(b"source", Some(b"config"), &value)
        );
    }
}
