use std::path::{Path, PathBuf};

pub trait Cachable {
    type Input;
    type Output: Clone;
    type Config;

    fn get_input(&self) -> anyhow::Result<&Self::Input>;

    fn get_output(&self) -> anyhow::Result<Self::Output>;

    fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Box<Self>>;

    fn new<P: AsRef<Path>>(
        cache_dir: P,
        input: Self::Input,
        output: Self::Output,
    ) -> anyhow::Result<(PathBuf, Box<Self>)>;

    fn matches(&self, input: &Self::Input, config: &Self::Config) -> bool;

    fn matches_file_name(file_name: String) -> bool;
}
