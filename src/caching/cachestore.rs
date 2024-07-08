use log::warn;
use std::any::type_name;
use std::fs;
use std::ops::Deref;
use std::path::PathBuf;
use tokio::sync::RwLock;

use crate::caching::cachable::Cachable;

pub struct CacheStore<T>
where
    T: Cachable,
{
    // The path where cache is stored on disk.
    dir: PathBuf,

    // The in-memory store.
    store: RwLock<Vec<Box<T>>>,
}

impl<T> CacheStore<T>
where
    T: Cachable,
    T: Clone,
{
    pub fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            store: Default::default(),
        }
    }

    pub async fn store(&self, input: T::Input, output: T::Output) -> anyhow::Result<(PathBuf, T)> {
        let (path, cachable) = match T::new(&self.dir, input, output) {
            Ok((path, cachable)) => (path, cachable),
            Err(err) => return Err(err),
        };

        let mut writable_store = self.store.write().await;
        writable_store.push(cachable.clone());

        Ok((path, *cachable))
    }

    // Loads all inference files from the inference store path.
    pub async fn load(&self) -> anyhow::Result<()> {
        let mut write_store = self.store.write().await;

        fs::read_dir(&self.dir)?
            .filter_map(Result::ok)
            .filter(|entry| {
                T::matches_file_name(
                    entry
                        .path()
                        .file_name()
                        .unwrap()
                        .to_os_string()
                        .into_string()
                        .unwrap(),
                )
            })
            .map(|r| r.path())
            .filter_map(|p| T::from_file(p).ok())
            .for_each(|c| write_store.push(c));

        Ok(())
    }

    pub async fn find_output(
        &self,
        match_input: &T::Input,
        config: &T::Config,
    ) -> Option<T::Output> {
        let readable_store = self.store.read().await;

        for cachable in readable_store.deref() {
            if cachable.matches(match_input, config) {
                match cachable.get_output() {
                    Ok(o) => return Some(o),
                    Err(err) => warn!("error encountered during the output fetching of a match in {} cachestore: {err}", type_name::<T>().rsplit("::").next().unwrap())
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use crate::caching::cachable::Cachable;
    use crate::caching::cachestore::CacheStore;
    use std::fs::File;
    use std::path::{Path, PathBuf};
    use tempdir::TempDir;

    #[derive(Clone)]
    struct TestCachable {
        input: u8,
        output: u8,
    }

    impl Cachable for TestCachable {
        type Input = u8;
        type Output = u8;
        type Config = ();

        fn get_input(&self) -> anyhow::Result<&Self::Input> {
            return Ok(&self.input);
        }

        fn get_output(&self) -> anyhow::Result<Self::Output> {
            return Ok(self.output.clone());
        }

        fn from_file<P: AsRef<Path>>(path: P) -> anyhow::Result<Box<Self>> {
            // Extract the file stem.
            let input = path
                .as_ref()
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .parse::<u8>()?;

            // Read string content from file.
            let output = std::fs::read_to_string(&path)?.parse::<u8>()?;

            Ok(Box::new(TestCachable { input, output }))
        }

        fn new<P: AsRef<Path>>(
            cache_dir: P,
            input: Self::Input,
            output: Self::Output,
        ) -> anyhow::Result<(PathBuf, Box<Self>)> {
            let path = cache_dir.as_ref().join(format!("{input}.test"));

            // Write the output to the file as text.
            File::create(&path)?;
            std::fs::write(&path, output.to_string())?;

            Ok((path, Box::new(TestCachable { input, output })))
        }

        fn matches(&self, input: &Self::Input, _config: &Self::Config) -> bool {
            self.input == *input
        }

        fn matches_file_name(file_name: String) -> bool {
            file_name.ends_with(".test")
        }
    }

    #[tokio::test]
    async fn it_stores() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();
        let cache_store = CacheStore::<TestCachable>::new(tmp_path.clone());

        let (path, cachable) = cache_store.store(1, 2).await.unwrap();
        assert_eq!(path, tmp_path.join("1.test"));
        assert_eq!(1, cachable.input);
        assert_eq!(2, cachable.output);
    }

    #[tokio::test]
    async fn it_loads() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();

        // Create a file.
        let path = tmp_path.join("1.test");
        File::create(&path).unwrap();
        std::fs::write(&path, "2").unwrap();

        // Load the file.
        let cache_store = CacheStore::<TestCachable>::new(tmp_path.clone());
        cache_store.load().await.unwrap();

        let readable_store = cache_store.store.read().await;
        let first_item = readable_store.first().unwrap();
        assert_eq!(1, first_item.input);
        assert_eq!(2, first_item.output);
    }

    #[tokio::test]
    async fn it_matches() {
        let tmp_dir = TempDir::new("inference_store_test").unwrap();
        let tmp_path = tmp_dir.path().to_path_buf();
        let cache_store = CacheStore::<TestCachable>::new(tmp_path.clone());

        let _ = cache_store.store(1, 2).await.unwrap();

        let output = cache_store.find_output(&1, &()).await.unwrap();

        assert_eq!(2, output);
    }
}
