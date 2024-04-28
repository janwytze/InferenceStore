use std::collections::HashMap;
use std::hash::Hash;

/// Compare two hashmaps based on the provided keys. The `include_keys` argument determines if the
/// keys should be included or excluded.
///
/// # Arguments
///
/// * `map1` - The first map to compare.
/// * `map2` - The second map to compare.
/// * `keys_to_compare` - The keys that should be compared or should not be compared.
/// * `exclude_keys` - When false the keys provided are compared, when true the keys provided are
/// not compared.
///
pub fn hashmap_compare<K, V>(
    map1: HashMap<K, V>,
    map2: HashMap<K, V>,
    keys_to_compare: Vec<K>,
    exclude_keys: bool,
) -> bool
where
    K: Eq + Hash,
    V: PartialEq,
{
    if exclude_keys {
        let map1_filtered: HashMap<_, _> = map1
            .iter()
            .filter(|(key, _)| !keys_to_compare.contains(key))
            .collect();
        let map2_filtered: HashMap<_, _> = map2
            .iter()
            .filter(|(key, _)| !keys_to_compare.contains(key))
            .collect();
        map1_filtered == map2_filtered
    } else {
        keys_to_compare
            .iter()
            .all(|key| map1.get(key) == map2.get(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_compares_using_excluding_keys() {
        let a = HashMap::from([("hoi".to_string(), "doei".to_string())]);
        let b = HashMap::from([("hoi".to_string(), "doei".to_string())]);
        assert!(hashmap_compare(a, b, Vec::new(), true));
    }

    #[test]
    fn it_compares_using_including_keys() {
        let a = HashMap::from([("hoi".to_string(), "doei".to_string())]);
        let b = HashMap::from([("hoi".to_string(), "doei".to_string())]);
        assert!(hashmap_compare(a, b, vec!["hoi".to_string()], false));
    }

    #[test]
    fn it_excludes_keys() {
        let a = HashMap::from([
            ("hoi".to_string(), "doei".to_string()),
            ("hoi2".to_string(), "doei".to_string()),
        ]);
        let b = HashMap::from([("hoi".to_string(), "doei".to_string())]);
        assert!(hashmap_compare(a, b, vec!["hoi2".to_string()], true));
    }
}
