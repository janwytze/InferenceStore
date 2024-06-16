use std::collections::{BTreeMap, HashMap};
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
pub fn btreemap_compare<K, V>(
    map1: BTreeMap<K, V>,
    map2: BTreeMap<K, V>,
    keys_to_compare: Vec<K>,
    exclude_keys: bool,
) -> bool
where
    K: Eq + Hash + std::cmp::Ord,
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
