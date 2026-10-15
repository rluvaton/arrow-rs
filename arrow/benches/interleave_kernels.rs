// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#[macro_use]
extern crate criterion;

use criterion::Criterion;
use std::ops::Range;

use rand::distr::Alphanumeric;
use rand::prelude::StdRng;
use rand::{RngExt, SeedableRng};

use arrow::datatypes::*;
use arrow::util::test_util::seedable_rng;
use arrow::{array::*, util::bench_util::*};
use arrow_buffer::ScalarBuffer;
use arrow_select::interleave::interleave;
use std::hint;
use std::sync::Arc;

fn do_bench(
    c: &mut Criterion,
    prefix: &str,
    len: usize,
    base: &dyn Array,
    slices: &[Range<usize>],
) {
    let arrays: Vec<_> = slices
        .iter()
        .map(|r| base.slice(r.start, r.end - r.start))
        .collect();
    let values: Vec<_> = arrays.iter().map(|x| x.as_ref()).collect();
    bench_values(
        c,
        &format!("interleave {prefix} {len} {slices:?}"),
        len,
        &values,
    );
}

fn bench_values(c: &mut Criterion, name: &str, len: usize, values: &[&dyn Array]) {
    let mut rng = seedable_rng();
    let indices: Vec<_> = (0..len)
        .map(|_| {
            let array_idx = rng.random_range(0..values.len());
            let value_idx = rng.random_range(0..values[array_idx].len());
            (array_idx, value_idx)
        })
        .collect();

    c.bench_function(name, |b| {
        b.iter(|| hint::black_box(interleave(values, &indices).unwrap()))
    });
}

fn add_benchmark(c: &mut Criterion) {
    let i32 = create_primitive_array::<Int32Type>(1024, 0.);
    let i32_opt = create_primitive_array::<Int32Type>(1024, 0.5);
    let boolean = create_boolean_array(1024, 0.0, 0.5);
    let boolean_opt = create_boolean_array(1024, 0.1, 0.5);
    let string = create_string_array_with_len::<i32>(1024, 0., 20);
    let string_opt = create_string_array_with_len::<i32>(1024, 0.5, 20);
    let values = create_string_array_with_len::<i32>(10, 0.0, 20);
    let dict = create_dict_from_values::<Int32Type>(1024, 0.0, &values);

    let struct_i32_no_nulls_i32_no_nulls = StructArray::new(
        Fields::from(vec![
            Field::new("a", Int32Type::DATA_TYPE, false),
            Field::new("b", Int32Type::DATA_TYPE, false),
        ]),
        vec![
            Arc::new(create_primitive_array::<Int32Type>(1024, 0.)),
            Arc::new(create_primitive_array::<Int32Type>(1024, 0.)),
        ],
        None,
    );

    let struct_string_no_nulls_string_no_nulls = StructArray::new(
        Fields::from(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(create_string_array_with_len::<i32>(1024, 0., 20)),
            Arc::new(create_string_array_with_len::<i32>(1024, 0., 20)),
        ],
        None,
    );

    let struct_i32_no_nulls_string_no_nulls = StructArray::new(
        Fields::from(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Utf8, false),
        ]),
        vec![
            Arc::new(create_primitive_array::<Int32Type>(1024, 0.)),
            Arc::new(create_string_array_with_len::<i32>(1024, 0., 20)),
        ],
        None,
    );

    let values = create_string_array_with_len::<i32>(1024, 0.0, 20);
    let sparse_dict = create_sparse_dict_from_values::<Int32Type>(1024, 0.0, &values, 10..20);

    let string_view = create_string_view_array(1024, 0.0);

    // use 8192 as a standard list size for better coverage
    let list_i64 = create_primitive_list_array_with_seed::<i32, Int64Type>(8192, 0.1, 0.1, 20, 42);
    let list_i64_no_nulls =
        create_primitive_list_array_with_seed::<i32, Int64Type>(8192, 0.0, 0.0, 20, 42);

    let list_view_i64: ListViewArray =
        create_primitive_list_array_with_seed::<i32, Int64Type>(8192, 0.1, 0.1, 20, 42).into();
    let list_view_i64_no_nulls: ListViewArray =
        create_primitive_list_array_with_seed::<i32, Int64Type>(8192, 0.0, 0.0, 20, 42).into();

    // ListView with overlapping offset/size ranges: 100 unique element ranges of
    // 20 elements each, with 80 rows sharing each range (8000 rows, 2000 backing elements).
    let list_view_overlapping = {
        let num_unique = 100;
        let rows_per_unique = 80;
        let elems_per_row = 20;
        let total_rows = num_unique * rows_per_unique;
        let values = Arc::new(Int64Array::from_iter_values(
            0..((num_unique * elems_per_row) as i64),
        ));
        let offsets: Vec<i32> = (0..total_rows)
            .map(|i| ((i / rows_per_unique) * elems_per_row) as i32)
            .collect();
        let sizes = vec![elems_per_row as i32; total_rows];
        ListViewArray::new(
            Arc::new(Field::new_list_field(DataType::Int64, false)),
            ScalarBuffer::from(offsets),
            ScalarBuffer::from(sizes),
            values,
            None,
        )
    };

    let fsl_i64 = create_primitive_fixed_size_list_array::<Int64Type>(8192, 0.0, 0.0, 5);
    let fsl_i64_nulls = create_primitive_fixed_size_list_array::<Int64Type>(8192, 0.1, 0.1, 5);

    let map_str_i64 = create_string_map_array::<Int64Type>(8192, 0.0, 5, 8);
    let map_str_i64_nulls = create_string_map_array::<Int64Type>(8192, 0.1, 5, 8);

    let ree_run_ends = Int32Array::from_iter_values((1..=64).map(|i| i * 16));
    let ree_i64_values = create_primitive_array::<Int64Type>(64, 0.0);
    let ree_i64 = RunArray::<Int32Type>::try_new(&ree_run_ends, &ree_i64_values).unwrap();

    let dict_str_values = create_string_array_with_len::<i32>(20, 0.0, 12);
    let dict_for_ree = create_dict_from_values::<UInt32Type>(64, 0.0, &dict_str_values);
    let ree_dict = RunArray::<Int32Type>::try_new(&ree_run_ends, &dict_for_ree).unwrap();

    let cases: &[(&str, &dyn Array)] = &[
        ("bool(0.0)", &boolean),
        ("bool(0.1)", &boolean_opt),
        ("i32(0.0)", &i32),
        ("i32(0.5)", &i32_opt),
        ("str(20, 0.0)", &string),
        ("str(20, 0.5)", &string_opt),
        ("dict(20, 0.0)", &dict),
        ("dict_sparse(20, 0.0)", &sparse_dict),
        ("str_view(0.0)", &string_view),
        (
            "struct(i32(0.0), i32(0.0)",
            &struct_i32_no_nulls_i32_no_nulls,
        ),
        (
            "struct(str(20, 0.0), str(20, 0.0))",
            &struct_string_no_nulls_string_no_nulls,
        ),
        (
            "struct(i32(0.0), str(20, 0.0)",
            &struct_i32_no_nulls_string_no_nulls,
        ),
        ("list<i64>(0.1,0.1,20)", &list_i64),
        ("list<i64>(0.0,0.0,20)", &list_i64_no_nulls),
        ("list_view<i64>(0.1,0.1,20)", &list_view_i64),
        ("list_view<i64>(0.0,0.0,20)", &list_view_i64_no_nulls),
        ("list_view_overlapping<i64>(80x,20)", &list_view_overlapping),
        ("fixed_size_list<i64,5>(0.0,0.0)", &fsl_i64),
        ("fixed_size_list<i64,5>(0.1,0.1)", &fsl_i64_nulls),
        ("map<utf8,i64>(0.0,5,8)", &map_str_i64),
        ("map<utf8,i64>(0.1,5,8)", &map_str_i64_nulls),
        ("ree_i32<i64>(64 runs)", &ree_i64),
        ("ree_i32<dict<u32,utf8>>(64 runs)", &ree_dict),
    ];

    for (prefix, base) in cases {
        let slices: &[(usize, &[_])] = &[
            (100, &[0..100, 100..230, 450..1000]),
            (400, &[0..100, 100..230, 450..1000]),
            (1024, &[0..100, 100..230, 450..1000]),
            (1024, &[0..100, 100..230, 450..1000, 0..1000]),
        ];

        for (len, slice) in slices {
            do_bench(c, prefix, *len, *base, slice);
        }
    }

    for len in [100, 1024, 2048] {
        bench_values(
            c,
            &format!("interleave dict_distinct {len}"),
            len,
            &[&dict, &sparse_dict],
        );
    }

    bench_interleave_strings(c);
}

/// How the `(array, row)` indices passed to `interleave` are generated
#[derive(Clone, Copy)]
enum IndexProfile {
    /// Random array and random row for every output row
    Random,
    /// Random array for every output row, with row indices increasing per array
    /// (e.g. merging sorted streams)
    RandomArrayIncreasingRows,
    /// Runs of consecutive rows from a random array, starting at a random row
    Runs(usize),
    /// Random rows all taken from the first array (like `take`)
    SingleArray,
    /// Only a few distinct `(array, row)` pairs, each repeated many times
    /// (e.g. a join with skewed keys)
    FewDistinct(usize),
}

impl IndexProfile {
    fn name(&self) -> String {
        match self {
            Self::Random => "random".to_string(),
            Self::RandomArrayIncreasingRows => "random_array_increasing_rows".to_string(),
            Self::Runs(len) => format!("runs({len})"),
            Self::SingleArray => "single_array".to_string(),
            Self::FewDistinct(n) => format!("few_distinct({n})"),
        }
    }

    fn generate(
        &self,
        rng: &mut impl RngExt,
        num_arrays: usize,
        array_len: usize,
        output_len: usize,
    ) -> Vec<(usize, usize)> {
        match *self {
            Self::Random => (0..output_len)
                .map(|_| {
                    (
                        rng.random_range(0..num_arrays),
                        rng.random_range(0..array_len),
                    )
                })
                .collect(),
            Self::RandomArrayIncreasingRows => {
                let array_indices: Vec<usize> = (0..output_len)
                    .map(|_| rng.random_range(0..num_arrays))
                    .collect();

                // For each array, pick sorted random row indices for the rows taken from it
                let mut rows_per_array: Vec<std::vec::IntoIter<usize>> = (0..num_arrays)
                    .map(|array_idx| {
                        let count = array_indices.iter().filter(|&&a| a == array_idx).count();
                        let mut rows: Vec<usize> =
                            (0..count).map(|_| rng.random_range(0..array_len)).collect();
                        rows.sort_unstable();
                        rows.into_iter()
                    })
                    .collect();

                array_indices
                    .into_iter()
                    .map(|array_idx| (array_idx, rows_per_array[array_idx].next().unwrap()))
                    .collect()
            }
            Self::Runs(run_len) => {
                let mut indices = Vec::with_capacity(output_len);
                while indices.len() < output_len {
                    let array_idx = rng.random_range(0..num_arrays);
                    let start = rng.random_range(0..=array_len - run_len);
                    let len = run_len.min(output_len - indices.len());
                    indices.extend((start..start + len).map(|row| (array_idx, row)));
                }
                indices
            }
            Self::SingleArray => (0..output_len)
                .map(|_| (0, rng.random_range(0..array_len)))
                .collect(),
            Self::FewDistinct(n) => {
                let distinct: Vec<(usize, usize)> = (0..n)
                    .map(|_| {
                        (
                            rng.random_range(0..num_arrays),
                            rng.random_range(0..array_len),
                        )
                    })
                    .collect();
                (0..output_len)
                    .map(|_| distinct[rng.random_range(0..n)])
                    .collect()
            }
        }
    }
}

/// Distribution of string lengths, as real world data is rarely uniformly distributed
#[derive(Clone, Copy)]
enum StrLen {
    /// Every string has the same length (e.g. UUIDs, hashes, fixed width codes)
    Fixed(usize),
    /// Log-normal lengths around `median`, most real world text columns
    /// (names, words, URLs) have a right skewed length distribution like this
    LogNormal { median: f64, sigma: f64 },
    /// Pick one of the distributions with the given weight
    Mixture(&'static [(f64, StrLen)]),
}

impl StrLen {
    fn sample(&self, rng: &mut impl RngExt) -> usize {
        match *self {
            Self::Fixed(len) => len,
            Self::LogNormal { median, sigma } => {
                // Box-Muller transform to get a standard normal sample
                let u1: f64 = 1.0 - rng.random::<f64>();
                let u2: f64 = rng.random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                ((median * (sigma * z).exp()).round() as usize).min(4096)
            }
            Self::Mixture(parts) => {
                let total: f64 = parts.iter().map(|(w, _)| w).sum();
                let mut pick = rng.random::<f64>() * total;
                for (weight, len) in parts {
                    if pick < *weight {
                        return len.sample(rng);
                    }
                    pick -= weight;
                }
                parts.last().unwrap().1.sample(rng)
            }
        }
    }
}

/// Creates a string array whose lengths follow `str_len`
fn create_string_array_with_str_len<O: OffsetSizeTrait>(
    len: usize,
    null_density: f32,
    str_len: StrLen,
    seed: u64,
) -> GenericStringArray<O> {
    let rng = &mut StdRng::seed_from_u64(seed);
    (0..len)
        .map(|_| {
            if rng.random::<f32>() < null_density {
                None
            } else {
                let len = str_len.sample(rng);
                Some(
                    rng.sample_iter(&Alphanumeric)
                        .take(len)
                        .map(char::from)
                        .collect::<String>(),
                )
            }
        })
        .collect()
}

/// Short words, codes or names: mostly 5-20 characters
const NAMES: StrLen = StrLen::LogNormal {
    median: 10.0,
    sigma: 0.5,
};

/// Interleave strings from many arrays with different string lengths, null
/// densities, offset sizes and index profiles
fn bench_interleave_strings(c: &mut Criterion) {
    const ARRAY_LEN: usize = 8192;
    const OUTPUT_LEN: usize = 8192;
    const NUM_ARRAYS: usize = 100;

    fn create_arrays<O: OffsetSizeTrait>(
        num_arrays: usize,
        null_density: f32,
        str_len: StrLen,
    ) -> Vec<ArrayRef> {
        (0..num_arrays)
            .map(|i| {
                Arc::new(create_string_array_with_str_len::<O>(
                    ARRAY_LEN,
                    null_density,
                    str_len,
                    i as u64,
                )) as ArrayRef
            })
            .collect()
    }

    fn bench(
        c: &mut Criterion,
        data_name: &str,
        arrays: &[ArrayRef],
        index_profiles: &[IndexProfile],
    ) {
        let values: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
        for profile in index_profiles {
            let mut rng = seedable_rng();
            let indices = profile.generate(&mut rng, values.len(), ARRAY_LEN, OUTPUT_LEN);
            c.bench_function(
                &format!(
                    "interleave {data_name} {OUTPUT_LEN} from {} arrays of {ARRAY_LEN} {}",
                    values.len(),
                    profile.name()
                ),
                |b| b.iter(|| hint::black_box(interleave(&values, &indices).unwrap())),
            );
        }
    }

    let all_profiles = [
        IndexProfile::Random,
        IndexProfile::RandomArrayIncreasingRows,
        IndexProfile::Runs(32),
        IndexProfile::SingleArray,
        IndexProfile::FewDistinct(16),
    ];

    let data: &[(&str, f32, StrLen)] = &[
        // Ids such as UUIDs, all of the same length
        ("str(uuid, 0.0)", 0.0, StrLen::Fixed(36)),
        ("str(names, 0.0)", 0.0, NAMES),
        ("str(names, 0.3)", 0.3, NAMES),
        // Optional field where missing values are empty strings rather than nulls
        (
            "str(names_or_empty, 0.0)",
            0.0,
            StrLen::Mixture(&[(0.3, StrLen::Fixed(0)), (0.7, NAMES)]),
        ),
        // URLs or file paths: mostly 30-120 characters
        (
            "str(urls, 0.0)",
            0.0,
            StrLen::LogNormal {
                median: 60.0,
                sigma: 0.4,
            },
        ),
        // Free text such as comments or descriptions: mostly short with
        // occasional long values
        (
            "str(long_tail, 0.0)",
            0.0,
            StrLen::Mixture(&[
                (
                    0.9,
                    StrLen::LogNormal {
                        median: 12.0,
                        sigma: 0.8,
                    },
                ),
                (
                    0.1,
                    StrLen::LogNormal {
                        median: 400.0,
                        sigma: 0.7,
                    },
                ),
            ]),
        ),
    ];
    for (name, null_density, str_len) in data {
        let arrays = create_arrays::<i32>(NUM_ARRAYS, *null_density, *str_len);
        bench(c, name, &arrays, &all_profiles);
    }

    let arrays = create_arrays::<i64>(NUM_ARRAYS, 0.0, NAMES);
    bench(
        c,
        "large_str(names, 0.0)",
        &arrays,
        &[IndexProfile::RandomArrayIncreasingRows],
    );

    // 100 arrays are already covered above
    for num_arrays in [1, 2, 4, 10, 1000] {
        let arrays = create_arrays::<i32>(num_arrays, 0.0, NAMES);
        bench(
            c,
            "str(names, 0.0)",
            &arrays,
            &[
                IndexProfile::Random,
                IndexProfile::RandomArrayIncreasingRows,
                IndexProfile::Runs(32),
                IndexProfile::FewDistinct(16),
            ],
        );
    }
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
