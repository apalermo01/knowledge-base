Place a tensor on a specific devicef:

```python
with tf.device("GPU:0"):
    x = tf.random.uniform([1000, 1000])

```


# Data Loading

**Use layers when dealing with small datasets that use common preprocessing techniques that can fit into memory. Use tf.data when things are more complex**. 

- `layers.Normalization()` - class that normalizes numerical features
- `layers.StringLookup()` - maps strings to integer indices in a vocabulary
- `layers.CategoryEncoding` - convert indices to float 32 data

**To run normalization on all features**
```python
tf.keras.Sequential([
	normalize,
	...
])
```


**To run preprocessing feature-by-feature**: concatenate a list of preprocessed columns. When you build a model it will expect a dictionary of tensors in the form of `{feature_name: column_data (tensor)`

**Lazy execution**: preprocessing is carried out using lazy execution - you define what computations will take place before running the numbers through

### Datasets

- `tf.data.Datset.from_tensor_slices` - takes a dictionary of tensors as input data 
- **shuffle and batch the data**: `<dataset name>.shuffle(len(labels)).batch(batch_size)`

**build a dataset directly from a csv file**:
```python

dataset = tf.data.experimental.make_csv_dataset(
	<return value of tf.keras.util.get_file>,
	batch_size = ,
	label_name = ,
	num_epochs = 1, # MAKE SURE THIS IS A NUMBER - DEFAULT BEHAVIOR IS TO LOOP ENDLESSLY
	ignore_errors=True
)
```

- tensorflow can cache / take a snapshot of a csv file so there's less overhead.
- instead of a single file, you can pass a glob-like string as an argument for `file_pattern` that can parse multiple files at once

```python
fonts_ds = tf.data.experimental.make_csv_dataset(
    file_pattern = "fonts/*.csv",
    batch_size=10, num_epochs=1,
    num_parallel_reads=20,
    shuffle_buffer_size=1000
)
```
# References
- https://www.tensorflow.org/tutorials/load_data/csv