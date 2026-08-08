# Input data

`open_trivia_shuffled.csv` is copied from the Hugging Face dataset
`lasrprobegen/sycophancy-activations`, path `multichoice/open_trivia_shuffled.csv`.

The driver repeats the original fixed pandas shuffle (`random_state=42`) and then uses the first
80% of rows for `train` and the remaining 20% for `test`.
