This subdirectory contains a demo **spelling_correction.py** for trying out our context-sensitive spelling correction model with the best parameters from the experiments. This script can be run from the command line with a range of input parameters.

# Full documentation

Invoke a command without arguments to list available arguments and their default values. Use -h or --help to display more information per parameter, such as the
required input values.

```
$ python3 spelling_correction.py

The following arguments are mandatory:
	-input 			path to json file containing a list of tuples (misspelling, list of left context tokens, list of right context tokens)
	-output 		path to write corrections to
	-pathtofrequencies 	path to json file containing corpus frequencies for noisy channel ranking model
	-pathtomodel 		path to .bin file of trained fastText model

The following arguments are optional (default values between square brackets):
	-model 		whether to use the context-sensitive (1) or noisy channel (0) ranking model [1]
	-k 		number of top-ranked corrections to return [1]
	-language 	language of the input, English (1) or Dutch (0) [1]
	-backoff 	in case of insufficient context for context-sensitive ranking model, backoff to noisy channel model (1) or not (0) [1]
```










