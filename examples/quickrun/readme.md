# Using quickrun

## The main idea

The `quickrun` mode is a way to quickly run an FL experiment
without needing to understand the details of `declearn`.

Once you have `declearn` installed, try running `declearn-quickrun`
to get run the MNIST example.

This mode requires two files and some data :

* A TOML file, to store your experiment configurations.
* A model file, to store your model wrapped in a `declearn` object
* A folder with your data, either already split between clients or
to be split usiong our utility function

To run your own expriments, these three elements needs to be either
organizzed in a specific way, or referenced in the TOML file. See details
in the last section.

## The TOML file

TOML is a minimal, human-readable configuration file format.
We use is to store all the configurations of an FL experiment.

This file is your main entry point to everythiong else.
If you write your own, the absolute path to this file should be given
as an argument :  `declearn-quickrun --config <path_to_toml_file>`

For a minimal example of what it looks like in `declearn`, see
`./config.toml`. You can use it as a template
to write your own.

The TOML is parsed to python as dictionnary with each `[header]`
as a key. If you are a first time user, you just need to
understand how to write dictionnaries and lists. This :

```
[key1]
sub_key = ["a","b"]
[key2]
sub_key = 1
```

will be parsed as :

```python
{"key1":{"sub_key":["a","b"]},"key2":{"sub_key":1}}
```

Note the = sign and the absence of quotes around keys.

For more details, see the full doc : <https://toml.io/en/>

For a more detailed templates, with all options used
See `./custom/config_custom.toml`

## The Model file

The model file should just contain the model you built for
your data, e.g. a `torch` model, wrapped in a declearn object.
See `./model.py` for an example.

The wrapped model should be named "MyModel" by default. If you use
any other name, you'll need to mention it in the TOML file, as done
in `./custom/config_custom.toml`

## The data

Your data, in a standard tabular format. This data can either require
splitting or be already split by client

Requires splitting:

* You have a single dataset and want to use provided utils to split it
* In which case you need to mention your data source in the TOML file,
as well as details on how to split your data. See
`./custom/config_custom.toml` for details.
* Note that our data splitting util currently has a limited scope,
only dealing with classification tasks, excluding multi-label. If your
use case falls outside of that, you can write custom splitting code
and refer to the next paragraph

Already split:

* If your data is already split between clients, you will need to
add details in the TOML file on where to find this data. See
`./custom/config_custom.toml` for details.

## Organizing your files

The quickrun mode expects a `config` path as an argument. This can be the path to :

* A folder, expected to be structured a certain way
* A TOML file, where the location of every other object is mentionned

In both cases, the default is to check the folder provided, or the TOML
parent folder, is structured as follows:

```
    folder/
    │    config.toml - the config file
    │    model.py - the model
    └─── data*/
        └─── client*/
        │      train_data.* - training data
        │      train_target.* - training labels
        │      valid_data.* - validation data
        │      valid_target.* - validation labels
        └─── client*/
        │    ...
```

Any changes to this structure should be referenced in the TOML file, as
shown in `./custom/config_custom.toml`.
