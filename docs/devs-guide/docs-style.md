# Docstrings style guide

The declearn docstrings are written, and more importantly parsed, based on the
[numpy](https://numpydoc.readthedocs.io/en/latest/format.html) style. Here are
some details and advice as to the formatting, both for coherence within the
codebase and for proper rendering (see the dedicated section above on
[building the docs](#building-the-documentation)).

## Public functions and methods

For each public function or method, you should provide with:

### Description

As per [PEP 257](https://peps.python.org/pep-0257/), a function or method's
docstring should start with a one-line short description, written in imperative
style, and optionally followed by a multi-line description. The latter may be
as long and detailed as required, and possibly comprise cross-references or
be structured using [lists](#lists-rendering).

### "Parameters" section

If the function or method expects one or more (non-self or cls) argument,
they should be documented as part of a "Parameters" section.

Example:
```
Parameters
----------
toto:
    Description of the "toto" parameter.
babar:
    Description of the "babar" parameter.
```

It is not mandatory to specify the types of the parameters or their
default values, as they are already specified as part of the signature,
and parsed from it by mkdocs. You may however do it, notably in long
docstrings, or for clarity purposes on complex signatures.

### "Returns" or "Yields" section

If the function or method returns one or more values, they should be documented
as part of a "Returns" section. For generators, use an equivalent "Yields"
section.

Example:
```
Returns
-------
result:
    Description of the "result" output.
```

Notes:

- Naming outputs, even when a single value is returned, is useful for
  clarity purposes. Please choose names that are short yet unequivocal.
- It is possible to specify the types of the returned values, but this
  will override the type-hints in the mkdocs-rendered docs and should
  therefore be used sparingly.

### "Raises" section

If some exceptions are raised or expected by the function, they should be
documented as part of a "Raises" section.

Example:
```
Raises
------
KeyError
    Description of the reasons why a KeyError would be raised.
TypeError
    Description of the reasons why a TypeError would be raised.
```

Use this section sparingly: document errors that are raised as part
of this function or method, and those raised by its backend code if
any (notably private functions or methods it calls). However, unless
you actually expect some to be raised, do not list all exceptions
that may be raised by other components (e.g. FileNotFoundError in
case an input string does not point to an actual file) if these are
not the responsibility of declearn and/or should be obvious to the
end-users.

### "Warns" section

If some warnigns are or may be emitted by the function, they should be
documented as part of a "Warns" section.

- Use the same format as for "Raises" sections.
- You do not need to explicitly document deprecation warnings if the
  rest of the docstring alread highlights deprecation information.
- Similarly, some functions may automatically raise a warning, e.g.
  because they trigger unsafe code: these warnings do not need to be
  documented - but the unsafety does, as part of the main description.

## Private functions and methods

In general, private functions and methods should be documented following the
same rules as the public ones. You may however sparingly relax the rules
above, e.g. for simple, self-explanatory functions that may be described with
a single line, or if a private method shares the signature of a public one,
at which rate you may simply refer developers to the latter. As per the
[Zen of Python](https://peps.python.org/pep-0020/), code readability and
clarity should prevail.

## Classes

The main docstring of a public class should detail the following:

### Description

The formatting rules of the description are the same as for functions. A
class's description may be very short, notably for children of base APIs
that are already expensive as to how things work in general.

For API-defining (abstract) base classes, an overview of the API should
be provided, optionally structured into subsections.

## "Attributes" section

An "Attributes" section should detail the public attributes of class instances.

Example:
```
Attributes
----------
toto:
    Description of the toto attribute of instances.
babar:
    Description of the babar attribute of instances.
```


Historically, these have not been used in declearn half as much as they
should. Efforts towards adding them are undergoing, and new code should
not neglect this section.

## Additional sections

Optionally, one or more sections may be used to detail methods.

Detailing methods is not mandatory and should preferably be done sparingly.
Typically, you may highlight and/or group methods that are going to be of
specific interest to end-users and/or developers, but do not need to list
each and every method - as this can typically be done automatically, both
by python's `help` function and mkdocs when rendering the docs.


For such sections, you need to add listing ticks for mkdocs to parse the
contents properly:
```
Key methods
-----------
- method_a:
    Short description of the method `method_a`.
- method_b:
    Short description of the method `method_b`.
```

A typical pattern used for API-defining abstract bases classes in declearn
is to use the following three sections:

- "Abstract": overview of the abstract class attributes and methods.
- "Overridable": overview of the key methods that children classes are
  expected to overload or override to adjust their behavior.
- "Inheritance": information related to types-registration mechanisms
  and over inheritance-related details.

## Constants and class attributes

For constant variables and class attributes, you may optionally write up a
docstring that will be rendered by mkdocs (but have no effect in python),
by writing up a three-double-quotes string on the line(s) that follow(s)
the declaration of the variable.

Example:
```
MY_PUBLIC_CONSTANT = 42.0
"""Description of what this constant is and where/how it is used."""
```

This feature should be used sparingly: currently, constants that have such
a docstring attached will be exposed as part of the mkdocs-rendered docs,
while those that do not will not be rendered. Hence it should be used for
things that need to be exposed to the docs-consulting users; typically,
abstract (or base-class-level) class attributes, or type-hinting aliases
that are commonly used in some public function or method signatures.

## Public modules' main docstring

The main docstring of public modules (typically, that of `__init__.py` files)
are a key component of the rendered documentation. They should detail, in an
orderly manner, all the public functions, classes and submodules they expose,
so as to provide with a clear overview of the module and its subcomponents.
It should also contain cross-references to these objects, so that the mkdocs-
rendered documentation turns their names into clickable links to the detailed
documentation of these objects.

Example:
```
"""Short description of the module.

Longer description of the module.

Category A
----------
- [Toto][declearn.module.Toto]:
    Short description of exported class Toto.
- [func][declearn.module.func]:
    Short description of exported function func.

Submodules
----------
- [foo][declearn.module.foo]:
    Short description of public submodule foo.
- [bar][declearn.module.bar]:
    Short description of public submodule bar.
"""
```

Note that the actual formatting of the description if open: sections may be
used, as in the example above, to group contents (and add them to the rendered
website page's table of contents), but you may also simply use un-named lists,
or include the names of (some of) the objects and/or submodules as part of a
sentence contextualizing what they are about.

The key principles are:
- Clarity and readability should prevail over all.
- Readability should be two-fold: for python `help` users and for the rendered
  website docs.
- Provide links to everything that is exported at this level, so as to act as
  an exhaustive summary of the module.
- Optionally provide links to things that come from submodules, if it will
  help end-users to make their way through the package.

## General information and advice

### Lists rendering

To have a markdown-like listing properly rendered by mkdocs, you need to add
a blank line above it:

```
My listing:

- item a
- item b
```

will be properly rendered, whereas

```
My listing:
- item a
- item b
```
will be unproperly rendered as `My listing: - item a - item b`

For nested lists, make sure to indent by 4 spaces (not 2) so that items are
rendered on the desired level.

### Example code blocks

Blocks of example code should be put between triple-backquote delimiters to
be properly rendered (and avoid messing the rest of the docstrings' parsing).

Example:
``````
Usage
-----
```
>>> my_command(...)
... expected output
```

Rest of the docstring.
``````

### Pseudo-code blocks

You may render some parts of the documentation as unformatted content delimited
by a grey-background block, by adding an empty line above _and_ identing that
content. This may typically be used to render the pseudo-code description of
algorithms:
```
"""Short description of the function.

This function implements the following algorithm:

    algorithm...
    algorithm...
    etc.

Rest of the description, arguments, etc.
"""
```

### LaTeX rendering

You may include some LaTeX formulas as part of docstrings, that will be
rendered using [mathjax](https://www.mathjax.org/) in the website docs,
as long as they are delimited with `$$` signs, and you double all backslashes
to avoid bad python parsing (e.g. `$$ f(x) = \\sqrt{x} $$`).

These should be used sparingly, to avoid making the raw docstrings unreadable
by developers and python `help()` users. Typically, a good practice can be to
explain the formula and/or provide with some pseudo-code, and then add the
LaTeX formulas as a dedicated section or block (e.g. as an annex), so that
they can be skipped when reading the raw docstring without missing other
information than the formulas themselves.
