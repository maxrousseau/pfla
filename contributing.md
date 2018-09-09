# How to contribute


## Reporting Issues

To report an issues or bug please use the [github issues page](https://github.com/maxrousseau/pfla/issues). 

Before opening a new issue make sure that it has not already been mentionned in
a previous thread.

When reporting on a problem make sure that your initial comment includes the
following:
- Short title, and thorough description in the body of the initial comment
- Version of: Python, R, OpenCV 
- Operating system
- Terminal input and output
- List of contents of the directories being fed as input
- Output of the test ([see test section](https://github.com/maxrousseau/pfla))


## Submitting changes

Please send a [GitHub Pull Request to pfla](https://github.com/maxrousseau/pfla/pull/new/master) with a clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). 
Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."

## Coding conventions

Follow the [Google python style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) and
[Google R style guide](https://google.github.io/styleguide/Rguide.xml) when writing code for this package.

Additional notes:
- Indent your code with tabs
- Use CSV file for data storage
- Include tests and documentation for your newly implemented features
- Use TravisCI for building
- Use Sphinx for documentation
