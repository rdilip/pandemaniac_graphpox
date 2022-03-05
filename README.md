# Pandemaniac Submission
This is the repository for graphpox's submission to Pandemaniac 2022 for CMS 144 at Caltech.

We submitted a new repo because the one we used for the project is really messy and we messed up some of the version history, but if you want to see the original we can share you on that as well. Most of our strategies are contained in [strategies.py](strategies.py). [cstrategies.pyx](cstrategies.pyx) is essentially the same as strategies.py, but is compiled to C code using [setup.py](setup.py) for speed purposes. Finally, [graph_io.py](graph_io.py) contains most of our input/output routines, and requires a folder graphs (not in the repo) that contains all the input json files.
