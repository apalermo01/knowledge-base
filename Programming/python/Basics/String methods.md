

`split` The split function acts like a pair of scissors that cuts a string on a specific token, returning a list of individual pieces

example:

```python
                     #snip#            #snip#
some_string = "Hello friend you are my friend"

splitted = some_string.split("friend")
```

the value of `splitted` is `["Hello ", " you are my ", ""]`

`join` The join function acts like a glue that joins a list of strings together using some token. If we take the result from the last example:

```python

some_list = ["Hello ", " you are my ", ""]

joined = "enemy".join(some_list)
```

The join function will stitch the 3 strings together, inserting the word "enemy" between every element in the list: `Hello enemy you are my enemy`