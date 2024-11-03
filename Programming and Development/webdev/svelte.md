# A few important things to remember for svelte.

This isn't a comprehensive write-up (yet), just a few things that I want to make
sure I keep in mind when I revist this later.


## Reactivity

using the syntax 
```svelte
<script>
    $: some statements
</script>
```

The statements in front of `$` (or grouped by `$`) will execute whenever a value
contained inside it is updated. For example

```svelte
<script>
    let count = 0;
    $: doubled = count * 2;
    function increment() {
        count += 1
    }
</script>

<button on:click={increment}>
	Clicked {count}
	{count === 1 ? 'time' : 'times'}
</button>
```

One important caveat to keep in mind is that this reactivity won't work with
inplace methods (e.g. push methods on arrays). In the case of arrays, you can
use this pattern to re-assign the array and trigger the reactivity

```
numbers = [... numbers, newElement]
```

"A simple rule of thumb: the name of the updated variable must appear on the left hand side of the assignment."

