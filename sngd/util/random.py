import jax

# syntactic sugar for calling a random function and updating the rng inline
# example usage: y, rng = rngcall(f, rng, x)
def rngcall(f, rng, *args, **kwargs):
    rng1, rng2 = jax.random.split(rng)
    return f(rng2, *args, **kwargs), rng1
