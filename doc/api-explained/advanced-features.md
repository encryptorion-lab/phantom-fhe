# Advanced Features

## Hybrid Key-Switching

## Leveled BFV

## Hoisting

PhantomFHE implements hoisting technique which is commonly used by applications. Hoisting can be used to accelerate inter-slots accumulation which originally takes $$\log N$$ rotation-addition combinations.

To use hoisting, users should first specify all Galois elements and generate the corresponding Galois keys used in hoisting.

{% tabs %}
{% tab title="C++" %}
{% code overflow="wrap" lineNumbers="true" %}
```cpp
// assume other parameters are set already

// define rotation steps for hoisting
std::vector<int> hoisting_steps = {1, 2, 3, 4, 5, 6, 7};

// set required Galois elements
params.set_galois_elts(phantom::get_elts_from_steps(hoisting_steps, n));

// generate context and other keys

// generate Galois keys
PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

// encode and encrypt

// call hoisting
auto ct_out = phantom.hoisting(context, ct_in, glk, hoisting_steps)
```
{% endcode %}
{% endtab %}

{% tab title="Python" %}
{% code overflow="wrap" lineNumbers="true" %}
```python
# assume other parameters are set already

# define rotation steps for hoisting
hoisting_steps = [1, 2, 3, 4, 5, 6, 7]

# set required Galois elements
params.set_galois_elts(phantom.get_elts_from_steps(hoisting_steps, n))

# generate context and other keys

# generate Galois keys
glk = sk.create_galois_keys(context)

# encode and encrypt

# call hoisting
ct_out = phantom.hoisting(context, ct_in, glk, hoisting_steps)
```
{% endcode %}
{% endtab %}
{% endtabs %}
