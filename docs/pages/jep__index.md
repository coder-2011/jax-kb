- [](../index.html)
- [Developer notes](../contributor_guide.html)
- JAX Enhancement Proposals (JEPs)

[ ](https://github.com/jax-ml/jax "Source repository")

- [ .rst](../_sources/jep/index.rst "Download source file")
-  .pdf

# JAX Enhancement Proposals (JEPs)

## Contents

- [When you should use a JEP](#when-you-should-use-a-jep)
- [How to start a JEP](#how-to-start-a-jep)

# JAX Enhancement Proposals (JEPs)[\#](#jax-enhancement-proposals-jeps "Link to this heading")

Most changes can be discussed with simple issues/discussions and pull requests.

Some changes though are a bit larger in scope or require more discussion, and these should be implemented as JEP. This allows for writing longer documents that can be discussed in a pull request themselves.

The structure of JEPs is kept as lightweight as possible to start and might be extended later on.

## When you should use a JEP[\#](#when-you-should-use-a-jep "Link to this heading")

- When your change requires a design doc. We prefer collecting the designs as JEPs for better discoverability and further reference.

- When your change requires extensive discussion. It’s fine to have relatively short discussions on issues or pull requests, but when the discussion gets longer this becomes unpractical for later digestion. JEPs allow to update the main document with a summary of the discussion and these updates can be discussed themselves in the pull request adding the JEP.

## How to start a JEP[\#](#how-to-start-a-jep "Link to this heading")

First, create an issue with the [JEP label](https://github.com/jax-ml/jax/issues?q=label%3AJEP). All pull requests that relate to the JEP (i.e. adding the JEP itself as well as any implementing pull requests) should be linked to this issue.

Then create a pull request that adds a file named %d-{short-title}.md - with the number being the issue number.

- [263: JAX PRNG Design](263-prng.html)
- [2026: Custom JVP/VJP rules for JAX-transformable functions](2026-custom-derivatives.html)
- [4008: Custom VJP and \`nondiff_argnums\` update](4008-custom-vjp-update.html)
- [4410: Omnistaging](4410-omnistaging.html)
- [9263: Typed keys & pluggable RNGs](9263-typed-keys.html)
- [9407: Design of Type Promotion Semantics for JAX](9407-type-promotion.html)
- [9419: Jax and Jaxlib versioning](9419-jax-versioning.html)
- [10657: Sequencing side-effects in JAX](10657-sequencing-effects.html)
- [11830: \`jax.remat\` / \`jax.checkpoint\` new implementation](11830-new-remat-checkpoint.html)
- [12049: Type Annotation Roadmap for JAX](12049-type-annotations.html)
- [14273: \`shard_map\` (\`shmap\`) for simple per-device code](14273-shard-map.html)
- [15856: \`jax.extend\`, an extensions module](15856-jex.html)
- [17111: Efficient transposition of \`shard_map\` (and other maps)](17111-shmap-transpose.html)
- [18137: Scope of JAX NumPy & SciPy Wrappers](18137-numpy-scipy-scope.html)
- [25516: Effort-based versioning](25516-effver.html)
- [28661: Supporting the \`\_\_jax_array\_\_\` protocol](28661-jax-array-protocol.html)
- [28845: Stateful Randomness in JAX](28845-stateful-rng.html)

Several early JEPs were converted in hindsight from other documentation, issues, and pull requests, so they might not exactly reflect the process outlined above.

[](../autodidax2_part1.html "previous page")

previous

Autodidax2, part 1: JAX from scratch, again

[](263-prng.html "next page")

next

JAX PRNG Design

Contents

- [When you should use a JEP](#when-you-should-use-a-jep)
- [How to start a JEP](#how-to-start-a-jep)

By The JAX authors

© Copyright 2024, The JAX Authors.\
