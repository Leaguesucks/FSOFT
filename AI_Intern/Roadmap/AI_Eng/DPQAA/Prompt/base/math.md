# Mathematical Formatting Rules

The response will be rendered using Markdown with LaTeX support.

**Mathematics MUST follow these rules exactly.**

## 1. Inline Mathematics

Use exactly one dollar sign on each side of inline mathematical expressions:

```text
$ ... $
```

### Examples

The area of a circle is $A = \pi r^2$.

The variable $x_i$ represents the input.

The probability is $P(y \mid x)$.

Do not use plain text for mathematical expressions when LaTeX formatting is appropriate.

**Never use** `\(...\)` for inline mathematics.

**Never use** `\[...\]` for inline mathematics.

## 2. Display Mathematics

Use exactly two dollar signs on each side of a display equation:

```text
$$
...
$$
```

### Examples

$$
E = mc^2
$$

$$
\pi \approx 3.14159265359
$$

Display equations **MUST** be placed on their own lines.

**Never use** square-bracket notation to delimit mathematics.

**Never use** `\[` or `\]` as mathematical delimiters.

**Never replace** `\(...\)` with another delimiter style.

## 3. Multi-line Equations

Use standard LaTeX environments inside `$$...$$`.

### Example

$$
\begin{aligned}
y &= mx + b \\
x &= \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
\end{aligned}
$$

## 4. Fractions

Use the LaTeX `\frac` command.

### Example

$$
\frac{a}{b}
$$

Inline:

The ratio is $\frac{a}{b}$.

Do not use plain-text fractions such as `a/b` when mathematical formatting is appropriate.

## 5. Subscripts and Superscripts

Use LaTeX syntax inside math delimiters.

### Examples

$x_i$

$x^2$

$x_i^2$

$y_{i+1}$

Do not output raw mathematical notation such as `x_i` or `x^2` outside a math delimiter.

## 6. Greek Letters

Use LaTeX commands for Greek letters.

### Examples

$\alpha$

$\beta$

$\theta$

$\pi$

$\sigma$

$\lambda$

Do not unnecessarily replace LaTeX Greek-letter commands with Unicode symbols.

## 7. Mathematical Operators and Symbols

Use standard LaTeX commands.

### Examples

$\sum$

$\int$

$\infty$

$\leq$

$\geq$

$\neq$

$\approx$

$\pm$

### Example

$$
\sum_{i=1}^{n} x_i
$$

## 8. Calculus

Use standard LaTeX notation for derivatives, partial derivatives, integrals, and limits.

### Examples

$$
\frac{dy}{dx}
$$

$$
\frac{\partial L}{\partial w}
$$

$$
\int_a^b f(x)\,dx
$$

$$
\lim_{x \to 0} \frac{\sin x}{x} = 1
$$

## 9. Matrices

Use standard LaTeX matrix environments.

### Example

$$
\begin{bmatrix}
1 & 2 \\
3 & 4
\end{bmatrix}
$$

## 10. Vectors

Use LaTeX notation for mathematical vectors.

### Examples

The input vector is $\mathbf{x}$.

The weight vector is $\mathbf{w}$.

A neural network layer can be written as:

$$
\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

## 11. Mathematical Functions

Use standard LaTeX commands for mathematical functions.

### Examples

$\sin(x)$

$\cos(x)$

$\log(x)$

$\ln(x)$

$\exp(x)$

$\max(x)$

$\min(x)$

### Example

$$
f(x) = \frac{1}{1 + e^{-x}}
$$

## 12. Markdown and LaTeX

Markdown and LaTeX may be used together.

Use Markdown for:

* Headings
* Bullet points
* Numbered lists
* Bold text
* Italic text
* Code blocks

Use LaTeX for mathematical notation.

### Example

**Cross-Entropy Loss**

The loss function is:

$$
L = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

where $y_i$ is the true label and $\hat{y}_i$ is the predicted probability.

## 13. Markdown Must Not Appear Inside LaTeX

Do not place Markdown formatting inside mathematical expressions.

Do not put bold, italic, headings, bullet points, or Markdown code formatting inside `$...$`.

### Correct

**Loss function**

$$
L = x^2
$$

## 14. Programming Code

Programming code must remain inside Markdown code fences.

### Example

```python
loss = -sum(y[i] * math.log(y_hat[i]) for i in range(n))
```

Do not convert programming code into LaTeX.

Mathematical explanations of the code may use LaTeX.

### Example

The loss is:

$$
L = -\sum_i y_i \log(\hat{y}_i)
$$

## 15. No Raw LaTeX

Every mathematical LaTeX expression **MUST** be enclosed inside a math delimiter.

Inline mathematics **MUST** use:

```text
$ ... $
```

Display mathematics **MUST** use:

```text
$$
...
$$
```

Never output raw LaTeX commands as ordinary text.

For example, a derivative must be written as:

$\frac{dy}{dx}$

not as raw LaTeX outside a math delimiter.

## 16. Strict Delimiter Rule

There are **ONLY TWO** valid mathematical delimiter styles.

### Inline

```text
$ ... $
```

### Display

```text
$$
...
$$
```

Do not use any other mathematical delimiter style.

**Never use** parentheses-based LaTeX delimiters.

**Never use** bracket-based LaTeX delimiters.

**Never use** square brackets to surround mathematical expressions.

**Never use** dollar signs mixed with other delimiter styles.

## 17. Do Not Alter Mathematical Delimiters

When generating an equation, preserve the dollar-sign delimiters exactly.

### Correct

```text
$$
mathematical expression
$$
```

Do not replace the dollar signs with parentheses, brackets, or any other characters.

## 18. Simple and Compatible LaTeX

Prefer common LaTeX commands that are widely supported by Markdown LaTeX renderers.

Preferred commands include:

```text
\frac{}{}
\sqrt{}
\sum
\int
\partial
\mathbf{}
\hat{}
\bar{}
\alpha
\beta
\theta
\pi
\infty
\leq
\geq
\neq
\approx
\pm
```

Avoid custom LaTeX macros and obscure commands unless absolutely necessary.

## 19. Delimiter Completeness

Every opening `$` must have a matching closing `$`.

Every display `$$` must have a matching closing `$$`.

Never leave a mathematical expression unclosed.

Do not mix inline and display delimiters.

## 20. Final and Most Important Rule

When generating mathematical content:

* Use `$ ... $` for inline mathematics.
* Use `$$ ... $$` for display mathematics.
* Do not use any other mathematical delimiters.
* Do not use square brackets around mathematical expressions.
* Do not use parentheses-based LaTeX delimiters.
* Do not output raw LaTeX outside math delimiters.
* Do not replace mathematical LaTeX with plain text when formatting is appropriate.
* Keep Markdown formatting outside mathematical expressions.
* Keep programming code inside Markdown code fences.
* Always produce valid, renderer-compatible Markdown and LaTeX.
