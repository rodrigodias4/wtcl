# Annotation Guidelines for Check-Worthy Claim Detection

## Objective

The goal of this annotation phase is to identify **check-worthy claim spans** within speaker turns.

A **speaker turn** is the unit of annotation. Each turn may contain **zero, one, or multiple check-worthy claims**.

A **check-worthy claim** is a factual assertion that can be verified against external evidence and is relevant enough to merit fact-checking.

## What to Annotate

Annotate the **minimal contiguous span** that expresses the claim.

The span should:

- Contain enough information to preserve the factual proposition.
- Exclude unnecessary surrounding context.
- Be self-contained whenever possible.

### Example

**Speaker turn:**

> According to the latest census, unemployment dropped by 3% last year in Portugal.

**Annotate:**

> unemployment dropped by 3% last year in Portugal

Do **not** annotate:

> According to the latest census, unemployment dropped by 3% last year in Portugal.

Reason: "According to the latest census" is attribution, not part of the factual claim.

## Definition of a Check-Worthy Claim

A span is check-worthy if it satisfies all:

### Factual Verifiability

The claim can be checked against evidence.

✔ Examples:

- Inflation reached 8% in 2023.
- The law was passed in March.
- Lisbon has over 500,000 residents.

✘ Not check-worthy:

- I think taxes are too high.
- That policy is terrible.

### Specificity

The claim must be sufficiently concrete.

✔ Specific:

- Crime increased by 12% in 2024.

✘ Too vague:

- Crime is getting worse.

### Public Relevance

The claim should concern matters of public, political, social, economic, or historical importance.

Higher priority:

- Statistics
- Policy claims
- Historical claims
- Institutional actions
- Scientific/medical claims

Lower priority:

- Personal anecdotes
- Trivial biographical details

## Span Selection Rules

## A. Minimal Span Principle

Always annotate the smallest span that preserves the factual proposition.

✔ Good:

> increased by 15% in 2022

✘ Too broad:

> I believe it increased by 15% in 2022 because...

## B. Include Necessary Modifiers

Keep temporal, quantitative, and locative modifiers if they affect truth conditions.

✔ Include:

- in 2021
- by 30%
- in Europe

Example:

> exports grew by 10% in 2022

## C. Exclude Non-Claim Material

Do not include:

- Hedging ("I think", "probably")
- Attribution ("experts say", "according to X")
- Discourse fillers ("you know", "basically")
- Pure rhetorical framing

Example:

Turn:

> I think, according to WHO, malaria cases doubled.

Annotate:

> malaria cases doubled

## Special Cases

## Compound Claims

If multiple independent factual claims appear in one turn, annotate each separately.

Example:

> Inflation rose by 5% and unemployment fell by 2%.

Annotate:

- inflation rose by 5%
- unemployment fell by 2%

## Reported Speech Attribution

When the assertion concerns a speech act (e.g., someone said, claimed, denied, promised, admitted, or warned something), the reporting predicate and its complement are included in the span.
Discourse markers, conversational fillers, and rhetorical framing are excluded.
This only applies to speech attribution reported with respect to a domain-relevant and prominent entity whose utterance is of public interest.
If the reported actor corresponds to a generalized, non-specific or vague attribution (e.g. "the polls say ..."), this rule does not apply, and only the embedded claim should be evaluated.
Exclude evidential expressions that merely identify an information source (e.g., according to, based on, as reported by, per), unless the existence of the statement itself is the primary claim.

## Predictions

Future claims are check-worthy only if concrete and falsifiable.

✔:

> GDP will grow 4% next year.

✘:

> Things will get better.

## Comparisons

Annotate if measurable or factual.

✔:

> Portugal has higher inflation than Spain.

✘:

> Portugal is doing better than Spain.

## Discourse markers

Discourse markers and parentheticals that do not alter propositional content should be included in spans **only if** excluding them would fragment a single check-worthy claim. They are not independently annotated nor treated as span boundaries.

> “I took in billions and billions of dollars, as you know, from China”

## What NOT to Annotate

Do not annotate:

- Opinions
- Value judgments
- Preferences
- Questions
- Commands
- Hypotheticals
- Pure speculation
- Metaphors/jokes

Examples:

✘ This is the worst government ever.
✘ What if inflation rises again?
✘ We should lower taxes.

## Boundary Resolution Heuristics

When uncertain:

**Prefer shorter spans over longer spans.**

Ask:

Can this be independently fact-checked?
Does removing this word change the truth conditions?
Is this part necessary for the factual proposition?

If not necessary, exclude it.

## Consistency Rule

For repeated structural patterns, apply the same annotation logic consistently across the dataset.

Consistency is more important than edge-case perfection.

The goal of this phase is to maximize **inter-annotator agreement** and produce stable training data.

## Final Checklist

Before finalizing a span:

- Is it factual?
- Is it verifiable?
- Is it specific?
- Is it minimally bounded?
- Is it free of unnecessary framing?

If all yes → annotate.
If uncertain → lean conservative and annotate only the strongest factual core.
