# Annotation Guidelines for Check-Worthy Claim Detection

## Objective

The objective of this annotation task is to identify **check-worthy claim spans** within political debate transcripts.

The unit of annotation is the **speaker turn**. Each turn may contain zero, one, or multiple check-worthy claims.

A **check-worthy claim** is a factual assertion that is sufficiently specific, verifiable using external evidence, and relevant to public discourse. The goal is to identify the textual span that expresses the assertion itself, preserving its complete propositional meaning while excluding surrounding discourse that does not contribute to the claim.

## Core Principles

Every annotation should satisfy the following principles.

### Completeness

Annotate the **smallest complete contiguous span** that expresses the asserted proposition.

The span must contain all grammatical elements necessary to preserve the meaning of the claim. Do not remove subjects, auxiliary verbs, negation, reporting predicates (when applicable), or obligatory complements.

### Minimality

Exclude any material that does not contribute to the factual meaning of the assertion.

This includes conversational fillers, discourse markers, rhetorical framing, parenthetical remarks, evidential expressions, and other text that does not affect the truth conditions of the claim.

### Consistency

Identical linguistic constructions shall be annotated according to the same rule throughout the corpus.

Consistency is prioritized over isolated edge-case decisions. If two examples share the same syntactic structure, they should receive the same annotation unless a documented guideline explicitly distinguishes them.

## Definition of a Check-Worthy Claim

A claim should satisfy all of the following criteria.

### Verifiable

The assertion can be verified or falsified using external evidence.

Examples:

✔ Inflation reached 8%.

✔ The Supreme Court ruled in favor of the plaintiff.

✔ The unemployment rate fell by 3%.

Not check-worthy:

✘ I think inflation is terrible.

✘ This policy is a disaster.

### Specific

The assertion must contain enough information to permit meaningful verification.

Annotate:

> Crime increased by 12% last year.

Do not annotate:

> Crime is getting worse.

### Public Relevance

The assertion concerns matters of public interest, including politics, economics, public policy, law, institutions, science, health, security, international affairs, or other issues commonly subject to public fact-checking.

## Span Boundaries

### Minimal Complete Span

Always annotate the smallest contiguous span that preserves a complete factual assertion.

Prefer:

> **the economy recovered after the pandemic**

rather than

> economy recovered

or

> recovered after the pandemic

### Required Grammatical Material

Include grammatical elements that are necessary for the proposition.

These include:

- subjects
- auxiliary verbs
- negation
- reporting predicates
- required complements
- obligatory prepositional phrases

Examples:

✔ **the economy grew**

✔ **the Supreme Court ruled that...**

✔ **did not increase**

✔ **has never admitted**

Do not remove words solely to shorten the span if doing so produces an incomplete proposition.

### Necessary Modifiers

Include modifiers whenever they alter the factual meaning of the claim.

Examples include:

- dates
- quantities
- locations
- temporal references
- comparisons

Examples:

✔ unemployment increased **by 5%**

✔ exports doubled **in 2023**

✔ inflation was lower **than in 2020**

## Speech Attribution

When the assertion concerns a communicative act performed by a publicly relevant actor, include the reporting predicate within the annotated span.

Examples of reporting predicates include:

- said
- claimed
- stated
- promised
- denied
- admitted
- acknowledged
- warned
- announced
- testified

Examples:

Sentence:

> The former President said he would terminate the Constitution.

Annotate:

> **The former President said he would terminate the Constitution**

Sentence:

> Harris claimed the economy was improving.

Annotate:

> **Harris claimed the economy was improving**

The communicative act itself forms part of the factual assertion.

### Generalized Attribution

Generalized or vague attribution is not considered a speech-event claim.

Examples include:

- the polls say
- experts say
- people say
- critics argue
- many believe

These expressions function as general evidential framing rather than assertions about a specific communicative event.

Only the embedded factual assertion should be evaluated.

### Evidential Attribution

Exclude expressions whose sole purpose is to identify an information source.

Examples include:

- according to
- based on
- as reported by
- per

Sentence:

> According to the WHO, malaria cases doubled.

Annotate:

> **malaria cases doubled**

## Quotations

Quoted material should be treated according to the same rules as unquoted text.

When the quotation forms part of a speech-event claim, include both the reporting predicate and the quoted content.

Example:

> **The former President said "Stand back and stand by."**

Do not remove quoted material solely because it is not independently truth-conditional.

If parenthetical interruptions occur within a quotation (e.g., "I'm quoting"), exclude only the interruption while preserving a single contiguous span whenever possible.

## Relative Pronouns and Clauses

When a generic head noun (someone, person, individual, people, those who, etc.) merely introduces a relative clause, begin the span at the relative pronoun rather than the generic head.

> This is someone **who has...**

> the same individual **who defended...**

> that principle **which led to...**

## Subjects and Determiners

Include complete noun phrases functioning as the grammatical subject.

Examples:

✔ **the economy recovered**

✔ **the unemployment rate increased**

✔ **our military is stronger**

✔ **someone who has openly said...**

Do not remove articles ("the", "a", "an") or possessive determiners ("our", "their") when they belong to the subject noun phrase.

## Appositives

Include appositive phrases only when they contribute factual content.

Example:

> the Proud Boys, **a militia**

If the appositive merely identifies the entity and does not alter the factual proposition, it may be excluded.

If it introduces independently verifiable information, include it.

## Parenthetical Expressions

Exclude parenthetical material that does not alter the factual meaning.

Examples:

- you know
- as you know
- frankly
- I'm quoting
- believe me

Example:

> I took in billions and billions of dollars, **as you know**, from China.

Annotate:

> **I took in billions and billions of dollars from China**

## Multiple Claims

When two or more independent factual propositions occur within the same sentence, annotate each proposition separately.

Example:

> Inflation increased by 5% and unemployment fell by 2%.

Annotate:

- **inflation increased by 5%**
- **unemployment fell by 2%**

Do not split clauses that together express a single proposition.

## Predictions

Future-oriented claims are annotated only when sufficiently specific and objectively verifiable.

Annotate:

> GDP will grow by 4% next year.

Do not annotate:

> Things will get better.

## Comparisons

Comparative statements should be annotated only when objectively verifiable.

Annotate:

> Portugal has higher inflation than Spain.

Do not annotate:

> Portugal is doing better than Spain.

## Non-Claims

Do not annotate:

- opinions
- value judgments
- preferences
- rhetorical questions
- commands
- requests
- recommendations
- wishes
- hypotheticals
- jokes
- metaphors

unless they contain an embedded factual assertion satisfying the annotation criteria.

## Punctuation

Punctuation (e.g., ., !, ?) should not be included in annotated spans unless it forms part of the lexical representation of a token (e.g., decimal points, abbreviations, contractions) or the claim spans multiple sentences. Trailing and leading punctuation is never annotated.

## Boundary Resolution

When uncertainty remains, apply the following questions in order.

1. Does the span express a complete factual assertion?
2. Would removing any included token alter the factual meaning?
3. Does every included token contribute to the asserted proposition?
4. Would the same construction be annotated identically elsewhere in the corpus?

If the answer to the fourth question is **no**, revise the annotation to preserve corpus-wide consistency.

## Final Consistency Check

Before finalizing an annotation, verify that:

- The span expresses a factual assertion.
- The assertion is specific and verifiable.
- The span is contiguous.
- The span is grammatically complete.
- No unnecessary discourse material is included.
- The annotation follows the same rules applied to structurally similar examples elsewhere in the corpus.

When uncertainty remains between two valid annotations, prefer the alternative that is **most consistent with previously annotated examples**. Corpus-wide consistency is the primary objective of the annotation process.
