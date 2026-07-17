# Annotation Schema Overview

Each annotated span may receive labels from three independent dimensions:

| Axis              | Purpose                        |
| ----------------- | ------------------------------ |
| **reason_form**   | Structure of the factual claim |
| **reason_frame**  | Epistemic / discourse framing  |
| **reason_domain** | Topical domain                 |

---

# 1. reason_form (Claim Structure)

| Tag                      | Definition                                                | Include when                                      | Exclude when                     |
| ------------------------ | --------------------------------------------------------- | ------------------------------------------------- | -------------------------------- |
| quantitative precise     | Exact numeric value or measurement                        | “inflation is 4.2%”, “5,000 people died”          | vague estimates                  |
| quantitative approximate | Non-exact numeric magnitude                               | “about 5,000”, “nearly half”, “tens of thousands” | exact numbers                    |
| quantitative vague       | Ambiguous or vague non-numeric quantifiers                | "a big percentage", "many people"                 | numeric quantities               |
| non-quantitative fact    | Discrete factual assertion without quantitative structure | “the law passed”, “the president resigned”        | numeric or comparative claims    |
| forecast                 | Statement about future state of affairs                   | “will increase”, “expected to rise”               | past or present facts            |
| comparison               | Explicit relational comparison between entities           | “higher than”, “less than”, “as large as”         | standalone facts                 |
| temporal reference       | Anchoring in time without describing change               | “in 2020”, “last year”, “during WW2”              | trend/change over time           |
| trend/change             | Describes evolution across time                           | “rose over time”, “declined since 2010”           | single-time-point facts          |
| ranking                  | Ordered position in a set                                 | “first”, “top 3”, “ranked highest”                | non-ordered comparisons          |
| causation                | Explicit cause-effect relation                            | “X caused Y”, “due to X, Y happened”              | correlation or temporal sequence |

---

# 2. reason_frame (Epistemic / Discourse Framing)

| Tag                    | Definition                                                        | Include when                                                            | Exclude when                                                        |
| ---------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------- |
| speech attribution     | Source or reporting entity is cited                               | “the President said”, “the WHO said, quote”                             | when no source is referenced                                        |
| behavioral attribution | Real-world action or decision performed by actor                  | "Walmart fired employees", "you started the war"                        | when the actor is a passive subject or not the director of an event |
| epistemic              | Claim about existence/quality of evidence or knowledge            | “no evidence exists”, “studies were debunked”                           | direct factual claims without meta-evidence content                 |
| stance                 | Agent’s political/policy position                                 | “the senator voted against the bill”, “the president supported the war” | general opinions without relevant agent action                      |
| anecdote               | Personal or isolated experiential claim                           | “I saw”, “in my experience”, “a man told me”                            | aggregated/statistical claims                                       |
| conspiracy             | Claims involving covert coordination or hidden systemic deception | “secret plan”, “cover-up”, “hidden agenda by elites”                    | standard political disagreement or criticism                        |

---

# 3. reason_domain (Topical Domain)

| Tag                     | Definition                                                                   |
| ----------------------- | ---------------------------------------------------------------------------- |
| economy                 | Macroeconomics, markets, labor, inflation, fiscal policy                     |
| housing                 | Housing availability and costs, rents                                        |
| health/medicine         | Medical conditions, healthcare systems, epidemiology                         |
| science/research        | Scientific findings, academic studies, experiments                           |
| education               | Schools, learning systems, policy, pedagogy                                  |
| energy                  | Energy production, transition, utilities                                     |
| environment             | Climate, ecosystems, environmental policy                                    |
| domestic politics       | Internal political systems, elections, governance                            |
| international relations | Diplomatic relations, treaties, sanctions, inter-state policy                |
| defense/military        | Military capability, defense policy, armed forces (non-active conflict)      |
| war/conflict            | Armed conflict, invasions, battles, warfare events                           |
| terrorism               | Non-state asymmetric violence targeting civilians or symbolic infrastructure |
| ethnicity/race          | Ethnic/racial group-related social or statistical claims                     |
| immigration             | Migration flows, asylum, border policy                                       |
| religion                | Religious groups, beliefs, institutions                                      |
| crime                   | Criminal activity, policing, illegal acts                                    |
| justice                 | Courts, legal systems, sentencing, law enforcement institutions              |
| narcotics               | Drugs, trafficking, substance policy                                         |
| technology              | Computing, AI, engineering, digital systems                                  |
| LGBTQ+                  | Sexual orientation and gender identity topics                                |
| gender equality         | Gender-related social/economic equality                                      |
| abortion                | Abortion policy and reproductive rights                                      |
| euthanasia              | Assisted dying, end-of-life policy                                           |
| personal_life           | Individual/private life events not in public domain                          |
| other                   | None of the above                                                            |

---

# Key Design Principles (Important for Annotators)

### 1. Orthogonality rule

- `reason_form` = **how the claim is structured**
- `reason_frame` = **how it is epistemically or rhetorically framed**
- `reason_domain` = **what it is about**

Do not use domain to encode form or frame.

---

### 2. Minimal multi-labeling

Only assign multiple tags when each clearly contributes independent information:

- Example:
  “According to WHO, about 5,000 people died in 2020”
  - form: quantitative approximate
  - frame: attribution
  - domain: health/medicine

---

### 3. Stance specificity

`stance` only applies when:

- an **identifiable agent** takes a position or action toward an issue

Not general sentiment or opinion.
