# Action boundaries and reference annotation

## Sub-goal and boundary

DocuPilot follows the event segmentation account of **Kurby & Zacks (2008)**:
people perceive an ongoing activity hierarchically. The task is divided into
**sub-goals**, and each sub-goal is carried out through a sequence of finer
actions, all of which serve it.

- In the task "prepare a report", "insert a chart", "format a heading" or
  "save the file under a new name" would be sub-goals.
- Clicking, typing, opening a menu or a dialog, scrolling or checking a
  result are finer actions *within* a sub-goal.
- Correcting a mistake, such as a typo, does not change the sub-goal.

An **action boundary** lies where the sub-goal changes: where the person stops
pursuing one sub-goal and begins the next. An action step is the segment
between two boundaries.

The segmentation looks for boundaries, not for labelled sections – like *Generic
Event Boundary Detection* (Shou et al. 2021). Predefined action labels could not
be defined meaningfully across applications.

## Where exactly the boundary lies

The reference boundary lies at the **first perceivable hint** of the new
sub-goal, as seen by an observer who perceives image, sound and input
simultaneously:

- a sentence announcing the step ("Now I'll insert …"),
- the first screen change that serves the new sub-goal.

Opening a menu or dialog is not a change in itself; it marks one only if it is
the first action towards a new sub-goal. A pause for thought without a
perceivable change does not set a boundary. The first sub-goal of a recording
also begins with a boundary – it follows the quiet start.

In code, this definition is the boundary definition `begin`. The definition
`end` (boundary at the end of a step) is carried along but is not the primary
one.

The same definition appears – as `SUBGOAL_DEFINITION` – in both model prompts,
so that the models answer the same question as the annotation
({doc}`../reference/model-prompts`).

## Reference annotation

The annotation is created in the annotation window and stored in
`ground_truth.json` ({doc}`../reference/data-formats`). Each entry states its
definition and its **run**: run 1 is the reference, runs 2, 3, … are
repetitions. All entries count equally; no distinction is made by the origin
of the data.

When reading, two edge rules of the guideline apply
({py:func}`~docupilot.evaluation.dataset.apply_edge_rules`):

- Boundaries in the first or last 0.3 s are dropped.
- Boundaries less than 0.1 s apart are merged to their mean – a double click
  on the annotation button is one boundary.

## Temporal uncertainty

A boundary is not an exactly reproducible instant: the reaction time lies
between the hint and the key press, and different hints of the same change lie
fractions of a second apart. The evaluation therefore does not measure
point-exactly but with a tolerance τ
({doc}`metrics`).
