# Otology Agent Benchmark Set

Use this benchmark to evaluate retrieval quality, evidence ranking, synthesis quality, calibration, and clinical usefulness across prompt or workflow changes.

## How To Use

1. Start a fresh chat session for each question if you want isolated results.
2. Save the full model answer and the cited sources.
3. Score each answer from 1 to 5 on the rubric below.
4. Compare average scores across prompt versions or model versions.

## Scoring Rubric

Score each category from 1 to 5:

- `Retrieval quality`: Did it retrieve the right kind of sources for the question?
- `Evidence ranking`: Did it prefer guidelines, systematic reviews, and trials over weak cohorts when appropriate?
- `Synthesis quality`: Did it answer the clinical question directly instead of stitching together disconnected papers?
- `Calibration`: Did it avoid overclaiming and clearly acknowledge thin, mixed, or low-certainty evidence?
- `Citation quality`: Were the cited sources relevant and did they support the claims made?
- `Clinical usefulness`: Would a clinician find the answer accurate, concise, and actionable?

Suggested interpretation:

- `5`: Excellent
- `4`: Good
- `3`: Mixed / acceptable but flawed
- `2`: Weak
- `1`: Poor / misleading

## Questions

### 1. Guidelines / Current Practice

These test whether the agent prefers guideline or consensus sources over random observational studies.

1. What are the current indications for tympanostomy tubes in children?
2. What do recent guidelines recommend for sudden sensorineural hearing loss?
3. What is the current guideline-based approach to otitis media with effusion in children?
4. When is imaging recommended for unilateral tinnitus or asymmetric sensorineural hearing loss?

### 2. Treatment Evidence

These test evidence hierarchy, endpoint-aware synthesis, and weak-evidence handling.

5. What is the best evidence for treating Meniere's disease, and which recommendations are based on weak rather than strong data?
6. What is the evidence for intratympanic steroids versus systemic steroids in sudden sensorineural hearing loss?
7. What is the best evidence for treating vestibular migraine in adults?
8. Does the Epley maneuver outperform medication for posterior canal BPPV?

### 3. Procedure / Surgical Questions

These test whether the agent overcalls retrospective surgical literature.

9. What are hearing and complication outcomes after pediatric tympanoplasty?
10. What is the evidence comparing canal wall up versus canal wall down mastoidectomy for cholesteatoma recurrence?
11. What are outcomes of cochlear implantation in single-sided deafness?
12. What predicts success or failure after stapes surgery?

### 4. Diagnosis / Workup

These test whether the agent can handle diagnostic accuracy and workup questions rather than only treatments.

13. What is the diagnostic accuracy of MRI for vestibular schwannoma in patients with asymmetric hearing loss?
14. How should superior semicircular canal dehiscence be diagnosed, and how reliable are VEMP and CT?
15. What is the evidence base for diagnosing Eustachian tube dysfunction in adults?
16. How should autoimmune inner ear disease be diagnosed, and what are the limitations of the evidence?

### 5. Weak / Mixed / Controversial Evidence

These are useful because the correct answer often requires restraint.

17. Is betahistine effective for Meniere's disease?
18. Do dietary sodium restriction and caffeine avoidance improve Meniere's symptoms?
19. Is balloon dilation effective for Eustachian tube dysfunction?
20. Do antibiotics improve outcomes for uncomplicated tympanostomy tube otorrhea compared with drops alone?

### 6. Population-Specific Questions

These test subgroup handling and whether the agent keeps the question focused.

21. What are cochlear implant outcomes in older adults?
22. What is the evidence for tympanostomy tubes in children with Down syndrome or cleft palate?
23. How does chronic otitis media affect hearing outcomes in underserved pediatric populations?
24. What are vestibular rehabilitation outcomes in older adults with chronic vestibular hypofunction?

### 7. Negative Controls

These should trigger honest scope limitation rather than confident hallucination.

25. What is the best treatment for chronic rhinosinusitis with nasal polyps?
26. What are the latest recommendations for pediatric tonsillectomy?
27. What is the evidence for glaucoma screening in adults?

## Score Sheet

Copy this block for each test run:

```text
Question:
Prompt version:
Model:
Date:

Retrieval quality: __/5
Evidence ranking: __/5
Synthesis quality: __/5
Calibration: __/5
Citation quality: __/5
Clinical usefulness: __/5

Total: __/30

Notes:
-
-
-
```

## Fast Comparison Template

Use this when comparing two versions side by side:

```text
Question:

Version A total: __/30
Version B total: __/30

Winner:

Why:
-
-
-
```
