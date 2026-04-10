# Adversarial Machine Learning Against Homomorphic Encryption Using Metadata Leakage

## A Complete Step-by-Step Project Explanation

This document is a deep, teaching-oriented explanation of the full project. It is intentionally more detailed than the README and is written to help someone understand not just how the code works, but why each design choice exists.

The goal is to make difficult concepts from cryptography, systems, and machine learning feel approachable. Whenever a concept is abstract, this guide uses analogies to make it easier to visualize.

---

## 1. Big Picture

### What does this project study?

This project studies whether an attacker can learn something useful about encrypted computations without ever seeing the plaintext.

In this case, the attacker does **not** try to break the encryption mathematically.

Instead, the attacker watches **metadata**:

- how long the encrypted computation took
- how large the ciphertext appears to be
- how many operations were performed
- how deep the multiplication chain became
- how much of the available noise margin seems to have been consumed

Then the attacker trains machine learning models to answer:

> "Can I guess what encrypted operation was executed just from the side-channel traces?"

That is the core research question of the project.

### The simplest intuition

Imagine five people are cooking in five separate sealed kitchens. You are not allowed to enter, and the windows are blacked out. You cannot see the ingredients, the utensils, or the food.

But you are allowed to observe:

- how long the kitchen stays active
- how much electricity it consumes
- how many times appliances turn on
- how noisy the kitchen is

Even without seeing inside, you may start guessing:

- "That one was probably baking."
- "That one seems like deep frying."
- "That one looks like blending a smoothie."

This project does the same thing for encrypted computation.

The encrypted data remains hidden, but the **shape of the work** may still leak information.

---

## 2. What Is FHE?

### FHE stands for Fully Homomorphic Encryption

Homomorphic encryption is a special kind of encryption that allows you to compute on encrypted data **without decrypting it first**.

Normally, encryption works like this:

1. You encrypt the data.
2. Someone stores or transmits it.
3. To compute on it, they must decrypt it first.

That means the system performing the computation gets access to the plaintext.

FHE changes this story:

1. You encrypt the data.
2. A remote server computes directly on the ciphertext.
3. The result stays encrypted.
4. Only the owner decrypts the output later.

### A simple analogy

Imagine you put numbers into a locked magic box. Someone else can shake, combine, scale, and transform those boxes in certain valid ways, even though they never open them.

At the end, you receive a new locked box.

Only you have the key, so only you can open it and read the answer.

That is the spirit of FHE.

### Why is FHE useful?

FHE is attractive because it promises privacy-preserving computation.

Examples:

- medical analytics on encrypted patient data
- cloud inference on sensitive financial records
- encrypted machine learning pipelines
- privacy-preserving outsourced computation

### Why is FHE hard?

Because performing arithmetic on encrypted values is computationally expensive and structurally constrained.

In normal computing, multiplying two numbers is easy.

In FHE, multiplying two ciphertexts is like asking:

> "How do I combine these two locked magic boxes in a way that still preserves the meaning of the hidden numbers?"

That is much more complicated.

---

## 3. Why CKKS?

### CKKS is the scheme used in this project

This project prefers the **CKKS** scheme because it is well-suited for approximate arithmetic on real-valued vectors.

### What does approximate arithmetic mean?

CKKS is designed for computations where tiny numerical error is acceptable.

That makes it a good fit for:

- machine learning
- statistics
- vector arithmetic
- signal processing

### Analogy

CKKS is like using a scientific calculator that rounds very slightly at many stages, but still gives answers accurate enough for practical numerical tasks.

It is not ideal when you need exact integer arithmetic like counting coins one by one.

It is ideal when you care about "close enough with high precision," like many ML workloads do.

---

## 4. What Is Metadata Leakage?

### Metadata

Metadata means "data about data."

If plaintext is the letter inside an envelope, metadata is everything visible from the outside:

- envelope size
- weight
- delivery time
- return address

In computing, metadata can include:

- timing
- resource usage
- packet size
- memory footprint
- operation counts

### Leakage

Leakage means some useful hidden pattern escapes indirectly.

The system may keep the message secret while still exposing clues.

### Combined meaning

Metadata leakage means:

> Even if the content stays encrypted, the system behavior still reveals useful information.

### Why this matters in FHE

FHE protects the **values**, but the system executing FHE still has runtime behavior.

Different encrypted programs often have different computational signatures.

For example:

- a mean operation mostly adds values together
- a variance calculation needs squaring
- logistic regression approximation needs deeper polynomial evaluation

Those differences change:

- time
- depth
- operation count
- noise depletion

That difference becomes the attack surface.

---

## 5. What Is an Adversarial ML Attack in This Context?

### Adversarial

In this project, "adversarial" means the machine learning model is used by an attacker, not a defender.

The attacker is not classifying cats and dogs.

The attacker is classifying hidden encrypted workloads.

### Machine learning’s role

ML is used because the relationship between metadata and workload type is noisy and multidimensional.

A human might not easily look at five metadata values and classify the operation every time.

But a trained model can find patterns across thousands of examples.

### Analogy

Think of a detective listening outside different locked rooms.

By hearing footsteps, time spent, and repeated sounds, the detective slowly learns to tell:

- "this room is probably someone typing"
- "this one sounds like someone assembling furniture"
- "this one sounds like cooking"

The detective never sees inside. They learn from repeated external patterns.

That is what the adversarial ML model does here.

---

## 6. What the Project Does End to End

The project follows this full pipeline:

1. Set up an FHE context.
2. Encrypt synthetic vectors.
3. Run encrypted operations.
4. Collect metadata side-channels.
5. Build a labeled dataset.
6. Train models to infer operation type.
7. Compare baseline and defended runs.
8. Visualize the leakage.
9. Present everything in a UI.

Each stage mirrors a real research workflow.

---

## 7. Project Structure Explained

### `fhe/`

This folder contains the cryptography-facing layer.

- [fhe_setup.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\fhe\fhe_setup.py)
- [operations.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\fhe\operations.py)
- [metadata_collector.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\fhe\metadata_collector.py)

### `data/`

This folder generates and stores the metadata dataset.

- [dataset_generator.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\data\dataset_generator.py)
- [dataset.csv](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\data\dataset.csv)

### `models/`

This folder handles training and utility logic.

- [model_utils.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\models\model_utils.py)
- [train_sklearn.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\models\train_sklearn.py)
- [train_torch.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\models\train_torch.py)

### `evaluation/`

This folder computes and visualizes results.

- [metrics.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\evaluation\metrics.py)
- [plots.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\evaluation\plots.py)

### `utils/`

This contains general support logic like config and logging.

- [config.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\utils\config.py)
- [logger.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\utils\logger.py)

### App and pipeline entry points

- [main.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\main.py)
- [app.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\app.py)

### UI files

- [index.html](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\templates\index.html)
- [styles.css](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\static\styles.css)
- [app.js](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\static\app.js)

---

## 8. Step 1: FHE Setup

### File

[fhe_setup.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\fhe\fhe_setup.py)

### What this module does

It creates the encrypted computing environment.

This includes:

- initializing the CKKS context
- generating keys
- encrypting vectors
- decrypting vectors
- wrapping encrypted objects in a common container

### Important terminology

#### Context

The FHE context is the cryptographic environment.

It defines:

- what encryption scheme is used
- what parameter sizes are used
- what scale is used
- what keys are available

### Analogy for context

The context is like the rulebook and toolset for a board game.

Before anyone starts playing, everyone must agree:

- what board is used
- what pieces exist
- what moves are legal

Similarly, before encrypted computation begins, the system must know the rules of the cryptographic world it operates in.

#### Keys

Keys are cryptographic permissions.

In simple terms:

- a public key lets you encrypt
- a secret key lets you decrypt
- auxiliary keys help with special homomorphic operations

#### Galois keys

These are special keys used for vector rotations and certain structured operations on packed ciphertexts.

### Analogy

If an encrypted vector is like a sealed train of compartments, Galois keys help rearrange the compartments without opening them.

#### Relinearization keys

These help simplify ciphertexts after multiplication.

### Analogy

Imagine multiplying ciphertexts causes a tool to become bulky and awkward. Relinearization is like compacting the tool back into a manageable shape after a heavy operation.

#### Global scale

In CKKS, values are represented in a scaled form. The scale helps preserve precision through approximate arithmetic.

### Simple explanation

You can think of scale as the "decimal precision budget" the ciphertext carries during numerical work.

If the scale becomes badly managed, numerical operations can break or become unstable.

That is why the TenSEAL fixes mattered earlier in the project.

---

## 9. Step 2: Encrypted Operations

### File

[operations.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\fhe\operations.py)

### What this module does

This module defines the encrypted computations that the attacker later tries to infer.

The implemented operations are:

1. Mean
2. Variance
3. Dot product
4. Linear regression inference
5. Logistic regression approximation

### Why these operations?

Because they create progressively different computational signatures:

- `mean` is simple
- `variance` introduces nonlinear structure
- `dot_product` combines multiplication and reduction
- `linear_regression_inference` resembles real ML inference
- `logistic_regression_approx` adds polynomial depth

That makes them ideal classes for a metadata-leakage study.

### Important terminology

#### Ciphertext

A ciphertext is encrypted data.

### Analogy

If plaintext is a readable note, ciphertext is the note after being scrambled into a locked puzzle box.

#### Encrypted domain

This means operations are being performed on ciphertexts, not plaintext values.

The code intentionally does **not** decrypt inside the operation functions.

That is very important, because otherwise the experiment would not reflect encrypted computation.

#### Dot product

The dot product multiplies corresponding entries in two vectors and sums them.

Example:

`[a, b, c] . [x, y, z] = ax + by + cz`

### Why dot product matters

Dot products are everywhere in ML.

Linear models, neural networks, and many statistical methods rely on them.

#### Linear regression inference

This computes a weighted sum plus bias.

Simple form:

`y = w.x + b`

### Analogy

Think of a teacher grading with weights:

- homework counts some amount
- quizzes count some amount
- final exam counts some amount
- then a bias term adjusts the final score

That is similar to a weighted sum.

#### Logistic regression approximation

Normally logistic regression uses the sigmoid function:

`sigmoid(x) = 1 / (1 + e^-x)`

But true sigmoid is expensive in FHE because exponentials and division are hard.

So the project uses a **polynomial approximation**.

### Analogy

If exact sigmoid is like drawing a perfectly smooth curved road, the polynomial approximation is like building a simpler road with a few straight and gently curved sections that still follows the same route closely enough.

---

## 10. Step 3: Metadata Collection

### File

[metadata_collector.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\fhe\metadata_collector.py)

### What this module does

It records the side-channel features associated with each encrypted operation.

The core recorded features are:

- `time`
- `size`
- `ops`
- `depth`
- `noise`

### Why these features?

Because they reflect how expensive or complex the encrypted circuit was.

### Terminology and intuition

#### Execution time

How long the operation took.

### Analogy

If two chefs both cook behind closed doors, the one who takes longer may be preparing a more complicated dish.

#### Ciphertext size

This is a proxy for how large the encrypted object is.

It is not the plaintext size. It is the structural size of the encrypted representation.

### Analogy

Two gifts may both be secret, but one comes in a larger box because it requires more protective packaging.

#### Operation count

A rough count of how many homomorphic steps were required.

### Analogy

If a recipe requires 3 steps and another requires 20 steps, even if both produce hidden dishes, the execution trace looks different.

#### Multiplicative depth

This is one of the most important FHE ideas.

It tracks how many layers of multiplication occur along the deepest path of the encrypted computation.

### Why multiplication matters so much

Additions are relatively cheap in FHE.

Multiplications are much more expensive and consume more of the ciphertext’s usable capacity.

### Analogy

Imagine encrypted computation as hiking with a battery-powered flashlight:

- additions are like walking on flat ground
- multiplications are like steep uphill climbs

Multiplicative depth is like the maximum number of steep hills in the hardest route.

#### Noise budget

In FHE, ciphertexts accumulate noise as operations are performed.

Too much noise can make decryption fail or become unreliable.

The "noise budget" is the remaining safety margin.

### Analogy

Imagine writing on paper through layers of carbon copies. Each copy gets fuzzier. If you keep copying again and again, the text eventually becomes unreadable. Noise budget is how many more copies you can make before the result becomes too blurry.

### Why this project sometimes uses a proxy

Some libraries do not expose an exact noise budget in an easy way, especially through Python-friendly abstractions.

So this project tracks a **noise hint** or noise-like proxy that behaves similarly for research purposes.

That is acceptable in a research prototype focused on side-channel trends.

---

## 11. Step 4: Dataset Generation

### File

[dataset_generator.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\data\dataset_generator.py)

### What this module does

It creates the dataset that the attacker’s ML model trains on.

Each row corresponds to:

- one encrypted computation
- one metadata trace
- one label describing the hidden operation type

### Label

A label is the correct answer the model should learn to predict.

Examples:

- `mean`
- `variance`
- `dot_product`
- `linear_regression_inference`
- `logistic_regression_approx`

### Why balanced data matters

The dataset is generated so every class appears equally often.

### Analogy

If one class appears 95% of the time, a lazy model can look smart by guessing that class constantly.

A balanced dataset prevents that shortcut.

### Why synthetic inputs are used

The project uses random synthetic vectors, not real sensitive data.

That is important because:

- the goal is to study metadata behavior, not real personal information
- the experiment remains safe and reproducible
- class distributions can be carefully controlled

### Why variability was added

A major improvement in the project was reducing overly perfect class separation.

Now the dataset generator introduces:

- variable vector lengths
- different workload amplitudes
- sparsity variation
- realistic jitter and rounding effects

This makes the metadata more believable.

### Analogy

Instead of assuming every chef follows the exact same recipe timing every day, the simulation now allows:

- slight delays
- ingredient differences
- stove variability
- different batch sizes

That makes the side-channel more realistic.

---

## 12. Step 5: Feature Preparation

### File

[model_utils.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\models\model_utils.py)

### What this module does

It converts the raw dataframe into ML-ready inputs.

This includes:

- extracting the feature matrix
- encoding string labels as integers
- scaling numerical features
- splitting into train and test sets

### Terminology

#### Feature

A feature is an input variable used by the model.

In this project:

- time
- size
- ops
- depth
- noise

These are the features.

#### Feature matrix

This is the table of all input features, typically called `X` in machine learning.

#### Labels / targets

These are the correct outputs, typically called `y`.

#### Label encoding

ML models usually want class labels as integers instead of strings.

So:

- `mean` might become `3`
- `variance` might become `4`

The mapping is stored so results can be interpreted later.

#### Standardization / scaling

Different features may have very different numerical ranges.

Example:

- `time` could be `0.0008`
- `size` could be `320`
- `ops` could be `27`

If not scaled, some models may overreact to features just because their numbers are larger.

### Analogy

Suppose you compare height in meters, weight in kilograms, and annual income in rupees. If you do not normalize them, income dominates numerically even if it is not the most meaningful feature.

Scaling brings features into a more comparable range.

#### Train-test split

The dataset is divided into:

- training set: used to learn patterns
- test set: used to evaluate generalization

### Why this matters

If you evaluate on the same data used for training, you may fool yourself into thinking the model is better than it really is.

### Analogy

It is like memorizing the answers to one practice exam and claiming you understand the subject. A real test should use unseen questions.

---

## 13. Step 6: Baseline Models

### File

[train_sklearn.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\models\train_sklearn.py)

### What this module trains

- Dummy classifier
- Random Forest
- SVM

### Why include a dummy classifier?

Because every serious experiment needs a weak baseline.

The dummy classifier tells us what happens if we make almost no intelligent prediction at all.

### Analogy

If five classes are equally likely, blind guessing is roughly 20% accurate. Any meaningful model should do better than that.

This gives context to the stronger models.

### Random Forest

A Random Forest is a collection of decision trees.

Each tree asks branching questions like:

- Is `depth > 2`?
- Is `noise < 85`?
- Is `ops > 14`?

Then many trees vote together.

### Analogy

Imagine a panel of detectives. Each one notices different clues and makes a guess. The final answer is based on majority voting.

That is more reliable than trusting a single detective.

### Why Random Forest works well here

Because metadata leakage is often nonlinear and threshold-based.

Random Forests are very good at discovering those patterns in tabular data.

### SVM

SVM stands for Support Vector Machine.

It tries to separate classes by building decision boundaries in feature space.

With nonlinear kernels, it can carve out complex boundaries.

### Analogy

Imagine dots painted on a sheet, each color representing a different class. SVM tries to place fences so that colors stay separated as cleanly as possible.

---

## 14. Step 7: Advanced Model

### File

[train_torch.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\models\train_torch.py)

### What it implements

A simple multilayer perceptron, or MLP.

Architecture:

- Linear layer
- ReLU activation
- Linear layer

### Terminology

#### Neural network

A neural network is a flexible function approximator made of layers.

It transforms inputs through learned weights to produce outputs.

#### Linear layer

A linear layer computes:

`output = W x input + b`

It is a weighted combination of inputs plus a bias.

#### ReLU

ReLU means Rectified Linear Unit.

It outputs:

- `x` if `x > 0`
- `0` otherwise

### Analogy

Think of ReLU as a one-way gate that blocks negative flow and lets positive flow continue.

#### Epoch

One epoch means one full pass through the training data.

#### Batch size

Instead of training on the full dataset all at once, the model trains on small chunks called batches.

### Why the Torch model may behave differently

Neural networks often need:

- enough data
- enough training time
- careful tuning

For small metadata datasets, classical tabular models like Random Forest may outperform simple neural networks.

That is not a bug. It is common in practice.

---

## 15. Step 8: Evaluation

### Files

- [metrics.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\evaluation\metrics.py)
- [plots.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\evaluation\plots.py)

### Metrics used

- Accuracy
- Precision
- Recall
- F1 score
- Confusion matrix

### Accuracy

Accuracy is the fraction of predictions that were correct.

### Example

If the model predicts correctly on 80 out of 100 samples, accuracy is 80%.

### Precision

Precision asks:

> "When the model predicts class X, how often is it correct?"

### Recall

Recall asks:

> "Out of all true class X samples, how many did the model successfully find?"

### F1 score

F1 balances precision and recall.

It is useful when you want one number that punishes models that are strong on one and weak on the other.

### Confusion matrix

A confusion matrix shows where the classifier gets confused.

Rows are true labels.

Columns are predicted labels.

### Analogy

Imagine a teacher grading students’ answers by category. The confusion matrix is like a detailed report showing not just how many were right, but exactly which categories students confused with each other.

### Feature importance

For Random Forest, feature importance estimates which metadata features the model relied on most.

### Why this is scientifically useful

Because it tells us **what is leaking**.

Maybe time is the strongest clue.

Maybe multiplicative depth is the strongest clue.

That helps both attack understanding and defense design.

---

## 16. Step 9: Defense Mechanism

### What defense is implemented?

The project includes a simple defense:

**dummy metadata noise injection**

This means the recorded metadata is perturbed slightly before model training and evaluation.

### Why this matters

If the model performs much worse after metadata perturbation, that suggests the leakage is sensitive to those features.

### Analogy

If an eavesdropper identifies songs by hearing rhythm through a wall, introducing extra background noise makes the rhythm harder to distinguish.

The song is still playing, but the side-channel becomes less useful.

### Important note

This is a simulation-style defense, not a production-certified mitigation strategy.

Its role is to demonstrate the idea of leakage reduction.

---

## 17. Step 10: Main Pipeline

### File

[main.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\main.py)

### What it does

This is the main orchestrator.

It:

1. loads configuration
2. initializes the FHE backend
3. generates baseline data
4. generates defended data
5. saves the dataset
6. trains the models
7. computes metrics
8. saves plots
9. returns a structured result object

### Why orchestration matters

A research prototype is not just isolated code fragments.

It needs a reproducible workflow.

The pipeline ensures the experiment can be rerun consistently.

---

## 18. Step 11: The UI

### Files

- [app.py](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\app.py)
- [index.html](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\templates\index.html)
- [styles.css](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\static\styles.css)
- [app.js](C:\Users\Rushabh\Downloads\Semester 10\CS\CS Project\static\app.js)

### Why add a UI?

Because research code is much easier to understand when someone can:

- run the experiment visually
- change parameters
- compare baseline and defense
- inspect plots in one place
- understand the narrative behind the numbers

### What the UI provides

- simulation controls
- backend selection
- configurable sample count
- configurable epochs
- explanatory project narrative
- result cards
- model comparison
- plot gallery

### Why this matters for presentation

For a project demo, raw logs are hard to follow.

A UI turns the prototype into a teachable system.

---

## 19. Why 100% Accuracy Can Be a Bad Sign

Earlier in the project, some runs gave perfect accuracy.

At first glance that sounds impressive.

But in research, perfect accuracy can also indicate a flawed experiment.

### Why?

Because the classes may be too cleanly separated in an unrealistic way.

For example:

- one operation may always use shorter vectors
- one class may always have distinctly different size
- metadata noise may be too low

Then the model is not learning subtle leakage.

It is exploiting obvious synthetic shortcuts.

### Analogy

Suppose you design a "face recognition" task where one class always wears red shirts and the other always wears blue shirts. The model may get 100% accuracy without learning faces at all. It only learned shirt color.

That is why the project was improved to:

- introduce more overlap
- vary workload sizes
- add jitter
- include a dummy baseline
- show warnings in the UI when accuracy looks suspiciously perfect

That makes the experiment more honest.

---

## 20. Why the TenSEAL Path Needed Fixes

When the project used the real TenSEAL backend, some operations initially failed with errors like:

- `scale out of bounds`

### Why that happens

CKKS manages approximate arithmetic using scales and modulus levels.

Repeated ciphertext multiplication increases complexity.

If the scale becomes too large relative to the remaining modulus chain, the operation can fail.

### Analogy

Imagine stacking books on a shelf with limited height.

Each multiplication adds another thick book.

If the shelf is too short, the stack no longer fits.

The project solved this by:

- adjusting CKKS scale
- extending the modulus chain
- simplifying the variance formula
- using a lower-degree polynomial for logistic approximation on the TenSEAL path

These are practical engineering fixes, not cryptographic shortcuts.

---

## 21. Difficult Concepts Explained Simply

### Homomorphic encryption

Locked-box arithmetic.

You can compute on sealed boxes without opening them.

### Ciphertext

The locked box itself.

### Plaintext

The original number or message before locking.

### Noise budget

How much computational "clarity" remains before the ciphertext becomes too degraded.

### Multiplicative depth

The number of multiplication layers in the hardest branch of the computation.

### Side-channel

Information leaked indirectly through system behavior rather than content.

### Metadata

Observable traces about the computation, not the hidden values.

### Feature

An input clue given to the ML model.

### Label

The correct answer the model should learn to predict.

### Baseline model

A simple reference model used for comparison.

### Overfitting

When the model memorizes training patterns too specifically and does not generalize well.

### Generalization

The ability to perform well on unseen data.

### Confusion matrix

A table showing exactly which classes the model mixes up.

### Approximation

Using a simpler mathematical substitute that is close enough for practical use.

---

## 22. What Makes This a Good Research Prototype

This project is strong because it combines:

- cryptographic computation
- systems-level side-channel reasoning
- dataset engineering
- adversarial ML
- evaluation
- visualization
- reproducibility

It also explicitly avoids a misleading claim:

> It does not break FHE mathematically.

That is important.

The point is not "encryption is useless."

The point is:

> Even strong encryption can still leak information through surrounding system behavior.

That is a much more realistic and scientifically valuable insight.

---

## 23. How to Explain the Whole Project in One Minute

If you need a very short presentation version:

> This project demonstrates a side-channel attack against homomorphic encryption systems. We run several encrypted computations, collect only metadata such as runtime, ciphertext size proxy, operation count, multiplicative depth, and noise proxy, then train machine learning models to infer which hidden computation was executed. The results show that even when plaintext remains secure, system-level traces can still leak workload identity. We also simulate a defense by injecting metadata noise and compare accuracy before and after the defense.

---

## 24. How to Explain It to a Non-Technical Audience

You can say:

> Think of encrypted computation as doing math inside locked boxes. No one can see the numbers inside. But if an attacker watches how long the locked-box process takes, how much work it seems to do, and how its behavior changes, they may still guess what kind of task is happening. This project shows that such external clues can reveal hidden computation types, even though the encryption itself is not broken.

---

## 25. Final Takeaway

The main lesson of this project is simple:

> Protecting the contents of computation is not always enough. You must also think about what the system reveals around the computation.

FHE protects the data.

But metadata can still speak.

This project gives that idea a complete experimental form:

- encrypted workloads
- metadata capture
- dataset creation
- adversarial ML training
- evaluation
- defense comparison
- interactive visualization

That is what makes it a complete, modular, production-style research prototype.
