for MFCAD++ dataset first.ipynb(A small summary) to understand the code logiv
Chunk 0: Setup

Checks that the runtime is CPU (PyTorch 2.14.1+cpu, 13.6 GB RAM) and mounts Google Drive.
Creates the project folder with data, models, results and notes.
Writes helpers.py with functions to save and load JSON and to record finished chunks in progress.json, so work survives Colab disconnects.

Chunk 1: Load the dataset

Checks that the 1.6 GB MFCAD++ zip on Drive is valid and unzips it to Colab's local disk.
Counts files by type and folder. The STEP files total 59,665 (train 41,766, val 8,950, test 8,949), which matches the official numbers.
Prints feature_labels.txt, h5_structure.txt and the first lines of train.txt.

Chunk 2: Understand the data (val split)

Reads the readme and the h5 structure description. The dataset comes with ready-made hierarchical graphs: a face level, a mesh-facet level and the links between them.
Opens one batch and prints the shape of every array (faces, facets, labels, adjacency).
Counts faces per class on val: Stock is the largest class (22.3%), and Round (0.4%) and Circular through slot (0.5%) are the rarest. This is why we report macro-F1 and IoU, not only accuracy.

Chunk 2b: Verify the h5 files

Checks that the face adjacency is symmetric, that label counts match face counts, and what the surface-type codes mean (codes 1, 2, 3, 5, 7, 8 appear).
Finds that the h5 files hold fewer models than the official lists (val 8,639 of 8,950, about 3.5% missing).
Finds that idx marks where each model ends, and compares it with connected components of the face graph.

Chunk 3: Build the face-level cache

cache_builder.py converts each h5 batch into compact tensors: 16 face features (area, centroid, 12-slot one-hot surface type), class ids, face edges and edge types (convex, concave, smooth).
Saves train, val and test caches to Drive (train 40,292 models and 1.2 million faces).
Checks passed: no unknown edge types, surface codes decoded cleanly, about 30 faces per model in every split.
Tests the idea that one model is dropped at each batch boundary. The gaps are all positive and about one part wide, which is consistent with the idea but doesn't prove it.

Chunk 3b: Fix the part numbering

Connected components split some models into two pieces (batch 11: one model of 26 faces became 18 + 8).
Part numbers (face_model) are rebuilt from idx in all caches. The old numbering is kept as face_model_cc.
The numbering changed in exactly 30 val, 18 test and 92 train batches, as expected. Training was not affected, but the per-part analysis needed this.

Checkpoint cell (after the 12-hour break)

Lists which Drive files and progress records survived the reset. The caches and models were intact and only the unzipped data was gone.

Chunk 4: MLP baseline

A small network that sees only each face's own 16 features, with no neighbours.
Result on val: accuracy 0.441, macro-F1 0.227, mean IoU 0.149. Only Stock is recognised well (IoU 0.89).
This is the "no neighbours" reference point. It shows that a face's own features are not enough.

Chunk 5: Face-level GNN, version 1 and version 2

One message-passing layer per edge type (convex, concave, smooth), with mean over neighbours, residual connections and cross-entropy loss. It is written in plain PyTorch, so nothing extra needs installing.
v1 (hidden 64, 4 layers, 15 epochs): accuracy 0.801, macro-F1 0.775, mean IoU 0.664.
v2 (hidden 128, 6 layers, 30 epochs): accuracy 0.929, macro-F1 0.909, mean IoU 0.846.
A quick-test switch (TEST_RUN) and a checkpoint after every epoch keep the runs safe on free Colab.
A small fix cell restores the results files (v1 had been overwritten by v2). In the report, say that the v1 confusion matrix was not kept.

Chunk 6: Prepare the mesh level (6, 6a)

Unzips only the h5 files and prints the mesh arrays (V_2 facets, A_2 facet adjacency, A_3 facet-to-face links).
Verifies that A_3 maps each facet to exactly one face and that facet labels equal the label of their face.
Builds the mesh caches (12.8 million train facets). Checks: facet adjacency is symmetric in every batch, and idx[:,1] agrees with the face-based part numbers.
The 4th facet value (the plane offset d) has a huge range, so it is compressed with a signed log. That was my choice and is not from the paper.

Chunk 6b and 6c: Simplified hierarchical model (quick test, then full run)

Facet level: 2 neighbour rounds (hidden 32) on the mesh. Facets are averaged into their face and added to the face's own features.
Face level: the same 6-layer GNN as v2.
Result on val: accuracy 0.9455, macro-F1 0.9229, mean IoU 0.868, about 3 minutes per epoch. This is the best model, a gain of 1.6 points in accuracy over v2 (one run each).

Chunk 7: Confusion analysis (val)

The Slanted/Rectangular through step pair causes about a quarter of all errors (2,689 + 1,048 faces).
The other main confusions are wall-count mistakes (6-sided vs rectangular pocket and passage, triangular vs rectangular).
The paper reports the same weak spot.

Chunk 7b: Test evaluation (run once)

Reloads the final checkpoint. The check on val reproduces 0.9455, which shows the right weights are loaded.
Test: accuracy 0.9454, macro-F1 0.9222, mean IoU 0.8668 on 259,128 faces. This is almost identical to val.
The paper reports 97.37% accuracy on MFCAD++, so we are about 2.8 points below it.

Chunk 7c: Per-part errors (the last cell, which has no label)

Counts wrong faces per model using face_model and model_ids.
50.7% of the 8,632 test parts have no wrong face, and the worst 10% of parts hold 44.9% of all errors.
The worst parts are mostly wall-count confusions and O-ring predicted as Blind hole.
