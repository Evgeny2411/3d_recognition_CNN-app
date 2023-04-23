# CNN shape recognition

This is end-to-end educational project of modeling in deploing CNN to classify .obj primitives.

This project implemented using PyTorch

Task description
----
A set of 3D primitives is given. We need to solve the problem of classifying objects using a neural network represented in the form of a point cloud.

Primitives model cen recognize: cone, cube, cylinder, plane, torus, sphere.

Data splited into: train, test, valid

For estimating model we'll use scikit classification_report on test set.

---
Requirements for the application
---

Functionality:
 - Ability to load user's .obj 3d model
 - Show rendered 3d object
 - Show how confident model in prediction
 
 Application deploed by docker container.
 CI/CD maked by GitHub Actions
 
 
