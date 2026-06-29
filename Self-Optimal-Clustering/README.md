# Self-Optimal Clustering for Color Image Segmentation

This repository presents a Python implementation of the **Self-Optimal Clustering (SOC)** algorithm for color image segmentation, developed as part of the **EE656: Artificial Intelligence and Machine Learning** course at **IIT Kanpur**. The implementation follows the methodology proposed in the reference paper and reproduces the complete SOC framework from scratch.

The SOC algorithm improves conventional Mountain Clustering by replacing heuristic threshold selection with an adaptive threshold optimization strategy based on **Lagrange interpolation**. During optimization, the threshold is iteratively refined to maximize the **Global Silhouette Index (GSI)**, while simultaneously determining the optimal number of clusters for image segmentation.

The project includes implementations of **Mountain Clustering (MC), Improved Mountain Clustering (IMC-1), Improved Mountain Clustering (IMC-2), and Self-Optimal Clustering (SOC)**. Experimental evaluation was performed on benchmark RGB color images, and the proposed SOC algorithm was compared with **K-Means, FCM, EM, K-Medoid, IMC-1, and IMC-2** using standard clustering validation measures.

The repository also contains the complete implementation, experimental results, project report, presentation slides, and the original reference paper used during development.

### Key Features

* Complete implementation of the SOC algorithm from scratch.
* Adaptive threshold optimization using Lagrange interpolation.
* Automatic selection of the optimal number of clusters using GSI.
* Comparative evaluation with conventional clustering algorithms.
* Experimental validation on benchmark color images.

### Technologies Used

* Python
* NumPy
* OpenCV
* Matplotlib
* pandas
* Scikit-learn
* Google Colab

