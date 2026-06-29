
# Upload Multiple Test Images


import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

TARGET_SIZE = (80,80)

images = []
image_names = []

for filename in uploaded.keys():

    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, TARGET_SIZE)

    images.append(image)
    image_names.append(filename)

print(f"Total Images Uploaded : {len(images)}")


import pandas as pd
for image_number, image in enumerate(images):

                  print("="*80)
                  print(f"IMAGE {image_number+1}")
                  print(image_names[image_number])
                  print("="*80)

                  rows, cols, channels = image.shape

                  X = image.reshape((-1,3)).astype(np.float64)








                  print("Image Shape :",image.shape)
                  print("Feature Matrix :",X.shape)

                  cluster_candidates = [2,3,4,5]
                  experiment_results = {}
                  best_GSI = -np.inf
                  optimal_clusters = None
                  best_labels = None
                  best_centers = None
                  best_history = None


                  for M in cluster_candidates:
                      print("="*60)
                      print(f"Running SOC for M = {M}")
                      print("="*60)

                      soc = SOC_Algorithm( M_clusters=M,max_iterations=10)
                      labels, centers, history = soc.fit(X)
                      print(soc.best_GSI)
                      experiment_results[M] = {"labels": labels,"centers": centers,"history": history,"GSI": soc.best_GSI}
                      if soc.best_GSI > best_GSI:
                          best_GSI = soc.best_GSI
                          optimal_clusters = M
                          best_labels = labels
                          best_centers = centers
                          best_history = history

                  print("="*60)
                  print("Experimental Results")
                  print("="*60)
                  print("Optimal Number of Clusters :",optimal_clusters)
                  print("Maximum Global SI :",round(best_GSI,4))
                  print("="*60)


                  segmented = best_centers[best_labels.astype(int)]

                  # Convert back to 0–255
                  segmented = (segmented * 255).clip(0,255)

                  segmented = segmented.reshape(rows, cols, channels)





                  plt.figure(figsize=(12,5))

                  plt.subplot(121)
                  plt.imshow(image)
                  plt.title(f"Original Image\n{image_names[image_number]}")
                  plt.axis("off")

                  plt.subplot(122)
                  plt.imshow(segmented.astype(np.uint8))
                  plt.title(f"SOC Segmentation\n{image_names[image_number]} (M={optimal_clusters})")
                  plt.axis("off")

                  plt.show()








                  # Plot 1 : Global Silhouette Index vs Number of Clusters
                  cluster_list = sorted(experiment_results.keys())
                  gsi_values = [experiment_results[m]["GSI"] for m in cluster_list]
                  plt.figure(figsize=(6,4))
                  plt.plot(cluster_list, gsi_values, marker='o', linewidth=2)
                  plt.xlabel("Number of Clusters (M)")
                  plt.ylabel("Maximum Global Silhouette Index (GSI)")
                  plt.title(f"GSI vs Number of Clusters\n{image_names[image_number]}")
                  plt.grid(True)
                  plt.show()


                  # Plot 2 : GSI vs Iteration (Optimal Cluster Only)
                  optimal_history = best_history
                  iterations = np.arange(
                      1,
                      len(optimal_history["GSI"]) + 1
                  )
                  plt.figure(figsize=(6,4))
                  plt.plot(
                      iterations,
                      optimal_history["GSI"],
                      marker='o',
                      linewidth=2
                  )

                  plt.xlabel("SOC Iteration")
                  plt.ylabel("Global Silhouette Index (GSI)")
                  plt.title(f"GSI vs Iteration\n{image_names[image_number]} (M={optimal_clusters})")
                  plt.grid(True)
                  plt.show()



                  # Iteration-wise SOC Result Table
                  rows = []
                  for i in range(len(optimal_history["GSI"])):

                      rows.append({

                          "Iteration" : i + 1,

                          "Threshold δm" :
                          np.round(optimal_history["Threshold"][i],5),

                          "Cluster SI (Sm)" :
                          np.round(optimal_history["Sm"][i],4),

                          "GSI" :
                          round(optimal_history["GSI"][i],4),

                          "η (Optimal Threshold)" :
                          (
                              round(optimal_history["Eta"][i],5)
                              if i < len(optimal_history["Eta"])
                              else "-"
                          ),

                          "βm" :
                          (
                              np.round(optimal_history["Beta"][i],4)
                              if i < len(optimal_history["Beta"])
                              else "-"
                          )

                      })

                  results_table = pd.DataFrame(rows)

                  print("="*80)
                  print("="*80)
                  print(f"Image : {image_names[image_number]}")
                  print(f"Optimal Number of Clusters : {optimal_clusters}")
                  print(f"Maximum GSI : {best_GSI:.4f}")
                  print("="*80)
                  print("="*80)

                  display(results_table)
                  print("="*100)
                  print("End of Image", image_number + 1)
                  print("="*100)

                  results_table.to_csv(f"SOC_Results_{image_names[image_number].split('.')[0]}.csv",index=False)

