
# Self-Optimal Clustering (SOC)
# This class implements the complete SOC algorithm proposed in the paper.
class SOC_Algorithm:
 # Initialize SOC parameters
    def __init__(self, M_clusters, max_iterations=10):
        # Number of clusters (M)
        self.M = M_clusters
        # Maximum SOC optimization iterations
        self.max_iter = max_iterations
        # Best Global Silhouette Index
        self.best_GSI = -1.0
        # Best clustering result
        self.best_labels = None
        # Best cluster centers
        self.best_centers = None


    #Threshold Function
    # Computes the initial threshold (δ)
    def _calculate_base_delta(self, X_norm):
        # Number of samples
        num_samples = len(X_norm)
        threshold_sum = 0.0
        for sample in range(num_samples):
          minimum_feature = np.min(X_norm[sample])
          feature_sum = np.sum(X_norm[sample])
          if feature_sum != 0:
            threshold_sum += (minimum_feature /feature_sum)
        base_delta = (1 / (2 * num_samples)) * threshold_sum
        return base_delta


    # Execute Self-Optimal Clustering
    def fit(self, X_raw):
        # Step 1 : Normalize data
        X_norm = min_max_normalize(X_raw)
        num_samples = len(X_norm)
        # Initialize optimization factor βm
        # Initially βm = 1
        beta_m = np.ones(self.M)

       # Step 2 : Compute initial threshold (δ)
        base_delta = self._calculate_base_delta(X_norm)
        # Store GSI of every SOC iteration
        history = {"Threshold": [],"Eta": [],"Beta": [],"GSI": [],"Sm": []}

        # SOC Optimization Loop
        for iteration in range(self.max_iter):
            # Step 2
            # Compute threshold for every cluster
            # δm = δ × βm
            delta_m = base_delta * beta_m
            # Cluster centers
            cluster_centers = []
            # Cluster labels
            cluster_labels = np.full(num_samples,-1 )
            # Initially all samples are active
            remaining_indices = np.arange(num_samples)

            # Steps 3 - 8
            # Improved Mountain Clustering (IMC)
            # for M clusters
            for cluster in range(self.M):
                # Stop if all samples are assigned
                if len(remaining_indices) == 0:
                   break
                # Remaining data points
                remaining_points = X_norm[remaining_indices]

                # Step 3
                # Compute Mountain Potential
                mountain_potential = np.zeros(len(remaining_points))
                for point in range(len(remaining_points)):
                    squared_distance = np.sum(( remaining_points -remaining_points[point]) ** 2,axis=1)
                    mountain_potential[point] = np.sum(np.exp(-squared_distance /(delta_m[cluster] ** 2)))

                # Step 4
                # Select point with maximum potential
                # as the cluster center
                center_index = np.argmax(mountain_potential)

                cluster_center = remaining_points[center_index]
                cluster_centers.append(cluster_center)

                # Steps 5-7
                # Assign samples lying inside δm
                # Remove assigned samples
                # Continue with remaining samples
                updated_remaining = []
                for sample_index in remaining_indices:
                       distance = sq_euclidean(X_norm[sample_index],cluster_center)
                       if distance <= delta_m[cluster]:
                        cluster_labels[sample_index] = cluster
                       else:
                        updated_remaining.append(sample_index)
                # Remaining samples for next cluster
                remaining_indices = np.array(updated_remaining)

            # Step 8 : Assign Remaining Samples
            # Assign all unclustered samples to the nearest
            # cluster center using squared Euclidean distance.
            for sample_index in remaining_indices:
                  distance_to_centers = [
                             sq_euclidean(
                             X_norm[sample_index],
                            center) for center in cluster_centers]

                  cluster_labels[sample_index] = np.argmin(distance_to_centers)

            # At this point IMC clustering is done

            # Step 9 : Compute Cluster-wise Silhouette Index
            # Calculates:
            #   Sm  : Silhouette Index of each cluster
            #   GSI : Global Silhouette Index
            S_m, GSI = calculate_silhouette(X_norm, cluster_labels,len(cluster_centers))

            # Store GSI of current SOC iteration
            history["GSI"].append(GSI)
            history["Threshold"].append(delta_m.copy())
            # history["Eta"].append(eta)
            # history["Beta"].append(beta_m.copy())
            history["Sm"].append(S_m.copy())



            # Store the best clustering obtained so far.
            # The paper selects the clustering corresponding
            # to the maximum GSI.
            if GSI > self.best_GSI:
                 self.best_GSI = GSI
                 self.best_labels = cluster_labels.copy()
                 self.best_centers = np.array(cluster_centers)


            # Steps 10 - 15
            # Threshold Optimization using
            if iteration < self.max_iter - 1:
               # Compute the optimized threshold (η)
               # using the Threshold–SI interpolation polynomial.
               eta = find_optimal_threshold(delta_m,S_m)

               history["Eta"].append(eta)
               history["Beta"].append(beta_m.copy())


                # Step 14
                # Compute Optimization Factor
                # βm = η / δm
               for cluster in range(self.M):
                    if delta_m[cluster] > 0:
                      beta_m[cluster] = (eta /delta_m[cluster])




                # Step 15
                # Updated βm is used in the next SOC iteration
                # to compute the new threshold values.
                # Return Final SOC Result


        return (self.best_labels, self.best_centers,history)
print("SOC Pipeline Done")

