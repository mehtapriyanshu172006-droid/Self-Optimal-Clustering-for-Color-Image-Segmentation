
# Steps11-13: Find the Optimal Threshold (η)
#
# Step 11 :Set the desired Silhouette value S(δ) = 1
# Step 12 :Evaluate the Lagrange interpolation polynomial
#           over the threshold search interval
# Step 13 : Select the threshold η for which |S(δ) - 1| is minimum



def find_optimal_threshold(delta_m, S_m):


    # delta_m : Threshold values (δ1, δ2, ..., δM)

    # S_m : Cluster-wise Silhouette Index
    # (S1, S2, ..., SM)

    # Desired Silhouette value
    target_S = 1.0

    # Search interval for threshold
    delta_min = 0.0
    delta_max = np.max(delta_m) * 2

    search_delta = np.linspace(
        delta_min,
        delta_max,
        1000
    )

    # Initialize optimum threshold
    eta = delta_min

    # Initialize minimum interpolation error
    minimum_error = np.inf

    # Evaluate interpolation polynomial
    for delta in search_delta:

        # Predicted Silhouette value S(δ)
        S_predicted = lagrange_interpolate(
            delta_m,
            S_m,
            delta
        )

        # Difference from ideal value
        interpolation_error = abs(
            S_predicted - target_S
        )

        # Update optimum threshold
        if interpolation_error < minimum_error:

            minimum_error = interpolation_error
            eta = delta

    # eta : Optimized threshold (η)
    return eta
