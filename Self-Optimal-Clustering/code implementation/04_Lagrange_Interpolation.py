
# Lagrange Interpolation
# Constructs the Threshold–SI interpolation polynomial
# using (δm , Sm) pairs (Paper Equations 8–10)


def lagrange_interpolate(delta_m, S_m, delta):

   # delta_m : Threshold values (δ1, δ2, ..., δM)
   # S_m : Cluster-wise Silhouette Index   (S1, S2, ..., SM)
   #delta : Threshold at which the interpolation polynomial is evaluated

   # Number of interpolation points
    M = len(delta_m)

    # Interpolated SI value
    S_delta = 0.0

    # Compute Lagrange interpolation polynomial
    for i in range(M):

        # ith Lagrange basis polynomial
        L_i = S_m[i]

        for j in range(M):

            if i != j:

                denominator = delta_m[i] - delta_m[j]

                # Avoid division by zero
                if denominator == 0:
                    denominator = 1e-10

                L_i *= (delta - delta_m[j]) / denominator

        S_delta += L_i

  # S(delta) : Interpolated Silhouette value  at the current threshold delta using this updated  relationship
    return S_delta
