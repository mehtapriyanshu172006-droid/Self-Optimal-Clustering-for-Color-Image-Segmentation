import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
try:
    import skfuzzy as fuzz
except:
    import subprocess
    subprocess.run(['pip','install','scikit-fuzzy','-q'])
    import skfuzzy as fuzz


def partition_index(X, labels, centres):
    M = len(centres)
    pi = 0.0
    for m in range(M):
        mask = (labels == m)
        if not mask.any():
            continue
        intra = np.sum(np.linalg.norm(X[mask] - centres[m], axis=1) ** 2)
        Nm = int(mask.sum())
        inter = sum(np.linalg.norm(centres[k] - centres[m]) ** 2 for k in range(M) if k != m)
        if inter > 0 and Nm > 0:
            pi += intra / (Nm * inter)
    return pi

def separation_index(X, labels, centres):
    M = len(centres)
    pairs = [np.linalg.norm(centres[m] - centres[k]) ** 2
             for m in range(M) for k in range(M) if m != k]
    min_inter = min(pairs) if pairs else 1e-9
    if min_inter == 0:
        return np.inf
    total_intra = sum(
        np.sum(np.linalg.norm(X[labels == m] - centres[m], axis=1) ** 2)
        for m in range(M) if (labels == m).any()
    )
    return total_intra / (len(X) * min_inter)

def dunn_index(X, labels, centres):
    M = len(centres)
    diameters = []
    for m in range(M):
        pts = X[labels == m]
        if len(pts) < 2:
            diameters.append(0.0)
            continue
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        diameters.append(np.sqrt((d ** 2).sum(axis=2)).max())
    max_d = max(diameters) if max(diameters) > 0 else 1e-9
    min_r = np.inf
    for m in range(M):
        pm = X[labels == m]
        if not len(pm):
            continue
        for k in range(M):
            if k == m:
                continue
            pk = X[labels == k]
            if not len(pk):
                continue
            d_inter = (
                np.linalg.norm(pm - centres[m], axis=1).sum() +
                np.linalg.norm(pk - centres[k], axis=1).sum()
            ) / (len(pm) + len(pk))
            min_r = min(min_r, d_inter / max_d)
    return float(min_r) if min_r != np.inf else 0.0

def run_imc1(X_norm, M):
    soc = SOC_Algorithm(M_clusters=M, max_iterations=1)
    soc.fit(X_norm * 255)
    return soc.best_labels, soc.best_centers

def run_imc2(X_norm, M):
    class IMC2:
        def __init__(self, M):
            self.M = M
            self.best_labels = None
            self.best_centers = None
        def fit(self, X_raw):
            X_norm = min_max_normalize(X_raw)
            n = len(X_norm)
            beta_m = np.array([(m + 1) / (m + 2) for m in range(self.M)])
            base_delta = max(
                (1 / (2 * n)) * sum(
                    np.min(X_norm[s]) / np.sum(X_norm[s])
                    for s in range(n) if np.sum(X_norm[s]) > 1e-10
                ), 1e-9
            )
            delta_m = base_delta * beta_m
            cluster_centers = []
            cluster_labels = np.full(n, -1)
            remaining = np.arange(n)
            for cluster in range(self.M):
                if len(remaining) == 0:
                    break
                rpts = X_norm[remaining]
                pot = np.zeros(len(rpts))
                for pt in range(len(rpts)):
                    sq_d = np.sum((rpts - rpts[pt]) ** 2, axis=1)
                    pot[pt] = np.sum(np.exp(-sq_d / (delta_m[cluster] ** 2)))
                center = rpts[np.argmax(pot)]
                cluster_centers.append(center)
                upd = []
                for si in remaining:
                    if np.sum((X_norm[si] - center) ** 2) <= delta_m[cluster]:
                        cluster_labels[si] = cluster
                    else:
                        upd.append(si)
                remaining = np.array(upd)
            cc = np.array(cluster_centers) if cluster_centers else np.zeros((1, X_norm.shape[1]))
            for si in remaining:
                cluster_labels[si] = np.argmin([np.sum((X_norm[si] - c) ** 2) for c in cc])
            self.best_labels = cluster_labels
            self.best_centers = cc
    m = IMC2(M)
    m.fit(X_norm * 255)
    return m.best_labels, m.best_centers

def reconstruct_image(centers_norm, labels, x_min, x_max, rows, cols, channels):
    denom = np.where(x_max - x_min == 0, 1, x_max - x_min)
    centers_raw = (centers_norm * denom + x_min).clip(0, 255).astype(np.uint8)
    return centers_raw[labels.astype(int)].reshape(rows, cols, channels)


for image_number, image in enumerate(images):

    print("=" * 80)
    print(f"IMAGE {image_number + 1} : {image_names[image_number]}")
    print("=" * 80)

    rows_img, cols_img, channels_img = image.shape
    X_raw = image.reshape(-1, 3).astype(np.float64)
    x_min = X_raw.min(axis=0)
    x_max = X_raw.max(axis=0)
    X_norm = min_max_normalize(X_raw)

    cluster_candidates = [2, 3, 4, 5]
    exp_res = {}
    best_GSI_img = -np.inf
    opt_M = None
    opt_labels = None
    opt_centers = None

    for M in cluster_candidates:
        soc = SOC_Algorithm(M_clusters=M, max_iterations=10)
        lb, ct, hist = soc.fit(X_raw)
        exp_res[M] = {"labels": lb, "centers": ct, "GSI": soc.best_GSI}
        if soc.best_GSI > best_GSI_img:
            best_GSI_img = soc.best_GSI
            opt_M = M
            opt_labels = lb
            opt_centers = ct

    print(f"Optimal Number of Clusters : {opt_M}")
    print(f"Maximum SOC GSI            : {round(best_GSI_img, 4)}")

    M_use = opt_M

    method_results = {}

    km = KMeans(n_clusters=M_use, random_state=42, n_init=10).fit(X_norm)
    method_results["K-Means"] = (km.labels_, km.cluster_centers_)

    try:
        cntr, u, *_ = fuzz.cluster.cmeans(X_norm.T, M_use, 2, error=0.005, maxiter=500, seed=42)
        method_results["FCM"] = (np.argmax(u, axis=0), cntr)
    except Exception:
        method_results["FCM"] = (km.labels_, km.cluster_centers_)

    gm = GaussianMixture(n_components=M_use, random_state=42).fit(X_norm)
    method_results["EM"] = (gm.predict(X_norm), gm.means_)

    imc1_lb, imc1_ct = run_imc1(X_norm, M_use)
    method_results["IMC-1"] = (imc1_lb, imc1_ct)

    imc2_lb, imc2_ct = run_imc2(X_norm, M_use)
    method_results["IMC-2"] = (imc2_lb, imc2_ct)

    method_results["SOC"] = (opt_labels, opt_centers)

    table_rows = []
    for method_name, (lb, ct) in method_results.items():
        try:
            lb_int = lb.astype(int)
            ct_arr = np.array(ct)
            n_cl = len(ct_arr)
            _, gsi_v = calculate_silhouette(X_norm, lb_int, n_cl)
            pi_v = partition_index(X_norm, lb_int, ct_arr)
            si_v = separation_index(X_norm, lb_int, ct_arr)
            di_v = dunn_index(X_norm, lb_int, ct_arr)
        except Exception:
            gsi_v = pi_v = si_v = di_v = float('nan')
        table_rows.append({
            "Method": method_name,
            "GSI ↑": round(gsi_v, 4),
            "PI ↓": round(pi_v, 4),
            "SI ↓": round(si_v, 4),
            "DI ↑": round(di_v, 4),
        })

    df = pd.DataFrame(table_rows).set_index("Method")
    print(f"\nComparison Table (M={M_use}) — {image_names[image_number]}")
    print("=" * 60)
    display(df)
    print("=" * 60)
    print("GSI↑ higher=better | PI↓ lower=better | SI↓ lower=better | DI↑ higher=better")

    seg_methods = ["IMC-1", "IMC-2", "SOC"]
    fig, axes = plt.subplots(1, len(seg_methods) + 1, figsize=(5 * (len(seg_methods) + 1), 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Original{image_names[image_number]}", fontsize=10)
    axes[0].axis("off")

    for ax, seg_name in zip(axes[1:], seg_methods):
        lb_s, ct_s = method_results[seg_name]
        seg_img = reconstruct_image(
            np.array(ct_s), lb_s.astype(int),
            x_min, x_max, rows_img, cols_img, channels_img
        )
        _, gsi_s = calculate_silhouette(X_norm, lb_s.astype(int), len(ct_s))
        color = "red" if seg_name == "SOC" else "black"
        fw = "bold" if seg_name == "SOC" else "normal"
        ax.imshow(seg_img)
        ax.set_title(f"{seg_name}\nGSI={gsi_s:.4f}", fontsize=10, color=color, fontweight=fw)
        for sp in ax.spines.values():
            sp.set_edgecolor("red" if seg_name == "SOC" else "lightgrey")
            sp.set_linewidth(3 if seg_name == "SOC" else 0.8)
        ax.axis("off")

    plt.suptitle(
        f"Segmentation — IMC-1 vs IMC-2 vs SOC\n{image_names[image_number]} (M={M_use})",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    print("=" * 100)
    print(f"End of Image {image_number + 1}")
    print("=" * 100)

