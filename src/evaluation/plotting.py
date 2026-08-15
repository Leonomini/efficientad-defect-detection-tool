import matplotlib.pyplot as plt


def plot_evaluation_summary(
    fpr_img,
    tpr_img,
    img_auroc,
    scores_normal,
    scores_anom,
    fprs,
    pros,
    score_threshold=None,
    save_path=None,
    dpi=300,
):
    """
    Creates a summary figure from precomputed evaluation results.

    The figure contains:
    (a) Image-level ROC curve
    (b) PRO curve
    (c) Distribution of image anomaly scores
    """

    # ---------- Figure ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ----------(a) Image-level ROC ----------
    ax = axes[0]
    ax.plot(fpr_img, tpr_img, label=f"AUC = {img_auroc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    ax.set_title("(a) Image-level ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)

    # ----------(b) PRO Curve ----------
    ax = axes[1]
    ax.plot(fprs, pros)

    ax.set_title("(b) PRO Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Per-Region Overlap")
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 1)
    ax.grid(True)

    # ---------- (c) Score Distribution ----------
    ax = axes[2]
    ax.hist(
        scores_normal,
        bins=40,
        density=True,
        alpha=0.6,
        label="Normal",
    )
    ax.hist(
        scores_anom,
        bins=40,
        density=True,
        alpha=0.6,
        label="Anomalous",
    )

    if score_threshold is not None:
        ax.axvline(
            score_threshold,
            linestyle="--",
            linewidth=1,
            label="Threshold",
        )

    ax.set_title("(c) Score Distribution")
    ax.set_xlabel("Image Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

    # ---------- Finalize ----------
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.close()
