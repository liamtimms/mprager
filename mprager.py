#!/usr/bin/env python3

import argparse
import logging
import os
import warnings

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from skimage.morphology import remove_small_holes, remove_small_objects

log = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPRAGEr: convert INV2 and UNI images from an MP2RAGE sequence into MPRAGE-like contrast"
    )
    parser.add_argument(
        "-i",
        "--inv2",
        type=str,
        help="Path to the INV2 image from the MP2RAGE sequence",
        default="INV2.nii.gz",
    )
    parser.add_argument(
        "-u",
        "--uni",
        type=str,
        help="Path to the UNI image from the MP2RAGE sequence",
        default="UNIT1.nii.gz",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the output MPRAGE-like image",
        default="mprager.nii.gz",
    )
    parser.add_argument(
        "-m",
        "--mask",
        type=str,
        help="Path to save the brain mask used during processing",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of output files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    # Mask source: provide an existing mask OR let the script compute one via thresholding.
    mask_source = parser.add_mutually_exclusive_group()
    mask_source.add_argument(
        "--input-mask",
        type=str,
        metavar="PATH",
        default=None,
        help=(
            "Path to a pre-existing brain mask. "
            "When provided, thresholding is skipped and this mask is used directly "
            "for bias-field correction and as the output mask."
        ),
    )
    mask_source.add_argument(
        "--mask-strategy",
        type=str,
        choices=["li", "otsu"],
        default="li",
        help="Thresholding strategy used to compute the brain mask automatically (default: li)",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if not os.path.exists(args.inv2):
        raise FileNotFoundError(f"INV2 image not found: {args.inv2}")
    if not os.path.exists(args.uni):
        raise FileNotFoundError(f"UNI image not found: {args.uni}")
    if args.input_mask is not None and not os.path.exists(args.input_mask):
        raise FileNotFoundError(f"Input mask not found: {args.input_mask}")
    if not args.force and os.path.exists(args.output):
        raise FileExistsError(f"Output file already exists: {args.output}")
    if args.mask is not None and os.path.exists(args.mask) and not args.force:
        raise FileExistsError(f"Mask output file already exists: {args.mask}")


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkNearestNeighbor,
) -> sitk.Image:
    """Resample image into the physical space of reference."""
    return sitk.Resample(
        image, reference, sitk.Transform(), interpolator, 0.0, image.GetPixelID()
    )


def same_space(a: sitk.Image, b: sitk.Image, tol: float = 1e-4) -> bool:
    """Return True if two images share the same grid within floating-point tolerance.

    Direct equality on float tuples from NIfTI headers is unreliable due to
    precision differences introduced by reading/writing files (e.g. 0.6999… vs 0.7).
    """
    if a.GetSize() != b.GetSize():
        return False

    def close(seq1, seq2) -> bool:
        return all(abs(x - y) < tol for x, y in zip(seq1, seq2))

    return (
        close(a.GetSpacing(), b.GetSpacing())
        and close(a.GetOrigin(), b.GetOrigin())
        and close(a.GetDirection(), b.GetDirection())
    )


def threshold_image(img: sitk.Image, strategy: str) -> sitk.Image:
    """Return a binary brain mask using the chosen thresholding strategy."""
    if strategy == "li":
        return sitk.LiThreshold(img, 0, 1)
    elif strategy == "otsu":
        return sitk.OtsuThreshold(img, 0, 1)
    else:
        raise ValueError(f"Unknown mask strategy: {strategy!r}")


def clean_mask(mask: NDArray, min_size: int = 100, hole_area_threshold: int = 500) -> NDArray:
    """Remove small spurious objects and fill holes in a binary mask.

    Parameters
    ----------
    min_size:
        Minimum object size in voxels. Objects smaller than this are removed.
    hole_area_threshold:
        Maximum hole area (voxels) to fill. The scikit-image default of 64 is
        far too small for 3-D brain masks at sub-mm resolution; 500 voxels
        (~8×8×8 cube) is a more conservative starting point.
    """
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = remove_small_holes(mask, area_threshold=hole_area_threshold)
    return mask


def main():
    args: argparse.Namespace = get_args()
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )
    validate_args(args)

    # Load and normalise INV2
    log.info("Loading INV2 from %s", args.inv2)
    inv2_img = sitk.ReadImage(args.inv2)
    inv2_img = sitk.Cast(inv2_img, sitk.sitkFloat32)
    inv2_img = sitk.RescaleIntensity(inv2_img, 0, 255)

    log.info("Loading UNI from %s", args.uni)
    uni_img = sitk.ReadImage(args.uni)
    uni_img = sitk.Cast(uni_img, sitk.sitkFloat32)

    # UNI and INV2 must share a grid for the final Multiply. In a standard
    # MP2RAGE acquisition they always do; warn and resample if they don't.
    if not same_space(uni_img, inv2_img):
        warnings.warn(
            f"UNI image grid {uni_img.GetSize()} differs from INV2 {inv2_img.GetSize()}; "
            "resampling UNI to INV2 space with linear interpolation.",
            stacklevel=2,
        )
        uni_img = resample_to_reference(uni_img, inv2_img, sitk.sitkLinear)

    # Determine the mask to use for bias-field correction.
    # If the user supplied a pre-existing mask, load and use it directly.
    # Otherwise, compute and clean one from INV2 using the selected thresholding strategy.
    if args.input_mask is not None:
        log.info("Loading input mask from %s", args.input_mask)
        bias_mask = sitk.ReadImage(args.input_mask)
        # Resample to INV2 space if the mask came from an external tool with a
        # different grid (e.g. BET run at a different resolution).
        if not same_space(bias_mask, inv2_img):
            warnings.warn(
                f"Input mask grid {bias_mask.GetSize()} differs from INV2 {inv2_img.GetSize()}; "
                "resampling mask to INV2 space with nearest-neighbour interpolation.",
                stacklevel=2,
            )
            bias_mask = resample_to_reference(bias_mask, inv2_img)
    else:
        log.info("Computing brain mask via %s thresholding", args.mask_strategy)
        raw_mask = threshold_image(inv2_img, args.mask_strategy)
        mask_arr = clean_mask(sitk.GetArrayFromImage(raw_mask))
        bias_mask = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        bias_mask.CopyInformation(raw_mask)

    # Shrink images for faster N4 bias field correction, then reconstruct at full res.
    # BinShrink averages over each neighbourhood block before downsampling, giving N4
    # a better-conditioned input than the stride-sampling of plain Shrink.
    log.info("Running N4 bias-field correction (this may take a few minutes)…")
    shrink_factors = [2] * inv2_img.GetDimension()
    shrunk_img = sitk.BinShrink(inv2_img, shrink_factors)
    mask_shrink = sitk.Shrink(bias_mask, shrink_factors)

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_corrector.Execute(shrunk_img, mask_shrink)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(inv2_img)
    corr_img_fullres = inv2_img / sitk.Exp(log_bias_field)

    log.info("Combining UNI × corrected-INV2 and rescaling…")
    mprage_img = sitk.Multiply(uni_img, corr_img_fullres)
    mprage_img = sitk.RescaleIntensity(mprage_img, 0, 4096)
    # Store as uint16: matches the integer dtype expected by downstream tools
    # (FreeSurfer, fMRIPrep, ANTs) and halves file size vs float32.
    mprage_img = sitk.Cast(mprage_img, sitk.sitkUInt16)

    log.info("Writing output to %s", args.output)
    sitk.WriteImage(mprage_img, args.output)

    # Save the mask used/computed during processing if an output path was given.
    if args.mask is not None:
        if args.input_mask is not None:
            # User supplied their own mask — write it through unchanged.
            log.info("Writing input mask to %s", args.mask)
            sitk.WriteImage(bias_mask, args.mask)
        else:
            # Derive a clean final mask from the MPRAGE output via thresholding.
            log.info("Computing final mask from MPRAGE output…")
            mask = threshold_image(mprage_img, args.mask_strategy)
            mask_arr = clean_mask(sitk.GetArrayFromImage(mask))
            mask_mod = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask_mod.CopyInformation(mask)
            log.info("Writing mask to %s", args.mask)
            sitk.WriteImage(mask_mod, args.mask)

    log.info("Done.")


if __name__ == "__main__":
    main()
