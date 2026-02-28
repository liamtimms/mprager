#!/bin/env python

import argparse
import os

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from skimage.morphology import remove_small_holes, remove_small_objects


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


def threshold_image(img: sitk.Image, strategy: str) -> sitk.Image:
    """Return a binary brain mask using the chosen thresholding strategy."""
    if strategy == "li":
        return sitk.LiThreshold(img, 0, 1)
    elif strategy == "otsu":
        return sitk.OtsuThreshold(img, 0, 1)
    else:
        raise ValueError(f"Unknown mask strategy: {strategy!r}")


def clean_mask(mask: NDArray, min_size: int = 100) -> NDArray:
    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = remove_small_holes(mask)
    return mask


def main():
    args: argparse.Namespace = get_args()
    validate_args(args)

    # Load and normalise INV2
    inv2_img = sitk.ReadImage(args.inv2)
    inv2_img = sitk.Cast(inv2_img, sitk.sitkFloat32)
    inv2_img = sitk.RescaleIntensity(inv2_img, 0, 255)

    uni_img = sitk.ReadImage(args.uni)
    uni_img = sitk.Cast(uni_img, sitk.sitkFloat32)

    # Determine the mask to use for bias-field correction.
    # If the user supplied a pre-existing mask, load and use it directly.
    # Otherwise, compute one from INV2 using the selected thresholding strategy.
    if args.input_mask is not None:
        bias_mask = sitk.ReadImage(args.input_mask)
    else:
        bias_mask = threshold_image(inv2_img, args.mask_strategy)

    # Shrink images for faster N4 bias field correction, then reconstruct at full res
    shrunk_img = sitk.Shrink(inv2_img, [2] * inv2_img.GetDimension())
    mask_shrink = sitk.Shrink(bias_mask, [2] * bias_mask.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_corrector.Execute(shrunk_img, mask_shrink)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(inv2_img)
    corr_img_fullres = inv2_img / sitk.Exp(log_bias_field)

    mprage_img = sitk.Multiply(uni_img, corr_img_fullres)
    mprage_img = sitk.RescaleIntensity(mprage_img, 0, 4096)
    mprage_img = sitk.Cast(mprage_img, sitk.sitkFloat32)
    sitk.WriteImage(mprage_img, args.output)

    # Save the mask used/computed during processing if an output path was given.
    if args.mask is not None:
        if args.input_mask is not None:
            # User supplied their own mask — write it through unchanged.
            sitk.WriteImage(bias_mask, args.mask)
        else:
            # Derive a clean final mask from the MPRAGE output via thresholding.
            mask = threshold_image(mprage_img, args.mask_strategy)
            mask_arr = clean_mask(sitk.GetArrayFromImage(mask))
            mask_mod = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask_mod.CopyInformation(mask)
            sitk.WriteImage(mask_mod, args.mask)


if __name__ == "__main__":
    main()
