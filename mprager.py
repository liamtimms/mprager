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
        help="Path to save the brain mask image",
        default=None,
    )
    parser.add_argument(
        "--mask-strategy",
        type=str,
        choices=["li", "otsu"],
        default="li",
        help="Thresholding strategy used to generate the brain mask (default: li)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of output files",
    )
    return parser.parse_args()


def validate_fnames(
    inv2_fname: str,
    uni_fname: str,
    output_fname: str,
    mask_fname: str,
    force: bool,
):
    if not os.path.exists(inv2_fname):
        raise FileNotFoundError(f"INV2 image not found: {inv2_fname}")
    if not os.path.exists(uni_fname):
        raise FileNotFoundError(f"UNI image not found: {uni_fname}")
    if not force and os.path.exists(output_fname):
        raise FileExistsError(f"Output file already exists: {output_fname}")
    if mask_fname is not None and os.path.exists(mask_fname) and not force:
        raise FileExistsError(f"Mask file already exists: {mask_fname}")


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
    validate_fnames(
        args.inv2,
        args.uni,
        args.output,
        args.mask,
        args.force,
    )
    # Load and normalise INV2
    inv2_img = sitk.ReadImage(args.inv2)
    inv2_img = sitk.Cast(inv2_img, sitk.sitkFloat32)
    inv2_img = sitk.RescaleIntensity(inv2_img, 0, 255)

    # Initial brain mask from INV2 (used to guide bias field correction)
    init_mask = threshold_image(inv2_img, args.mask_strategy)

    uni_img = sitk.ReadImage(args.uni)
    uni_img = sitk.Cast(uni_img, sitk.sitkFloat32)

    # Shrink images for faster N4 bias field correction, then reconstruct at full res
    shrunk_img = sitk.Shrink(inv2_img, [2] * inv2_img.GetDimension())
    mask_shrink = sitk.Shrink(init_mask, [2] * init_mask.GetDimension())

    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    bias_corrector.Execute(shrunk_img, mask_shrink)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(inv2_img)
    corr_img_fullres = inv2_img / sitk.Exp(log_bias_field)

    mprage_img = sitk.Multiply(uni_img, corr_img_fullres)
    mprage_img = sitk.RescaleIntensity(mprage_img, 0, 4096)
    mprage_img = sitk.Cast(mprage_img, sitk.sitkFloat32)
    sitk.WriteImage(mprage_img, args.output)

    # Derive final brain mask from the MPRAGE output and write if requested
    if args.mask is not None:
        mask = threshold_image(mprage_img, args.mask_strategy)
        mask_arr = clean_mask(sitk.GetArrayFromImage(mask))
        mask_mod = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        mask_mod.CopyInformation(mask)
        sitk.WriteImage(mask_mod, args.mask)


if __name__ == "__main__":
    main()
