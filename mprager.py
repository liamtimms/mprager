#!/bin/env python

import argparse
import os

import SimpleITK as sitk
from numpy.typing import NDArray
from skimage.filters import try_all_threshold
from skimage.morphology import remove_small_holes, remove_small_objects


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MPRAGEr: take an INV1, INV2, and UNI image from MP2RAGE sequence on Siemens and mimic MPRAGE contrast using this "
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
        help="Path to save the mask image",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite of output files",
    )
    return parser.parse_args()


def validate_fnames(
    # inv1_fname: str,
    inv2_fname: str,
    uni_fname: str,
    output_fname: str,
    mask_fname: str,
    force: bool,
):
    # if not os.path.exists(inv1_fname):
    # raise FileNotFoundError(f"INV1 image not found: {inv1_fname}")
    if not os.path.exists(inv2_fname):
        raise FileNotFoundError(f"INV2 image not found: {inv2_fname}")
    if not os.path.exists(uni_fname):
        raise FileNotFoundError(f"UNI image not found: {uni_fname}")
    if not force and os.path.exists(output_fname):
        raise FileExistsError(f"Output file already exists: {output_fname}")
    if mask_fname is not None and not os.path.exists(mask_fname) and not force:
        raise FileExistsError(f"Mask file already exists: {mask_fname}")
    pass


def clean_mask(mask: NDArray, min_size: int = 100) -> NDArray:
    mask = remove_small_objects(mask, min_size=min_size)
    mask = remove_small_holes(mask)
    return mask


def main():
    args: argparse.Namespace = get_args()
    validate_fnames(
        # args.inv1,
        args.inv2,
        args.uni,
        args.output,
        args.mask,
        args.force,
    )
    # load images
    inv2_img = sitk.ReadImage(args.inv2)
    inv2_img = sitk.Cast(inv2_img, sitk.sitkFloat32)
    inv2_img = sitk.RescaleIntensity(inv2_img, 0, 255)
    mask = sitk.LiThreshold(inv2_img, 0, 1)
    uni_img = sitk.ReadImage(args.uni)
    shrunk_img = sitk.Shrink(inv2_img, [2] * inv2_img.GetDimension())
    mask_shrink = sitk.Shrink(mask, [2] * mask.GetDimension())
    # otsu_mask = sitk.OtsuThreshold(shrunk_img)
    # corr_img = sitk.N4BiasFieldCorrection(shrunk_img)
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # bias_corrector.SetMaximumNumberOfIterations([50] * 3)
    bias_corrector.Execute(shrunk_img, mask_shrink)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(inv2_img)
    corr_img_fullres = inv2_img / sitk.Exp(log_bias_field)
    uni_img = sitk.Cast(uni_img, sitk.sitkFloat64)
    mprage_img = sitk.Multiply(uni_img, corr_img_fullres)
    mprage_img = sitk.RescaleIntensity(mprage_img, 0, 4096)
    mprage_img = sitk.Cast(mprage_img, sitk.sitkFloat32)
    sitk.WriteImage(mprage_img, args.output)
    sitk.WriteImage(mask, "mask.nii.gz")

    mask = sitk.LiThreshold(mprage_img, 0, 1)
    mask_img = sitk.GetArrayFromImage(mask)
    mask_img = clean_mask(mask_img.astype(int))
    mask_mod = sitk.GetImageFromArray(mask_img.astype(int))
    mask_mod.CopyInformation(mask)
    sitk.WriteImage(mask_mod, "mask_mod.nii.gz")

    masked_uni = sitk.Mask(uni_img, mask_mod)
    sitk.WriteImage(masked_uni, "masked_uni.nii.gz")

    return


if __name__ == "__main__":
    main()
