using System;

namespace EasyOcrNet.ImageProcessing;

/// <summary>
/// Morphological operations for binary images.
/// </summary>
public static class MorphologyOps
{
    /// <summary>
    /// Performs binary dilation on an image with a rectangular structuring element.
    /// Uses separable kernel optimization (horizontal then vertical passes).
    /// </summary>
    /// <param name="input">Input binary image where true = foreground, false = background.</param>
    /// <param name="kernelWidth">Width of the structuring element (must be odd and positive).</param>
    /// <param name="kernelHeight">Height of the structuring element (must be odd and positive).</param>
    /// <returns>Dilated binary image.</returns>
    public static bool[,] Dilate(bool[,] input, int kernelWidth, int kernelHeight)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (kernelWidth <= 0 || kernelWidth % 2 == 0)
            throw new ArgumentException("Kernel width must be odd and positive.", nameof(kernelWidth));

        if (kernelHeight <= 0 || kernelHeight % 2 == 0)
            throw new ArgumentException("Kernel height must be odd and positive.", nameof(kernelHeight));

        int height = input.GetLength(0);
        int width = input.GetLength(1);

        if (height == 0 || width == 0)
            return new bool[height, width];

        // Optimization: Use separable kernel (horizontal then vertical)
        // This reduces complexity from O(W*H*kw*kh) to O(W*H*(kw+kh))
        var temp = DilateHorizontal(input, kernelWidth);
        return DilateVertical(temp, kernelHeight);
    }

    /// <summary>
    /// Performs horizontal dilation (1D) on each row.
    /// </summary>
    private static bool[,] DilateHorizontal(bool[,] input, int kernelWidth)
    {
        int height = input.GetLength(0);
        int width = input.GetLength(1);
        var output = new bool[height, width];
        int radius = kernelWidth / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Check if any pixel in the kernel window is foreground
                bool hasForeground = false;
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int nx = x + kx;
                    if (nx >= 0 && nx < width && input[y, nx])
                    {
                        hasForeground = true;
                        break;
                    }
                }
                output[y, x] = hasForeground;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs vertical dilation (1D) on each column.
    /// </summary>
    private static bool[,] DilateVertical(bool[,] input, int kernelHeight)
    {
        int height = input.GetLength(0);
        int width = input.GetLength(1);
        var output = new bool[height, width];
        int radius = kernelHeight / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Check if any pixel in the kernel window is foreground
                bool hasForeground = false;
                for (int ky = -radius; ky <= radius; ky++)
                {
                    int ny = y + ky;
                    if (ny >= 0 && ny < height && input[ny, x])
                    {
                        hasForeground = true;
                        break;
                    }
                }
                output[y, x] = hasForeground;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs binary erosion on an image with a rectangular structuring element.
    /// Uses separable kernel optimization.
    /// </summary>
    /// <param name="input">Input binary image where true = foreground, false = background.</param>
    /// <param name="kernelWidth">Width of the structuring element (must be odd and positive).</param>
    /// <param name="kernelHeight">Height of the structuring element (must be odd and positive).</param>
    /// <returns>Eroded binary image.</returns>
    public static bool[,] Erode(bool[,] input, int kernelWidth, int kernelHeight)
    {
        if (input == null)
            throw new ArgumentNullException(nameof(input));

        if (kernelWidth <= 0 || kernelWidth % 2 == 0)
            throw new ArgumentException("Kernel width must be odd and positive.", nameof(kernelWidth));

        if (kernelHeight <= 0 || kernelHeight % 2 == 0)
            throw new ArgumentException("Kernel height must be odd and positive.", nameof(kernelHeight));

        int height = input.GetLength(0);
        int width = input.GetLength(1);

        if (height == 0 || width == 0)
            return new bool[height, width];

        var temp = ErodeHorizontal(input, kernelWidth);
        return ErodeVertical(temp, kernelHeight);
    }

    private static bool[,] ErodeHorizontal(bool[,] input, int kernelWidth)
    {
        int height = input.GetLength(0);
        int width = input.GetLength(1);
        var output = new bool[height, width];
        int radius = kernelWidth / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Check if ALL pixels in the kernel window are foreground
                bool allForeground = true;
                for (int kx = -radius; kx <= radius; kx++)
                {
                    int nx = x + kx;
                    if (nx < 0 || nx >= width || !input[y, nx])
                    {
                        allForeground = false;
                        break;
                    }
                }
                output[y, x] = allForeground;
            }
        }

        return output;
    }

    private static bool[,] ErodeVertical(bool[,] input, int kernelHeight)
    {
        int height = input.GetLength(0);
        int width = input.GetLength(1);
        var output = new bool[height, width];
        int radius = kernelHeight / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                // Check if ALL pixels in the kernel window are foreground
                bool allForeground = true;
                for (int ky = -radius; ky <= radius; ky++)
                {
                    int ny = y + ky;
                    if (ny < 0 || ny >= height || !input[ny, x])
                    {
                        allForeground = false;
                        break;
                    }
                }
                output[y, x] = allForeground;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs morphological opening (erosion followed by dilation).
    /// Useful for removing small foreground noise.
    /// </summary>
    public static bool[,] Open(bool[,] input, int kernelWidth, int kernelHeight)
    {
        var eroded = Erode(input, kernelWidth, kernelHeight);
        return Dilate(eroded, kernelWidth, kernelHeight);
    }

    /// <summary>
    /// Performs morphological closing (dilation followed by erosion).
    /// Useful for filling small holes in foreground.
    /// </summary>
    public static bool[,] Close(bool[,] input, int kernelWidth, int kernelHeight)
    {
        var dilated = Dilate(input, kernelWidth, kernelHeight);
        return Erode(dilated, kernelWidth, kernelHeight);
    }
}
