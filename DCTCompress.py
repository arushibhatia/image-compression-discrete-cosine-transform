import numpy as np
from matplotlib import pyplot as plt


def compression(original_image, quality_factor):
    """

    Args:
        original_image: csv file for the orignal image values
        quality_factor: quality factor for compression

    Returns: N/A

    """
    # Create 8 x 8 blocks and call DCT for each of them
    image = np.loadtxt(original_image, delimiter=',')
    rows, columns = int(image.shape[0]), int(image.shape[1])

    # Create an empty array to store the transformed values
    transform = np.full(image.shape, dtype=float, fill_value=0.0)

    # Iterate over 8x8 blocks and transform
    for i in range(int(rows/8)):
        for j in range(int(columns/8)):
            transform[(8*i):(8*i + 8),(8*j):(8*j + 8)] = DCT(image[(8*i):(8*i + 8),(8*j):(8*j + 8)])

    # Create empty array for quanitized values
    quantized = np.full(transform.shape, dtype=int, fill_value=0)

    # Iterate and Quantize
    for i in range(int(rows/8)):
        for j in range(int(columns/8)):
            quantized[(8 * i):(8 * i + 8), (8 * j):(8 * j + 8)] = quantize(transform[(8 * i):(8 * i + 8),
                                                                 (8 * j):(8 * j + 8)], quality_factor)

    # Print the stats for Problem 4a adn 4b
    print_quantization_stats(quantized, quality_factor)
    run_length_encoding(quantized)

    # calculate and plot the inverse
    inverse = np.full(quantized.shape, dtype=float, fill_value=0.0)
    for i in range(int(quantized.shape[0] / 8)):
        for j in range(int(quantized.shape[1] / 8)):
            inverse[(8 * i):(8 * i + 8), (8 * j):(8 * j + 8)] = inverse_quantize(quantized[(8 * i):(8 * i + 8), (8 * j):(8 * j + 8)], quality_factor)
            inverse[(8 * i):(8 * i + 8), (8 * j):(8 * j + 8)] = inverse_DCT(inverse[(8 * i):(8 * i + 8), (8 * j):(8 * j + 8)])

    plt.figure()
    plt.imshow(inverse, cmap='gray')
    plt.show()


def DCT(pixel_block):
    """
    This function will take in a pixel block and complete the 2D
    discrete cosine transformation. This will be performed by first
    computing the DCT for every row, and then on the row-transformed
    matrix, compute the DCT on each column.

    Args:
        pixel_block(matrix (nested numpy arrays)): pixel block to
        perform DCT on

    Returns:
        (matrix (nested numpy arrays)): DCT matrix for pixel_block
    """
    # DCT on the input rowwise
    row_trans = dct_helper(pixel_block)

    # Transpose and then perform the transform column wise
    transposed = np.array(row_trans).transpose()
    col_trans = dct_helper(transposed)

    # Return back to original shape (re-transpose) and return as np.array
    return np.array(col_trans).transpose()


def dct_helper(block):
    """
    Helper to perform a DCT on given block
    Args:
        block: list of lists containing values for transform

    Returns:
        list of list of transformed values
    """
    # Get dimensions
    N = int(block.shape[0])
    M = int(block.shape[1])

    # Set output block
    output = [[0]*M for i in range(N)]

    for i in range(N):  # sets the current row
        for k in range(M):  # Following the doc convention, sets the current column
            for n in range(M):  # loop over all n cells in the row

                # Perform the transform function
                output[i][k] += (block[i, n]) * (np.cos((np.pi/M)*(n+0.5)*k))

    return output


def inverse_DCT(DCT_matrix):
    """
    This function will take in a DCT matrix and compute the original
    values by performing inverse DCT.

    Args:
        DCT_matrix(matrix (nested numpy arrays)): pixel block to
        perform inverse DCT on

    Returns:
        original_matrix(matrix (nested numpy arrays)): original matrix
        recovered from DCT_matrix
    """
    # Create temp matrix for column inverse and then original matrix for final inverse
    temp_matrix = np.full(DCT_matrix.shape, dtype=float, fill_value=0.0)
    original_matrix = np.full(DCT_matrix.shape, dtype=float, fill_value=0.0)

    # Column Inverse
    for j in range(int(DCT_matrix.shape[1])):  # j = current column
        for i in range(int(DCT_matrix.shape[0])):  # i = index in column j that we are trying to compute coefficient
            x_ij = 1 / 2 * DCT_matrix[0, j]
            for k in range(1, int(DCT_matrix.shape[0])):  # k is being used to do the summation
                x_ij += DCT_matrix[k, j] * np.cos((np.pi / int(DCT_matrix.shape[0]) * (i + 1 / 2) * k))
            temp_matrix[i, j] = 2 / int(DCT_matrix.shape[0]) * x_ij

    # Row Inverse
    for i in range(int(temp_matrix.shape[0])):  # i = current row
        for j in range(int(temp_matrix.shape[1])):  # j = index in row i that we are trying to compute coefficient
            x_ij = 1/2*temp_matrix[i, 0]
            for k in range(1, int(temp_matrix.shape[1])):  # k is being used to do the summation
                x_ij += temp_matrix[i, k] * np.cos((np.pi / int(temp_matrix.shape[1]) * (j + 1 / 2) * k))
            original_matrix[i, j] = 2/int(temp_matrix.shape[1]) * x_ij

    return original_matrix


def quantize(DCT_matrix, quality_factor):
    """
    This function will take in a DCT matrix and create a sparse matrix
    by quantizing it. Quantizing involves two steps. The first step is
    to divide (element-wise) each entry in the DCT_matrix with its
    corresponding value in the Quantization Table stored in quant.csv.
    The second step is to divide each of those numbers further uniformly by
    the quality factor, and round the resulting numbers to the nearest integer.

    Args:
        DCT_matrix(matrix (nested numpy arrays)): pixel block to
        perform inverse DCT on

        quantize_table(csv file): represents quantize table that helps us turn the DCT_matrix
        into a sparse matrix

        quality_factor(int): number that lets you "tune" how aggressive
        the wavelet compression will be

    Returns:
        quantized_matrix(matrix (nested numpy arrays)): quantized matrix computed
    """
    # Initialize quantize table and an empty quantized matrix
    quantize_table = np.genfromtxt('quant.csv', delimiter=',')
    quantized_matrix = np.full(DCT_matrix.shape, dtype=float, fill_value=0.0)

    # Get N and M values
    N = int(DCT_matrix.shape[0])
    M = int(DCT_matrix.shape[1])

    # For-loop iterating over DCT_matrix, for each entry, divide
    # it by corresponding entry in quant.csv and quality factor, and round to nearest int
    for i in range(N):
        for j in range(M):
            quantized_matrix[i, j] = DCT_matrix[i, j]/(quantize_table[i, j] * (quality_factor))

    # Cast as int and return
    quantized_matrix = quantized_matrix.astype(int)
    return quantized_matrix


def print_quantization_stats(quantized_matrix, quality_factor):
    """
    This function will take in a quantization matrix and calculate the following stats:
    - Quality factor used
    - Fraction of total quantized coefficients that are nonzero
    - Number of bytes to store the original uncompressed image (a)
    - Number of bytes needed to store the sparse compressed coefficients (b)
    - Compression ratio (a/b)

    Args:
        quantized_matrix(matrix (nested numpy arrays)): quantized matrix whose stats
        we are printing

        quality_factor(int): number that was used to tune "tune" how aggressive
        the wavelet compression is

    Returns:
        quantized_matrix(matrix (nested numpy arrays)): quantized matrix computed
    """
    print("---QUANTIZATION STATS---")
    print("Image was compressed with a quality factor of " + str(quality_factor))

    num_non_zero = np.count_nonzero(quantized_matrix)
    num_zero = np.size(quantized_matrix) - num_non_zero

    # For loop, increment num_zero or num_non_zero respectively depending on the value
    # in the quantized matrix

    print("Number of zero numbers in the quantized matrix " + str(num_zero))
    print("Number of non-zero numbers in the quantized matrix " + str(num_non_zero))

    print("Fraction of non-zero numbers in the quantized matrix/total numbers " +
          str(num_non_zero) + "/" + str(np.size(quantized_matrix)))

    # CALCULATE: Number of bytes to store the original uncompressed image (a)
    # This is the number of entries in the original matrix (same dimensions as quantized matrix)
    # multiplied by log_2(255) = 8 bits = 1 byte to be able to store all of the original values. Thus,
    # the number of entries is equal to the number of bytes required to store all original values.
    bytes_original = np.size(quantized_matrix)
    print("Number of bytes to store the original uncompressed image " + str(bytes_original))

    # CALCULATE: Number of bytes needed to store the sparse compressed coefficients (b)
    # For each non-zero number in the quantized matrix, we must store three numbers,
    # the actual value, the row index, and the column index.
    # We are assuming simplicity that each individual number stored in these three lists is one byte
    bytes_sparse_format = num_non_zero * 3
    print("Number of bytes needed to store the quantized matrix in sparse row format " + str(bytes_sparse_format))

    # CALCULATE: Compression ratio (a/b)
    print("Compression ratio using sparse row format: " + str(bytes_original / bytes_sparse_format))


def run_length_encoding(quantized_matrix):
    """
    This function will take in a quantized matrix (sparse) and represent it using
    run length encoding to encode it in a compressed manner. We flatten the matrix
    into one long vector (consisting of the first row, then the second row, and so on).
    Then, rather than storing each value separately, we store (value, repetition) pairs.

    Args:
        quantized_matrix(matrix (nested numpy arrays)): quantized matrix to create
        run length encoding of

    Returns:

    """
    # Get size of matrix
    quantized_matrix_size = np.size(quantized_matrix)

    # Get original values and set counter to 0; initialize repetition pairs
    current_value = quantized_matrix[0][0]
    current_counter = 0
    value_repetition_pairs = []

    # Access each value
    for row in quantized_matrix:
        for entry in row:

            # if the value is the same, add to counter
            if entry == current_value:
                current_counter += 1

            # If different, add past value and count and reset (to 1 here)
            else:
                value_repetition_pairs.append([current_value, current_counter])
                current_value = entry
                current_counter = 1

    # Add final value seen from list that were not added in for loop
    value_repetition_pairs.append([current_value, current_counter])

    print("---RUN LENGTH ENCODING STATS---")
    print("Number of pairs " + str(len(value_repetition_pairs)))

    # Assuming for simplicity that a given pair requires 1.5 bytes to store on average,
    # the compression ratio (number of bytes to store the original uncompressed image
    # divided by the number of bytes needed to store the sparse compressed coefficients) is:
    print("Number of bytes to store the original uncompressed image " + str(quantized_matrix_size))

    print("Number of bytes needed to store the quantized matrix in run length encoding format "
          + str(len(value_repetition_pairs) * 1.5))

    print("Compression ratio using run length encoding format: "
          + str(quantized_matrix_size / (len(value_repetition_pairs) * 1.5)))


def inverse_quantize(quantized_matrix, quality_factor):
    """
    This function will take in a quantized matrix and recover the DCT matrix.
    There may be some loss in this recovery, as in the quantization process,
    the rounding resulted in loss of some data.
    This is done by re-multiplying element-wise by the quantization matrix
    and quality factor in order to get the values back on the original scale.

    Args:
        quantized_matrix(matrix (nested numpy arrays)): quantized matrix to
        perform inverse quantization on

        quality_factor(int): number that was used in quantization to "tune"
        how aggressive the wavelet compression will be, and is needed to perform
        inverse quantization

    Returns:
        DCT_matrix(matrix (nested numpy arrays)): DCT matrix computed by performing
        inverse quantization
    """
    # For-loop iterating over quantized_matrix, for each entry, multiply
    # it by corresponding entry in quant.csv and quality factor
    quantize_table = np.genfromtxt('quant.csv', delimiter=',')

    # Initialize a table for values
    inverse_quantized = np.full(quantized_matrix.shape, dtype=float, fill_value=0.0)

    # Loop and on each cell undo the quantize
    for i in range(int(quantized_matrix.shape[0])):
        for j in range(int(quantized_matrix.shape[1])):
            inverse_quantized[i,j] = quantized_matrix[i, j] * quantize_table[i, j]
            inverse_quantized[i, j] = inverse_quantized[i, j] * quality_factor

    return inverse_quantized


if __name__ == '__main__':
    print("\nCompression with Quality Factor 1:")
    compression('parrot.csv', 1)

    print("\nCompression with Quality Factor 2:")
    compression('parrot.csv', 2)

    print("\nCompression with Quality Factor 4:")
    compression('parrot.csv', 4)

    print("\nCompression with Quality Factor 32:")
    compression('parrot.csv', 32)




