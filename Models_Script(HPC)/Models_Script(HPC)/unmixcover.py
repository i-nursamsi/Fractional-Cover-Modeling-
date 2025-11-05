import numpy as np


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class UnmixError(Error):
    """
    Exception raised to trap errors in the unmixing routine.
    """
    pass


def unmix_fractional_cover(surface_reflectance, fc_model, inNull=0, outNull=0):
    """
    Unmixes an array of surface reflectance.

    Args:
        surface_reflectance (3d array): The surface reflectance data.
          Shape is nbands x nrows x ncolumns. There should be 6 bands,
          and values should be scaled to be between 0 and 1.

        fc_model (tflite_runtime.interpreter.Interpreter): The tensor flow model.
          Should be initiated like:
            `fc_model = tflite.Interpreter(model_path="path/to/model")`

        inNull, outNull (float): Null values (inNull) in the input image will
          be replaced by outNull.

    Returns:
        3D Array of fractional cover, where
            
        .. highlight:: python
        .. code-block:: python
        
          layer 1 = Bare
          layer 2 = Green
          layer 3 = Non-green


    """
    # Drop the Blue band. Blue is yukky
    inshape = surface_reflectance[1:].shape
    # reshape and transpose so it is (nrow x ncol) x 5
    ref_data = np.reshape(surface_reflectance[1:], (inshape[0], -1)).T

    # Run the prediction
    inputDetails = fc_model.get_input_details()
    outputDetails = fc_model.get_output_details()
    fc_model.resize_tensor_input(inputDetails[0]['index'], ref_data.shape)
    fc_model.allocate_tensors()
    fc_model.set_tensor(inputDetails[0]['index'], ref_data.astype(np.float32))
    fc_model.invoke()
    fc_layers = fc_model.get_tensor(outputDetails[0]['index']).T
    output_fc = np.reshape(fc_layers, (3, inshape[1], inshape[2]))
    # now do the null value swap
    output_fc[output_fc == inNull] = outNull
    return output_fc
