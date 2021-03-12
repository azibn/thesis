import numpy as np
from PyAstronomy.pyaC import pyaErrors as PE
from PyAstronomy.pyasl import _ic


"""The Python function for Cross Correlation (with adjustements to the final cross correlation function). Original source code
for the cross correlation function can be found at https://github.com/sczesla/PyAstronomy."""

def crosscorrRV(w, f, tw, tf, rvmin, rvmax, drv, mode="doppler", skipedge=0, edgeTapering=None):
    if not _ic.check["scipy"]:
        raise (PE.PyARequiredImport("This routine needs scipy (.interpolate.interp1d).", \
                                    where="crosscorrRV", \
                                    solution="Install scipy"))
    import scipy.interpolate as sci
    # Copy and cut wavelength and flux arrays
    w, f = w.copy(), f.copy()
    if skipedge > 0:
        w, f = w[skipedge:-skipedge], f[skipedge:-skipedge]

    if edgeTapering is not None:
        # Smooth the edges using a sine
        if isinstance(edgeTapering, float):
            edgeTapering = [edgeTapering, edgeTapering]
        if len(edgeTapering) != 2:
            raise (PE.PyAValError("'edgeTapering' must be a float or a list of two floats.", \
                                  where="crosscorrRV"))
        if edgeTapering[0] < 0.0 or edgeTapering[1] < 0.0:
            raise (PE.PyAValError("'edgeTapering' must be (a) number(s) >= 0.0.", \
                                  where="crosscorrRV"))
        # Carry out edge tapering (left edge)
        indi = np.where(w < w[0] + edgeTapering[0])[0]
        f[indi] *= np.sin((w[indi] - w[0]) / edgeTapering[0] * np.pi / 2.0)
        # Carry out edge tapering (right edge)
        indi = np.where(w > (w[-1] - edgeTapering[1]))[0]
        f[indi] *= np.sin((w[indi] - w[indi[0]]) / edgeTapering[1] * np.pi / 2.0 + np.pi / 2.0)

    # Speed of light in km/s
    c = 299792.458
    # Check order of rvmin and rvmax
    if rvmax <= rvmin:
        raise (PE.PyAValError("rvmin needs to be smaller than rvmax.",
                              where="crosscorrRV", \
                              solution="Change the order of the parameters."))
    # Check whether template is large enough
    if mode == "lin":
        meanWl = np.mean(w)
        dwlmax = meanWl * (rvmax / c)
        dwlmin = meanWl * (rvmin / c)
        if (tw[0] + dwlmax) > w[0]:
            raise (PE.PyAValError("The minimum wavelength is not covered by the template for all indicated RV shifts.", \
                                  where="crosscorrRV", \
                                  solution=["Provide a larger template", "Try to use skipedge"]))
        if (tw[-1] + dwlmin) < w[-1]:
            raise (PE.PyAValError("The maximum wavelength is not covered by the template for all indicated RV shifts.", \
                                  where="crosscorrRV", \
                                  solution=["Provide a larger template", "Try to use skipedge"]))
    elif mode == "doppler":
        # Ensure that the template covers the entire observation for all shifts
        maxwl = tw[-1] * (1.0 + rvmin / c)
        minwl = tw[0] * (1.0 + rvmax / c)
        if minwl > w[0]:
            raise (PE.PyAValError("The minimum wavelength is not covered by the template for all indicated RV shifts.", \
                                  where="crosscorrRV", \
                                  solution=["Provide a larger template", "Try to use skipedge"]))
        if maxwl < w[-1]:
            raise (PE.PyAValError("The maximum wavelength is not covered by the template for all indicated RV shifts.", \
                                  where="crosscorrRV", \
                                  solution=["Provide a larger template", "Try to use skipedge"]))
    else:
        raise (PE.PyAValError("Unknown mode: " + str(mode), \
                              where="crosscorrRV", \
                              solution="See documentation for available modes."))
    # Calculate the cross correlation
    drvs = np.arange(rvmin, rvmax, drv)
    cc = np.zeros(len(drvs))
    for i, rv in enumerate(drvs):
        if mode == "lin":
            # Shift the template linearly
            fi = sci.interp1d(tw + meanWl * (rv / c), tf)
        elif mode == "doppler":
            # Apply the Doppler shift
            fi = sci.interp1d(tw * (1.0 + rv / c), tf)
        # Shifted template evaluated at location of spectrum
        cc[i] = np.sum((f - (np.mean(f))) * (fi(w) - (np.mean(fi(w))))) / np.sqrt(
            (np.sum((f - np.mean(f)) ** 2)) * (np.sum((fi(w) - np.mean(fi(w))) ** 2)))

    return drvs, cc



