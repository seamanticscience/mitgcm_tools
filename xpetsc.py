import dask
import glob
import os
import struct

import ecco_v4_py as ecco
import h5py       as hf
import numpy      as np
import pandas     as pd
import xarray     as xr

from numbers                import Integral
from xarray.backends        import BackendEntrypoint
from xarray.backends.common import BackendArray
from xarray.core.indexing   import (
    ExplicitIndexer,
    IndexingSupport,
    explicit_indexing_adapter,
)


def is_petsc_file(filename_or_obj):
    """
    Checks if a file appears to be a Petsc binary file with headers.
    This is based on the assumption that the file starts with an 8-byte header
    consisting of two 4-byte integers.
    """
    classids = {
        1211216: 'Mat',
        1211214: 'Vec',
        1211218: 'IS',
        1211219: 'Bag',
    }
    try:
        with open(filename_or_obj, 'rb') as f:
            # Read the first 8 bytes (two 4-byte integers)
            header_bytes = f.read(np.dtype('>f8').itemsize)
            if len(header_bytes) < np.dtype('>f8').itemsize:
                return False  # File is too small to contain a valid Petsc header
            # Unpack the header into two integers
            file_id, vector_length = struct.unpack('>ii', header_bytes)
            if file_id not in classids.keys():
                # Check if file_id is in the list of petsc file ids
                return False
            elif vector_length <= 0:
                # Check if vector_length is a reasonable value (e.g., positive and not too large)
                return False
            else:
                return True  # The file appears to have a valid Petsc header
    except Exception as e:
        raise e
        return False  # If there's an error reading or interpreting the header, it's not a Petsc file


class TMMBackendArray(BackendArray):
    def __init__(
        self,
        filename,
        shape,
        dtype         = np.dtype(">f8"),
        header_length = 2,
        header_dtype  = np.dtype(">i4"),
       lock           = dask.utils.SerializableLock()

    ):
        self.filename      = filename
        self.shape         = shape
        self.header_length = (
            header_length  # Number of integers in the header (can be set to 0)
        )
        self.header_dtype  = header_dtype
        self.dtype         = dtype
        # on-disk dtype (may be big-endian) and native-endian output dtype
        self.file_dtype = np.dtype(dtype)
        self.out_dtype = self.file_dtype.newbyteorder('=')
        self.dtype = self.out_dtype # xarray inspects this
        self.vector_length = shape[1]  # Length of each data vector
        self.lock          = lock

    def __getitem__(self, key: ExplicitIndexer):
        return explicit_indexing_adapter(
            key,
            self.shape,
            IndexingSupport.BASIC,
            self._raw_indexing_method,
        )


    def _raw_indexing_method(self, key: tuple):
        t_idx = key[0]
        v_idx = key[1] if len(key) > 1 else slice(None)

        t_is_int = isinstance(t_idx, Integral)
        if t_is_int:
            t_list = [t_idx]
        elif isinstance(t_idx, slice):
            t_list = list(range(*t_idx.indices(self.shape[0])))
        else:
            t_list = list(t_idx)

        v_is_int = isinstance(v_idx, Integral)
        if v_is_int:
            out = np.empty((len(t_list),), dtype=self.out_dtype)
        else:
            if isinstance(v_idx, slice):
                start, stop, step = v_idx.indices(self.vector_length)
                nvals = max(0, (stop - start + (step - 1)) // step)
            else:
                v_idx = np.asarray(v_idx)
                nvals = v_idx.size
            out = np.empty((len(t_list), nvals), dtype=self.out_dtype)

        for i, t in enumerate(t_list):
            arr = self._read_time_entry_slice(t, v_idx)
            if isinstance(arr, np.ndarray) and arr.dtype.byteorder not in ('=', '|'):
                arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
            out[i] = arr

        if t_is_int and v_is_int:
            return out[0]
        if t_is_int and not v_is_int:
            return out[0, :]
        if v_is_int:
            return out
        return out


    def _read_time_entry_slice(self, time_index, value_indices):
        with open(self.filename, 'rb') as f:
            # compute using Python ints to avoid NumPy int overflows
            header_bytes = int(self.header_length) * int(self.header_dtype.itemsize)
            row_bytes = int(self.vector_length) * int(self.file_dtype.itemsize)
            stride_bytes = header_bytes + row_bytes
            base = int(time_index) * stride_bytes + header_bytes

            # sanity bounds check
            file_size = os.path.getsize(self.filename)
            if base < 0 or base > file_size:
                raise IndexError(f"Computed base offset out of bounds: {base} (file size {file_size})")

            if isinstance(value_indices, slice):
                start, stop, step = value_indices.indices(self.vector_length)
                if step == 1:
                    f.seek(base + start * self.file_dtype.itemsize, 0)
                    count = max(0, stop - start)
                    arr = np.fromfile(f, dtype=self.file_dtype, count=count)
                else:
                    f.seek(base, 0)
                    row = np.fromfile(f, dtype=self.file_dtype, count=self.vector_length)
                    arr = row[value_indices]
            elif isinstance(value_indices, Integral):
                f.seek(base + value_indices * self.file_dtype.itemsize, 0)
                arr = np.fromfile(f, dtype=self.file_dtype, count=1)
                arr = arr.reshape(())
            else:
                f.seek(base, 0)
                row = np.fromfile(f, dtype=self.file_dtype, count=self.vector_length)
                arr = row[value_indices]

            if isinstance(arr, np.ndarray) and arr.dtype.byteorder not in ('=', '|'):
                arr = arr.byteswap().view(arr.dtype.newbyteorder('='))
            return arr


class TMMBackend(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        start_time       = 0, 
        time_step        = 1, 
        times            = None,
        num_time_entries = None,
        time_map_file    = None,
        drop_variables   = None,
        vector_length    = None,
        header_length    = 2,
        dtype            = np.dtype(">f8"),
        header_dtype     = np.dtype(">i4"),
    ):
        if header_length == 2 and not is_petsc_file(filename_or_obj):
            header_length = 0  # No Petsc header detected, treat as generic binary file

        # Add warnings or errors if necessary
        if header_length == 0 and vector_length == 0 and num_time_entries is None:
            raise ValueError(
                "num_time_entries must be provided if header_length is zero to determine vector_length from the file. Supply num_time_entries and optionally vector_length."
            )

        # Determine vector_length and num_time_entries if not provided
        file_size         = os.path.getsize(filename_or_obj)
        header_size_bytes = (
            header_length * header_dtype.itemsize
        )  # Number of integers (header_dtype.itemsize bytes each)

        if vector_length is None or num_time_entries is None:
            if vector_length is None:
                if header_length > 0:
                    with open(filename_or_obj, "rb") as f:
                        data = np.fromfile(
                            f, 
                            dtype = header_dtype, 
                            count = 2)
                        _, vector_length = data
                else:
                    # Calculate vector_length if header_size is zero and num_time_entries is provided
                    vector_length = int(
                        file_size - header_size_bytes
                    ) // (
                        num_time_entries * dtype.itemsize
                    )
                    if vector_length <= 0:
                        raise ValueError(
                            "Calculated vector_length is invalid. Please check the input parameters and file structure."
                        )

            vector_size_bytes = float(vector_length) * dtype.itemsize

            if num_time_entries is None:
                # Calculate num_time_entries by iterating until the remainder is zero
                num_time_entries = 1
                while (
                    np.remainder(
                        num_time_entries * (
                            header_size_bytes + vector_size_bytes
                        ),
                        file_size,
                    )
                    > 0
                ):
                    num_time_entries += 1
                if num_time_entries <= 0:
                    raise ValueError(
                        "Calculated num_time_entries is invalid. Please check the input parameters and file structure."
                    )

        # Validate file size consistency
        vector_size_bytes = float(vector_length) * dtype.itemsize
        expected_file_size = num_time_entries * (
            header_size_bytes + vector_size_bytes
        )
        if expected_file_size != file_size:
            raise ValueError(
                f"Inconsistent file size: expected {expected_file_size} bytes, but got {file_size} bytes. Please check the file structure and parameters."
            )

        shape = (num_time_entries, vector_length)
        data  = xr.core.indexing.LazilyIndexedArray(
            TMMBackendArray(
                filename_or_obj,
                shape,
                dtype         = dtype,
                header_length = header_length,
                header_dtype  = header_dtype,
                lock          = dask.utils.SerializableLock()
            )
        )
        
        # Time coordinate logic
        if time_map_file is not None:
            mapping = np.loadtxt(time_map_file)
            if mapping.shape[1] < 2:
                raise ValueError(f"{time_map_file} must have at least two columns: iteration and time.")
            iterations, time_values = mapping[:, 0], mapping[:, 1]
            if len(iterations) != num_time_entries:
                print("Warning: iteration count does not match num_time_entries — times will be truncated or interpolated.")
            time_coord = time_values[:num_time_entries]
        elif times is not None:
            if len(times) != num_time_entries:
                raise ValueError(
                    f"Length of provided times ({len(times)}) does not match "
                    f"num_time_entries ({num_time_entries})."
                )
            time_coord = np.array(times)
        else:
            time_coord = np.arange(
                start_time,
                start_time + num_time_entries * time_step,
                time_step,
            )

        # Create xarray Dataset
        coords = {
            "time": time_coord,
            "box" : np.arange(vector_length)
        }
        
        return xr.Dataset(
            {"tmm_data": (["time", "box"], data)},
            coords = coords,
        )

        def guess_can_open(self, filename_or_obj):
            try:
                _, ext = os.path.splitext(filename_or_obj)
            except TypeError:
                return False
            return ext in {".petsc", ".bin"}

    description = "Use TMM files in Xarray"

class binaryBackendArray(xr.backends.BackendArray):
    def __init__(
        self,
        filename_or_obj,
        shape,
        dtype,
        lock,
    ):
        self.filename_or_obj = filename_or_obj
        self.shape = shape
        self.dtype = dtype
        self.lock = lock

    def __getitem__(self, key: tuple):
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key: tuple):
        key0 = key[0]
        size = np.dtype(self.dtype).itemsize

        if isinstance(key0, slice):
            start = key0.start or 0
            stop = key0.stop or self.shape[0]
            offset = size * start
            count = stop - start
        else:
            offset = size * key0
            count = 1

        with self.lock, open(self.filename_or_obj) as f:
            arr = np.fromfile(f, self.dtype, offset=offset, count=count)

        if isinstance(key, int):
            arr = arr.squeeze()

        return arr

class binaryBackend(xr.backends.BackendEntrypoint):
    def open_dataset(self, filename_or_obj, *, drop_variables=None, dtype=np.float32):
        size = np.dtype(dtype).itemsize
        shape = os.stat(filename_or_obj).st_size // size

        backend_array = binaryBackendArray(
            filename_or_obj=filename_or_obj,
            shape=(shape,),
            dtype=dtype,
            lock=dask.utils.SerializableLock(),
        )
        data = xr.core.indexing.LazilyIndexedArray(backend_array)

        var = xr.Variable(dims=("x"), data=data)
        return xr.Dataset({"foo": var})
        
# Helper functions to read, write, and interact with Transport Matrix Method model output
def read_config_mat(path, llc_grid=None):
    """
    Import TMM config files
    """
    config = dict()

    # Add h5 reference
    config["h5ref"] = hf.File(path, "r")

    for key in hf.File(path, "r").keys():
        if (   (key.find("File") != -1)
            or (key.find("Path") != -1)
            or (key.find("Name") != -1)
           ):
            # Directory name or file
            if hf.File(path, "r").get(key)[()][0] != 0:
                config[key] = "".join(
                    chr(int(a)) for a in hf.File(path, "r").get(key)[()]
                )
            else:
                config[key] = hf.File(path, "r").get(key)[()][0]
        elif (key.find("#") != -1) or (key.find("gcmfacesdata") != -1):
            continue
        else:
            config[key] = hf.File(path, "r").get(key)[()][0]

        # Check that the "mat" file has been properly loaded
        # hdf references might still be pointers!
        # if type(config[key][0]) is hf.h5r.Reference:
        if not isinstance(config[key], str) and config[key].dtype == np.dtype(object):
            for f, facet in enumerate(config[key]):
                config[key][f] = config["h5ref"][facet][()]

            if config[key][0].shape[0] != 1:
                # implies an ECCO CS/LLC grid. Assume LLC90?
                # Convert 5 "faces" of LLC grid to 13 "tiles"
                config[key] = ecco.llc_faces_to_tiles(
                    dict(enumerate(config[key].flatten(), 1)), less_output=True
                )

                # Convert numpy array of 13 "tiles" to xarray dataset
                if config[key][0].ndim == 2:
                    config[key] = ecco.llc_tiles_to_xda(
                        data_tiles = config[key],
                        grid_da    = llc_grid.XC,
                        var_type   = "c",
                    )
                elif config[key][0].ndim == 3:
                    config[key] = ecco.llc_tiles_to_xda(
                        data_tiles = config[key],
                        grid_da    = llc_grid.hFacC,
                        var_type   = "c",
                        dim4       = "k",
                    )
    return config


def get_tmm_times(
    fileName,
    diagList=[
        "empflux_surf.bin",
        "gasexflux_surf.bin",
        "pco2_surf.bin",
        "fe_dust_input.bin",
    ],
):
    """
    Get times and iterations of petsc output files (_?? forcing, *ini, pickup*, *avg, and state output)
    """
    if len(glob.glob(fileName + "_*")) > 1:
        # this is monthly forcing (#isforcing)
        # Time axis is 12 montly usually
        at = ai = np.arange(12)
        nt = 12
    else:
        # Do some sleuthing for the time axis
        indir = "/".join(fileName.split("/")[:-1])
        fname = "".join(fileName.split("/")[-1])
        _, ext = os.path.splitext(fileName)

        if fname.find("ini") != -1 or fname.find("hFacC") != -1:
            # isini file
            try:
                times = np.genfromtxt(os.path.join(indir, "output_time.txt"))
            except OSError:
                times = np.zeros((2, 3))
            at = [np.array(times[:, 1][0])]
            ai = [np.array(times[:, 0][0])]
            nt = 1
        elif fname.find("avg") != -1:
            # isavg file
            times = np.genfromtxt(os.path.join(indir, "time_average_output_time.txt"))
            at = times[:, 1]
            ai = times[:, 0]
            nt = len(at)
        elif fname.find("pickup") != -1:
            # ispickup file (time axis is actually the tracer axis)
            at = ai = np.arange(len(np.genfromtxt(fileName + ".info")))
            nt = len(at)
        elif fname.find("avg") == -1:
            # This is tricky because state outputs and diagnostic outputs fall here, and
            #  may have different numbers of timesteps
            if fname in diagList:
                # This is a diagnostic field
                try:
                    times = np.genfromtxt(os.path.join(indir, "diag_output_time.txt"))
                except FileNotFoundError:
                    # Make an assumption that the diags are output at the same time as avg fields
                    times = np.genfromtxt(
                        os.path.join(indir, "time_average_output_time.txt")
                    )
                at = times[:, 1]
                ai = times[:, 0]
                nt = len(at)
            else:
                try:
                    info = len(np.genfromtxt(fileName + ".info"))

                    if info == len(
                        np.genfromtxt(
                            os.path.join(indir, "time_average_output_time.txt")
                        )
                    ):
                        # This is a diagnostic field, which has the same length as an average field
                        times = np.genfromtxt(
                            os.path.join(indir, "time_average_output_time.txt")
                        )
                        at = times[:, 1]
                        ai = times[:, 0]
                        nt = len(at)
                    elif info == len(
                        np.genfromtxt(
                            os.path.join(indir, "output_time.txt")
                        )
                    ):
                        # This is state output (could also be the same number of timesteps if "append" was not set initially)
                        times = np.genfromtxt(os.path.join(indir, "output_time.txt"))
                        at = times[:, 1]
                        ai = times[:, 0]
                        nt = len(at)
                except FileNotFoundError:
                    if ext ==".bin":
                        at = [0]
                        ai = [0]
                        nt = 1
                    else:
                        # not implemented
                        at = ai = nt = 0
        else:
            # not implemented
            at = ai = nt = 0
    return at, nt, ai


def get_mask_var_name(name, modelGrid):
    """
    Return the string name of the c/s/w mask requested in model grid (could be cmask or maskC, for example)
    """
    return sorted(
                [a for a in
                    [
                        b for b in list(modelGrid.keys())+list(modelGrid.coords) if "mask" in b.lower()
                     ] if name.lower() in a.lower()
                 ]
    )[0]


def tmm_index_to_model(
    modelGrid,
    tmmBoxesDict,
    tmmProfDict,
    tmmGridDict,
    rearrangeProfiles="Ir_pre",
):
    """
    Transform the TMM indices to shoebox model grid
    """
    # Get the name of the mask variable from modelGrid
    maskName = get_mask_var_name("c", modelGrid)

    # From the modelMask dimensions, create blank dictionariy to hold Indices
    modelIndices = dict({i: [] for i in modelGrid[maskName].dims})

    # If this is an LLC/CS grid and has to be handled differently because of
    #  the 5 ragged tiles/facets, which xarray/xmitgcm/ecco represents as 13
    #  equally sized faces
    if "nFaces" in tmmGridDict.keys():
        # Have to muck around with Indices (TMM provides index for 5xtiles not 13xfaces)
        # Rearrange profiles used means we need Ir_pre to reorder
        iz = np.ravel(
            tmmBoxesDict["izBox"][
                tmmProfDict[rearrangeProfiles].astype("int") - 1
            ] - 1
        ).astype("int")
        iy = np.ravel(
            tmmBoxesDict["iyBox"][
                tmmProfDict[rearrangeProfiles].astype("int") - 1
            ] - 1
        ).astype("int")
        ix = np.ravel(
            tmmBoxesDict["ixBox"][
                tmmProfDict[rearrangeProfiles].astype("int") - 1
            ] - 1
        ).astype("int")
        ib = np.ravel(
            tmmBoxesDict["boxfacenum"][
                tmmProfDict[rearrangeProfiles].astype("int") - 1
            ] - 1
        ).astype("int")
        nf = len(tmmBoxesDict["nbFace"])
        #    I  = np.ravel(tmmProfDict[rearrangeProfiles]-1).astype('int')

        for dim in modelIndices.keys():
            index = modelGrid[maskName] * modelGrid[dim]

            # Convert mitgcm indices to TMM vecors
            indexDict = ecco.llc_tiles_to_faces(
                index.transpose("k", "face", "j", "i").values,
                less_output=True,
            )

            # Establish empty templates
            indexMat = (
                np.zeros(
                    (
                        nf,
                        np.squeeze(tmmGridDict["nz"].max().astype("int")).tolist(),
                        np.squeeze(tmmGridDict["ny"].max().astype("int")).tolist(),
                        np.squeeze(tmmGridDict["nx"].max().astype("int")).tolist(),
                    )
                )
                * np.nan
            )
            for iface in np.arange(nf):
                indexMat[
                    iface,
                    : np.squeeze(tmmGridDict["nz"].max().astype("int")).tolist(),
                    : np.squeeze(tmmGridDict["ny"][iface].astype("int")).tolist(),
                    : np.squeeze(tmmGridDict["nx"][iface].astype("int")).tolist(),
                ] = indexDict[iface + 1]

            modelIndices[dim] = np.ravel(np.squeeze(indexMat[ib, iz, iy, ix]))
    else:
        # For a lat-lon grid there is no need to muck around with indices
        modelIndices["Depth"] = np.ravel(
            tmmBoxesDict["Zboxnom"][
                tmmProfDict[rearrangeProfiles].astype("int") - 1
            ]
        )
        modelIndices["Longitude"] = np.ravel(
            tmmBoxesDict["Xboxnom"][
                tmmProfDict[rearrangeProfiles].astype("int") - 1
            ]
        )
        modelIndices["Latitude"] = np.ravel(
            tmmBoxesDict["Yboxnom"][
                tmmProfDict[rearrangeProfiles].astype("int") - 1
            ]
        )
    return modelIndices


def model_index_to_tmm(
    modelGrid,
    tmmBoxesDict,
    tmmProfDict,
    tmmGridDict,
    rearrangeProfiles="Ir_pre",
):
    """
    Transform the shoebox model indices to TMM vector
    """
    # Get the name of the mask variable from modelGrid
    maskName = get_mask_var_name("c", modelGrid)

    # Get model coordinates/Indices projected on the TMM "boxes" axis
    modelIndicesIn = tmm_index_to_model(
        modelGrid,
        tmmBoxesDict,
        tmmProfDict,
        tmmGridDict,
        rearrangeProfiles=rearrangeProfiles,
    )

    # create a dictionary of dimensions from modelIndices (could be:
    #  [Depth, Latitude, Longitude], or [k, face, i, j], depending...)
    #  projected on the TMM "boxes" axis to create xarray dataArray
    coordDict = dict()
    for dim in modelIndicesIn.keys():
        coordDict[dim] = ("box", modelIndicesIn[dim])
    coordDict["box"] = ("box", np.arange(tmmBoxesDict["nb"][0], dtype="int"))

    tmmBlnk = xr.DataArray(
            data   = np.ones((tmmBoxesDict['nb'][0].astype("int"))),
            coords = coordDict,
            dims   = ["box"],
            ).set_index(
        box=(modelGrid[maskName].dims)
    )

    # Ir_post is 1-numBoxes, but this order gets messed up when unstacking the faces 
    #  of an LLC grid so stack and unstack Ir_post, and use to rearrange the unstacked output
    if "nFaces" in tmmGridDict.keys():
        xIr_post = xr.DataArray(
            data   = np.ravel((tmmProfDict["Ir_post"]-1)),
            coords = coordDict,
            dims   = ["box"],
            ).set_index(
                box = modelGrid[maskName].dims,
            ).unstack(
                "box",
            ).broadcast_like(
                xr.ones_like(modelGrid[maskName])
        )
        Ir_post = xIr_post.stack(
                box = modelGrid[maskName].dims,
            ).dropna(
                dim = "box"
            ).argsort().to_numpy().astype("int")
    else:
        Ir_post = tmmProfDict["Ir_post"].astype("int")-1

    # From the modelMask dimensions, create blank dictionariy to hold Indices
    modelIndicesOut = dict({i: [] for i in modelGrid[maskName].dims})

    for dim in modelIndicesOut.keys():
        index = (
                modelGrid[maskName] * modelGrid[maskName].coords[dim]
            ).where(
                modelGrid[maskName]
        )

        modelIndicesOut[dim] = index.stack(
                box = modelGrid[maskName].dims,
            ).dropna(
                dim = "box",
            ).broadcast_like(
                tmmBlnk
            ).isel(
                box = Ir_post.astype("int"),
        ).values
    return modelIndicesOut, Ir_post


def read_tmm_output(
    fileName,
    modelGrid,
    tmmBoxesDict,
    tmmProfDict,
    tmmGridDict,
    chunks            = 'auto',
    isForcing         = False,
    isPickup          = False,
    isDiagnostic      = False,
    rearrangeProfiles = "Ir_pre",
):
    """
    Read in Petsc/binary TMM output (forcing, *ini, pickup*, *avg, state, and diagnostic files)
    """
    if rearrangeProfiles in ["Irr", "Ir_pre", "Ir_post"]:
        IBoxOrder = rearrangeProfiles
    else:
        raise ValueError(
            "Cannot use that value for rearrangeProfiles"
        )

    # Get file extension, if any
    try:
        _, ext = os.path.splitext(fileName)
    except TypeError:
        ext = []

    # Get output times
    at, nt, ai = get_tmm_times(fileName)

    # Read in file to xarray using custom "TMMBackend"
    if isForcing:
        # Forcing may be spread over several files so use "open_mfdataset"
        #   which loads each individual file and concatenates them.
        # We can supply vector_length because we know that already
        try:
            Tr_io = xr.open_mfdataset(
                sorted(glob.glob(fileName+'_*')),
                num_time_entries = 1,
                engine           = TMMBackend,
                combine          = "nested",
                concat_dim       = ["time"],
                chunks           = chunks,
            )['tmm_data']
        except ValueError:
            # This error occurs for the binary forcing, like wind
            #  have to supply zero header length
            Tr_io = xr.open_mfdataset(
                sorted(glob.glob(fileName+'_*')),
                num_time_entries = 1,
                header_length    = 0,
                engine           = TMMBackend,
                combine          = "nested",
                concat_dim       = ["time"],
                chunks           = chunks,
            )['tmm_data']
    elif isDiagnostic or ext == ".bin":
        # Diagnostics are usually 2d fields in binary files, so have to
        #   set header_length to zero and supply num_time_entries to
        #   calculate vector_length
        Tr_io = xr.open_dataarray(
            fileName,
            num_time_entries = nt,
            header_length    = 0,
            engine           = TMMBackend,
            chunks           = chunks,
        )
    elif ext == ".petsc":
        # Assume everything else is a 3d field in a petsc file and
        #   supply num_time_entries for efficiency
        Tr_io = xr.open_dataarray(
            fileName,
            engine           = TMMBackend,
            times            = at,
            num_time_entries = nt,
            chunks           = chunks,
        )
    else:
        raise TypeError(
                "could not open file from "+fileName
            )

    maskName = get_mask_var_name("c", modelGrid)

    # Get model coordinates/Indices projected on the TMM "boxes" axis
    modelIndices = tmm_index_to_model(
        modelGrid,
        tmmBoxesDict,
        tmmProfDict,
        tmmGridDict,
        rearrangeProfiles = IBoxOrder,
    )

    if isDiagnostic or Tr_io.sizes['box'] < tmmBoxesDict['nb'][0].astype('int'):
        # Assume 2d field and process accordingly
        # Get depth coord name
        kname = list(set(modelGrid['rA'].dims) ^ set(modelGrid[maskName].dims))[0]

        # Create a dummy array with the shape of the full model domain
        grdBlnk = xr.ones_like(
            modelGrid[maskName].isel(
                    {kname:0},
                ).expand_dims(
                    dim={"time": nt},
            ))

        # reduce modelIndices to surface only using the 2d field "Depth"
        for dim in modelGrid[maskName].isel({kname:0}).dims:
            modelIndices[dim] = np.where(
                modelIndices[kname] == np.nanmin(
                    modelIndices[kname],
                ),
                modelIndices[dim],
                np.nan,
            )
            modelIndices[dim] = modelIndices[dim][
                ~np.isnan(modelIndices[dim])
            ]

        # create a dictionary of dimensions from modelIndices (could be:
        #  [Latitude, Longitude], or [face, i, j], depending...)
        #  projected on the TMM "boxes" axis to create xarray dataArray
        coordDict = dict()
        for dim in modelIndices.keys():
            coordDict[dim] = (
                "box", modelIndices[dim],
            )
        coordDict["box"] = ("box", np.arange(
                np.sum(
                    np.where(
                        tmmBoxesDict["izBox"] == 1, 1, 0
                    )),
                    dtype="int",
        ).astype("float"))
        coordDict["iter"] = ("time", ai)
        coordDict["time"] = ("time", at)
        _ = coordDict.pop(kname)

        boxDims = modelGrid[maskName].isel({kname:0}).dims
    else:
        # Create a dummy array with the shape of the full model domain
        grdBlnk = xr.ones_like(modelGrid[maskName].expand_dims(dim={"time": nt}))

        # create a dictionary of dimensions from modelIndices (could be:
        #  [Depth, Latitude, Longitude], or [k, face, i, j], depending...)
        #  projected on the TMM "boxes" axis to create xarray dataArray
        coordDict = dict()
        for dim in modelIndices.keys():
            coordDict[dim] = ("box", modelIndices[dim].astype("float"))
        coordDict["iter"] = ("time", ai)
        coordDict["time"] = ("time", at)
        coordDict["box"] = ("box", np.arange(
                tmmBoxesDict["nb"][0],
                dtype = "int",
            ))

        boxDims = modelGrid[maskName].dims

    # Finally unstack the TMM output from a vector to a shoebox
    Trgrd = Tr_io.assign_coords(
                coordDict,
            ).set_index(
                box = boxDims,
            ).unstack(
                "box",
            ).broadcast_like(
                grdBlnk,
            )

    if isPickup:
        # If the file is a pickup, the time axis is actually the diff tracers
        Trgrd = Trgrd.rename({"time": "tracer"})

    return Trgrd


def write_tmm_input(
    fileName,
    var,
    modelGrid,
    tmmBoxesDict,
    tmmProfDict,
    tmmGridDict,
    rearrangeProfilesIn  = "Ir_pre",
    rearrangeProfilesOut = "Ir_post",
    isPETSC              = True,
    precision            = '>f8',
    fileSuffix           = None,
):
    if rearrangeProfilesIn in ["Irr", "Ir_pre", "Ir_post"]:
        IBoxOrderIn = rearrangeProfilesIn
    else:
        raise ValueError(
            "Cannot use that value for rearrangeProfilesIn"
        )
    if rearrangeProfilesOut in ["Irr", "Ir_pre", "Ir_post"]:
        IBoxOrderOut = rearrangeProfilesOut
    else:
        raise ValueError(
            "Cannot use that value for rearrangeProfilesOut"
        )

    # Get model coordinates/Indices projected on the TMM "boxes" axis
    modelIndices, Ir_post = model_index_to_tmm(
        modelGrid,
        tmmBoxesDict,
        tmmProfDict,
        tmmGridDict,
        rearrangeProfiles = IBoxOrderIn,
    )
    maskName = get_mask_var_name("c", modelGrid)
    kname    = list(set(
            modelGrid['rA'].dims
        ) ^ set(
            modelGrid[maskName].dims
    ))[0]

    if kname not in var.dims:
        # Assume 2d field and process accordingly
        nBox = np.sum(
            np.where(
                tmmBoxesDict["izBox"] == 1, 1, 0,
        ))
        maskC    = modelGrid[maskName].isel({kname:0})
        maskDims = modelGrid[maskName].isel({kname:0}).dims
        # reduce modelIndices to surface only using the 2d field "Depth"
        for dim in modelGrid[maskName].isel({kname:0}).dims:
            modelIndices[dim] = np.where(
                modelIndices[kname] == np.nanmin(
                    modelIndices[kname],
                ),
                modelIndices[dim],
                np.nan,
            )
            modelIndices[dim] = modelIndices[dim][
                ~np.isnan(modelIndices[dim])
            ]
        Ir_post = np.where(
            modelIndices[kname] == np.nanmin(
                modelIndices[kname],
            ),
            Ir_post,
            np.nan,
        )
        Ir_post = Ir_post[
                ~np.isnan(Ir_post)
            ].astype("int")
        Ir_out = np.where(
            modelIndices[kname] == np.nanmin(
                modelIndices[kname],
            ),
            np.ravel(
                tmmProfDict[IBoxOrderOut]-1
            ),
            np.nan,
        )
        Ir_out = Ir_out[
                ~np.isnan(Ir_out)
            ].argsort().astype("int")
        # create a dictionary of dimensions from modelIndices (could be:
        #  [Latitude, Longitude], or [face, i, j], depending...)
        #  projected on the TMM "boxes" axis to create xarray dataArray
        coordDict = dict()
        for dim in modelIndices.keys():
            coordDict[dim] = ("box", modelIndices[dim])
        coordDict["box"] = ("box", np.arange(nBox,
                dtype="int",
            ))
        _ = coordDict.pop(kname)
    else:
        nBox = tmmBoxesDict['nb'][0].astype("int")
        maskC = modelGrid[maskName]
        maskDims = modelGrid[maskName].dims
        # create a dictionary of dimensions from modelIndices (could be:
        #  [Depth, Latitude, Longitude], or [k, face, i, j], depending...)
        #  projected on the TMM "boxes" axis to create xarray dataArray
        coordDict = dict()
        for dim in modelIndices.keys():
            coordDict[dim] = ("box", modelIndices[dim])
        coordDict["box"] = ("box", np.arange(
                tmmBoxesDict["nb"][0],
                dtype = "int",
            ))
        Ir_out = np.ravel(
            tmmProfDict[IBoxOrderOut]-1
        ).astype("int")

    tmmBlnk = xr.DataArray(
                data   = np.ones((nBox)),
                coords = coordDict,
                dims   = ["box"],
                ).set_index(
            box=(maskDims)
        )

    # Now we are ready to transform from the modelGrid to tmmBoxProfile
    stacked = var.where(
            maskC
        ).stack(
            box = maskDims,
        ).dropna(
            dim = "box",
        ).broadcast_like(
            tmmBlnk
        ).isel(
            box = Ir_post,
    )
    stacked_arranged = stacked.isel(
        box = Ir_out,
    )

    # Define the PETSc header, not necessarily used
    pVec    = 1211214
    header1 = np.array([pVec], dtype=np.dtype('>i4'))  # First header as ">i4"
    header2 = np.array([nBox], dtype=np.dtype('>i4'))  # Second header: vector length as ">i4"

    # Write to binary file
    if stacked_arranged.ndim > 1:
        # Get the name of the extra dimension
        if "time" in stacked_arranged.dims:
            tname = "time"
        else:
            tname = list(set(maskC.dims) ^ set(stacked_arranged.dims))[0]

        for tt in np.arange(stacked_arranged.sizes[tname]):
            if fileSuffix is not None and len(fileSuffix) == stacked_arranged.sizes[tname]:
                # use provided suffixes for the filenames (e.g. tracer names)
                fileNameOut = '{0}{1}'.format(fileName, fileSuffix[tt])
            else:
                # just use numerical value
                fileNameOut = '{0}_{1:02d}'.format(fileName, tt)
            with open(fileNameOut, 'wb') as f:
                if isPETSC:
                    f.write(header1.tobytes())  # Write the first header
                    f.write(header2.tobytes())  # Write the second header
                f.write(
                    stacked_arranged.isel(
                        {tname: tt},
                    ).astype(
                        np.dtype(precision)
                    ).to_numpy().tobytes()
                )  # Write the data
    else:
        with open(fileName, 'wb') as f:
            if isPETSC:
                f.write(header1.tobytes())  # Write the first header
                f.write(header2.tobytes())  # Write the second header
            f.write(
                stacked_arranged.astype(
                    np.dtype(precision)
                ).to_numpy().tobytes()
            )  # Write the data
    return stacked_arranged, Ir_out


def read_pco2(fileName, prec=">f8"):
    # Read atmospheric CO2 output from the TMM
    pco2_out = np.fromfile(fileName, dtype=prec)

    # Do some sleuthing for the time axis
    indir = "/".join(fileName.split("/")[:-1])
    try:
        times = np.genfromtxt(os.path.join(indir, "atm_output_time.txt"))
        at = times[:, 1]
        ai = times[:, 0]
    except (UnicodeDecodeError, OSError, IndexError, FileNotFoundError):
        try:
            times = np.genfromtxt(os.path.join(indir, "output_time.txt"))
            at = times[:, 1]
            ai = times[:, 0]
        except (UnicodeDecodeError, OSError, IndexError, FileNotFoundError):
            at = np.arange(len(pco2_out))
            ai = at   
        
    if len(pco2_out) != len(at):
        # try to locate initial pCO2
        try:
            pco2_ini = np.fromfile(
                fileName.replace("avg", "ini").replace("output", "ini"), dtype=prec,
            )
            pco2 = np.insert(pco2_out, 0, pco2_ini, axis=0)
        except (OSError, FileNotFoundError):
            try:
                pco2_ini = np.fromfile(
                    sorted(gb.glob(os.path.join(indir, "pickup_pCO2atm*bin")))[1], dtype=prec,
                )
                pco2 = np.insert(pco2_out, 0, pco2_ini, axis=0)
            except (UnicodeDecodeError, OSError, IndexError):
                # if you really cant find initial, then reduce length of at and ai
                at = at[:-1]
                ai = at[:-1]
    else:
        pco2 = pco2_out

    # output as pandas dataframe
    df = pd.DataFrame(data={"Time": at, "Iter": ai, "atm_pCO2": pco2})

    return df
