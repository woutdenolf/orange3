"""Pandas DataFrame↔Table conversion helpers"""
from unittest.mock import patch

import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from pandas.core.arrays import SparseArray
from pandas.core.arrays.sparse.dtype import SparseDtype
from pandas.api.types import (
    is_categorical_dtype, is_object_dtype,
    is_datetime64_any_dtype, is_numeric_dtype,
)

from Orange.data import (
    Table, Domain, DiscreteVariable, StringVariable, TimeVariable,
    ContinuousVariable,
)
from Orange.data.table import Role

__all__ = ['table_from_frame', 'table_to_frame']


class OrangeDataFrame(pd.DataFrame):
    _metadata = ["orange_variables", "orange_weights",
                 "orange_attributes", "orange_role"]

    def __init__(self, *args, **kwargs):
        """
        A pandas DataFrame wrapper for one of Table's numpy arrays:
            - sets index values corresponding to Orange's global row indices
              e.g. ['_o1', '_o2'] (allows Orange to handle selection)
            - remembers the array's role in the Table (attribute, class var, meta)
            - keeps the Variable objects, and uses them in back-to-table conversion,
              should a column name match a variable's name
            - stores weight values (legacy)

        Parameters
        ----------
        table : Table
        orange_role : Role, (default=Role.Attribute)
            When converting back to an orange table, the DataFrame will
            convert to the right role (attrs, class vars, or metas)
        """
        if len(args) <= 0 or not isinstance(args[0], Table):
            super().__init__(*args, **kwargs)
            return
        table = args[0]
        if 'orange_role' in kwargs:
            role = kwargs.pop('orange_role')
        elif len(args) >= 2:
            role = args[1]
        else:
            role = Role.Attribute

        if role == Role.Attribute:
            data = table.X
            vars_ = table.domain.attributes
        elif role == Role.ClassAttribute:
            data = table.Y
            vars_ = table.domain.class_vars
        else:  # if role == Role.Meta:
            data = table.metas
            vars_ = table.domain.metas

        index = ['_o' + str(id_) for id_ in table.ids]
        varsdict = {var._name: var for var in vars_}
        columns = varsdict.keys()

        if sp.issparse(data):
            data = data.asformat('csc')
            sparrays = [SparseArray.from_spmatrix(data[:, i]) for i in range(data.shape[1])]
            data = dict(enumerate(sparrays))
            super().__init__(data, index=index, **kwargs)
            self.columns = columns
            # a hack to keep Orange df _metadata in sparse->dense conversion
            self.sparse.to_dense = self.__patch_constructor(self.sparse.to_dense)
        else:
            super().__init__(data=data, index=index, columns=columns, **kwargs)

        self.orange_role = role
        self.orange_variables = varsdict
        self.orange_weights = (dict(zip(index, table.W))
                               if table.W.size > 0 else {})
        self.orange_attributes = table.attributes

    def __patch_constructor(self, method):
        def new_method(*args, **kwargs):
            with patch(
                    'pandas.DataFrame',
                    OrangeDataFrame
            ):
                df = method(*args, **kwargs)
            df.__finalize__(self)
            return df

        return new_method

    @property
    def _constructor(self):
        return OrangeDataFrame

    def to_orange_table(self):
        return table_from_frame(self)

    def __finalize__(self, other, method=None, **_):
        """
        propagate metadata from other to self

        Parameters
        ----------
        other : the object from which to get the attributes that we are going
            to propagate
        method : optional, a passed method name ; possibly to take different
            types of propagation actions based on this

        """
        if method == 'concat':
            objs = other.objs
        elif method == 'merge':
            objs = other.left, other.right
        else:
            objs = [other]

        orange_role = getattr(self, 'orange_role', None)
        dicts = {dname: getattr(self, dname, {})
                 for dname in ('orange_variables',
                               'orange_weights',
                               'orange_attributes')}
        for obj in objs:
            other_role = getattr(obj, 'orange_role', None)
            if other_role is not None:
                orange_role = other_role

            for dname, dict_ in dicts.items():
                other_dict = getattr(obj, dname, {})
                dict_.update(other_dict)

        object.__setattr__(self, 'orange_role', orange_role)
        for dname, dict_ in dicts.items():
            object.__setattr__(self, dname, dict_)

        return self

    pd.DataFrame.__finalize__ = __finalize__


def _is_discrete(s, force_nominal):
    return (is_categorical_dtype(s) or
            is_object_dtype(s) and (force_nominal or
                                    s.nunique() < s.size ** .666))


def _is_datetime(s):
    if is_datetime64_any_dtype(s):
        return True
    try:
        if is_object_dtype(s):
            pd.to_datetime(s, infer_datetime_format=True)
            return True
    except Exception:  # pylint: disable=broad-except
        pass
    return False


def vars_from_df(df, role=None, force_nominal=False):
    if role is None and hasattr(df, 'orange_role'):
        _role = df.orange_role
    else:
        _role = role

    # If df index is not a simple RangeIndex (or similar), put it into data
    if not any(str(i).startswith('_o') for i in df.index) \
            and not (df.index.is_integer() and (df.index.is_monotonic_increasing
                                                or df.index.is_monotonic_decreasing)):
        df = df.reset_index()

    Xcols, Ycols, Mcols = [], [], []
    Xexpr, Yexpr, Mexpr = [], [], []
    attrs, class_vars, metas = [], [], []

    contains_strings = _role == Role.Meta
    for column in df.columns:
        s = df[column]
        if hasattr(df, 'orange_variables') and column in df.orange_variables:
            original_var = df.orange_variables[column]
            var = original_var.copy(compute_value=None)
            if _role == Role.Attribute:
                Xcols.append(column)
                Xexpr.append(None)
                attrs.append(var)
            elif _role == Role.ClassAttribute:
                Ycols.append(column)
                Yexpr.append(None)
                class_vars.append(var)
            else:  # if role == Role.Meta:
                Mcols.append(column)
                Mexpr.append(None)
                metas.append(var)
        elif _is_discrete(s, force_nominal):
            discrete = s.astype('category').cat
            var = DiscreteVariable(str(column),
                                   discrete.categories.astype(str).tolist())
            attrs.append(var)
            Xcols.append(column)
            Xexpr.append(lambda s, _: np.asarray(
                s.astype('category').cat.codes.replace(-1, np.nan)
            ))
        elif _is_datetime(s):
            var = TimeVariable(str(column))
            s = pd.to_datetime(s, infer_datetime_format=True)
            attrs.append(var)
            Xcols.append(column)
            Xexpr.append(lambda s, v: np.asarray(
                s.astype('str').replace('NaT', np.nan).map(v.parse)
            ))
        elif is_numeric_dtype(s):
            var = ContinuousVariable(str(column))
            attrs.append(var)
            Xcols.append(column)
            Xexpr.append(None)
        else:
            contains_strings = True
            var = StringVariable(str(column))
            metas.append(var)
            Mcols.append(column)
            Mexpr.append(lambda s, _: np.asarray(s, dtype=object))

    # if role isn't explicitly set, try to
    # export dataframes into one contiguous block.
    # for this all columns must be of the same role
    if isinstance(df, OrangeDataFrame) \
            and not role \
            and contains_strings \
            and not force_nominal:
        attrs.extend(class_vars)
        attrs.extend(metas)
        metas = attrs
        Xcols.extend(Ycols)
        Xcols.extend(Mcols)
        Mcols = Xcols
        Xexpr.extend(Yexpr)
        Xexpr.extend(Mexpr)
        Mexpr = Xexpr

        attrs, class_vars = [], []
        Xcols, Ycols = [], []
        Xexpr, Yexpr = [], []

    XYM = []
    for Avars, Acols, Aexpr in zip(
            (attrs, class_vars, metas),
            (Xcols, Ycols, Mcols),
            (Xexpr, Yexpr, Mexpr)):
        if not Acols:
            A = None if Acols != Xcols else np.empty((df.shape[0], 0))
            XYM.append(A)
            continue
        if not any(Aexpr):
            Adf = df if all(c in Acols
                            for c in df.columns) else df[Acols]
            if all(isinstance(a, SparseDtype) for a in Adf.dtypes):
                A = csr_matrix(Adf.sparse.to_coo())
            else:
                A = np.asarray(Adf)
            XYM.append(A)
            continue
        # we'll have to copy the table to resolve any expressions
        # TODO eliminate expr (preprocessing for pandas -> table)
        A = np.array([expr(df[col], var) if expr else np.asarray(df[col])
                      for var, col, expr in zip(Avars, Acols, Aexpr)]).T
        XYM.append(A)

    return XYM, Domain(attrs, class_vars, metas)


def table_from_frame(df, *, force_nominal=False):
    XYM, domain = vars_from_df(df, force_nominal=force_nominal)

    if hasattr(df, 'orange_weights') and hasattr(df, 'orange_attributes'):
        W = [df.orange_weights[i] for i in df.index
             if i in df.orange_weights]
        if len(W) != len(df.index):
            W = None
        attributes = df.orange_attributes
        ids = [int(i[2:]) if str(i).startswith('_o') else Table.new_id()
               for i in df.index]
    else:
        W = None
        attributes = None
        ids = None

    return Table.from_numpy(
        domain,
        *XYM,
        W=W,
        attributes=attributes,
        ids=ids
    )


def table_from_frames(xdf, ydf, mdf):
    dfs = xdf, ydf, mdf

    if not all(df.shape[0] == xdf.shape[0] for df in dfs):
        raise ValueError(f"Leading dimension mismatch "
                         f"(not {xdf.shape[0]} == {ydf.shape[0]} == {mdf.shape[0]})")

    xXYM, xDomain = vars_from_df(xdf, role=Role.Attribute)
    yXYM, yDomain = vars_from_df(ydf, role=Role.ClassAttribute)
    mXYM, mDomain = vars_from_df(mdf, role=Role.Meta)

    XYM = (xXYM[0], yXYM[1], mXYM[2])
    domain = Domain(xDomain.attributes, yDomain.class_vars, mDomain.metas)

    index_iter = (filter(lambda ind: ind.startswith('_o'),
                         set(df.index[i] for df in dfs))
                  for i in range(len(xdf.shape[0])))
    ids = (i[0] if len(i) == 1 else Table.new_id()
           for i in index_iter)

    attributes = {}
    W = None
    for df in dfs:
        if isinstance(df, OrangeDataFrame):
            W = [df.orange_weights[i] for i in df.index
                 if i in df.orange_weights]
            if len(W) != len(df.index):
                W = None
        else:
            W = None
        attributes.update(df.orange_attributes)

    return Table.from_numpy(
        domain,
        *XYM,
        W=W,
        attributes=attributes,
        ids=ids
    )


def table_to_frame(tab, include_metas=False):
    """
    Convert Orange.data.Table to pandas.DataFrame

    Parameters
    ----------
    tab : Table

    include_metas : bool, (default=False)
        Include table metas into dataframe.

    Returns
    -------
    pandas.DataFrame
    """

    def _column_to_series(col, vals):
        result = ()
        if col.is_discrete:
            codes = pd.Series(vals).fillna(-1).astype(int)
            result = (col.name, pd.Categorical.from_codes(
                codes=codes, categories=col.values, ordered=True
            ))
        elif col.is_time:
            result = (col.name, pd.to_datetime(vals, unit='s').to_series().reset_index()[0])
        elif col.is_continuous:
            dt = float
            # np.nan are not compatible with int column
            nan_values_in_column = [t for t in vals if np.isnan(t)]
            if col.number_of_decimals == 0 and len(nan_values_in_column) == 0:
                dt = int
            result = (col.name, pd.Series(vals).astype(dt))
        elif col.is_string:
            result = (col.name, pd.Series(vals))
        return result

    def _columns_to_series(cols, vals):
        return [_column_to_series(col, vals[:, i]) for i, col in enumerate(cols)]

    x, y, metas = [], [], []
    domain = tab.domain
    if domain.attributes:
        x = _columns_to_series(domain.attributes, tab.X)
    if domain.class_vars:
        y_values = tab.Y.reshape(tab.Y.shape[0], len(domain.class_vars))
        y = _columns_to_series(domain.class_vars, y_values)
    if domain.metas:
        metas = _columns_to_series(domain.metas, tab.metas)
    all_series = dict(x + y + metas)
    all_vars = tab.domain.variables
    if include_metas:
        all_vars += tab.domain.metas
    original_column_order = [var.name for var in all_vars]
    unsorted_columns_df = pd.DataFrame(all_series)
    return unsorted_columns_df[original_column_order]


def table_to_frames(table):
    xdf = OrangeDataFrame(table, Role.Attribute)
    ydf = OrangeDataFrame(table, Role.ClassAttribute)
    mdf = OrangeDataFrame(table, Role.Meta)

    return xdf, ydf, mdf


def amend_table_with_frame(table, df, role):
    arr = Role.get_arr(role, table)
    if arr.shape[0] != df.shape[0]:
        raise ValueError(f"Leading dimension mismatch "
                         f"(not {arr.shape[0]} == {df.shape[0]})")

    XYM, domain = vars_from_df(df, role=role)

    if role == Role.Attribute:
        table.domain = Domain(domain.attributes,
                              table.domain.class_vars,
                              table.domain.metas)
        table.X = XYM[0]
    elif role == Role.ClassAttribute:
        table.domain = Domain(table.domain.attributes,
                              domain.class_vars,
                              table.domain.metas)
        table.Y = XYM[1]
    else:  # if role == Role.Meta:
        table.domain = Domain(table.domain.attributes,
                              table.domain.class_vars,
                              domain.metas)
        table.metas = XYM[2]

    if isinstance(df, OrangeDataFrame):
        table.attributes.update(df.orange_attributes)
