import math
import itertools
from collections import defaultdict

from AnyQt.QtWidgets import QApplication, QStyle, QSizePolicy

import numpy as np
import scipy.sparse as sp

import Orange
from Orange.data import StringVariable, ContinuousVariable
from Orange.data.util import hstack
from Orange.widgets import widget, gui, settings
from Orange.widgets.utils import itemmodels
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import Input, Output


INSTANCEID = "Source position (index)"
INDEX = "Position (index)"

class OWMergeData(widget.OWWidget):
    name = "Merge Data"
    description = "Merge data sets based on the values of selected data features."
    icon = "icons/MergeData.svg"
    priority = 1110

    class Inputs:
        data = Input("Data", Orange.data.Table, default=True, replaces=["Data A"])
        extra_data = Input("Extra Data", Orange.data.Table, replaces=["Data B"])

    class Outputs:
        data = Output("Data",
                      Orange.data.Table,
                      replaces=["Merged Data A+B", "Merged Data B+A", "Merged Data"])

    attr_augment_data = settings.Setting('', schema_only=True)
    attr_augment_extra = settings.Setting('', schema_only=True)
    attr_merge_data = settings.Setting('', schema_only=True)
    attr_merge_extra = settings.Setting('', schema_only=True)
    attr_combine_data = settings.Setting('', schema_only=True)
    attr_combine_extra = settings.Setting('', schema_only=True)
    merging = settings.Setting(0)

    attr_a = settings.Setting('', schema_only=True)
    attr_b = settings.Setting('', schema_only=True)

    want_main_area = False

    def __init__(self):
        super().__init__()

        # data
        self.dataA = None
        self.dataB = None

        # GUI
        w = QWidget(self)
        self.controlArea.layout().addWidget(w)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        w.setLayout(grid)

        # attribute A selection
        boxAttrA = gui.vBox(self, self.tr("Attribute A"), addToLayout=False)
        grid.addWidget(boxAttrA, 0, 0)

        self.attrViewA = gui.comboBox(boxAttrA, self, 'attr_a',
                                      orientation=Qt.Horizontal,
                                      sendSelectedValue=True,
                                      callback=self._invalidate)
        self.attrModelA = itemmodels.VariableListModel()
        self.attrViewA.setModel(self.attrModelA)

        # attribute  B selection
        boxAttrB = gui.vBox(self, self.tr("Attribute B"), addToLayout=False)
        grid.addWidget(boxAttrB, 0, 1)

        self.attrViewB = gui.comboBox(boxAttrB, self, 'attr_b',
                                      orientation=Qt.Horizontal,
                                      sendSelectedValue=True,
                                      callback=self._invalidate)
        self.attrModelB = itemmodels.VariableListModel()
        self.attrViewB.setModel(self.attrModelB)

        # info A
        boxDataA = gui.vBox(self, self.tr("Data A Input"), addToLayout=False)
        grid.addWidget(boxDataA, 1, 0)
        self.infoBoxDataA = gui.widgetLabel(boxDataA, self.dataInfoText(None))

        # info B
        boxDataB = gui.vBox(self, self.tr("Data B Input"), addToLayout=False)
        grid.addWidget(boxDataB, 1, 1)
        self.infoBoxDataB = gui.widgetLabel(boxDataB, self.dataInfoText(None))

        gui.rubber(self)

    def _setAttrs(self, model, data, othermodel, otherdata):
        model[:] = allvars(data) if data is not None else []

        if data is not None and otherdata is not None and \
                len(numpy.intersect1d(data.ids, otherdata.ids)):
            for model_ in (model, othermodel):
                if len(model_) and model_[0] != INSTANCEID:
                    model_.insert(0, INSTANCEID)

    @Inputs.data
    @check_sql_input
    def setDataA(self, data):
        self.dataA = data
        self._setAttrs(self.attrModelA, data, self.attrModelB, self.dataB)
        curr_index = -1
        if self.attr_a:
            curr_index = next((i for i, val in enumerate(self.attrModelA)
                               if str(val) == self.attr_a), -1)
        if curr_index != -1:
            self.attrViewA.setCurrentIndex(curr_index)
        else:
            self.attr_a = INDEX
        self.infoBoxDataA.setText(self.dataInfoText(data))

    @Inputs.extra_data
    @check_sql_input
    def setDataB(self, data):
        self.dataB = data
        self._setAttrs(self.attrModelB, data, self.attrModelA, self.dataA)
        curr_index = -1
        if self.attr_b:
            curr_index = next((i for i, val in enumerate(self.attrModelB)
                               if str(val) == self.attr_b), -1)
        if curr_index != -1:
            self.attrViewB.setCurrentIndex(curr_index)
        else:
            self.attr_b = INDEX
        self.infoBoxDataB.setText(self.dataInfoText(data))

    def handleNewSignals(self):
        self._invalidate()

    def dataInfoText(self, data):
        ninstances = 0
        nvariables = 0
        if data is not None:
            ninstances = len(data)
            nvariables = len(data.domain)

        instances = self.tr("%n instance(s)", None, ninstances)
        attributes = self.tr("%n variable(s)", None, nvariables)
        return "\n".join([instances, attributes])

    def commit(self):
        self.Warning.duplicate_names.clear()
        if self.data is None or len(self.data) == 0 or \
                self.extra_data is None or len(self.extra_data) == 0:
            merged_data = None
        else:
            merged_data = self.merge()
            if merged_data:
                merged_domain = merged_data.domain
                var_names = [var.name for var in chain(merged_domain.variables,
                                                       merged_domain.metas)]
                if len(set(var_names)) != len(var_names):
                    self.Warning.duplicate_names()
        self.Outputs.data.send(merged_data)

    def _invalidate(self):
        self.commit()

    def send_report(self):
        attr_a = None
        attr_b = None
        if self.dataA is not None:
            attr_a = self.attr_a
            if attr_a in self.dataA.domain:
                attr_a = self.dataA.domain[attr_a]
        if self.dataB is not None:
            attr_b = self.attr_b
            if attr_b in self.dataB.domain:
                attr_b = self.dataB.domain[attr_b]
        self.report_items((
            ("Attribute A", attr_a),
            ("Attribute B", attr_b),
        ))


def allvars(data):
    return (INDEX,) + data.domain.attributes + data.domain.class_vars + data.domain.metas


def merge(A, varA, B, varB):
    join_indices = left_join_indices(A, B, (varA,), (varB,))
    seen_set = set()

    def seen(val):
        return val in seen_set or bool(seen_set.add(val))

    merge_indices = [(i, j) for i, j in join_indices if not seen(i)]

    all_vars_A = set(A.domain.variables + A.domain.metas)
    iter_vars_B = itertools.chain(
        enumerate(B.domain.variables),
        ((-i, m) for i, m in enumerate(B.domain.metas, start=1))
    )
    reduced_indices_B = [i for i, var in iter_vars_B if not var in all_vars_A]
    reduced_B = B[:, list(reduced_indices_B)]

    return join_table_by_indices(A, reduced_B, merge_indices)


def group_table_indices(table, key_vars, exclude_unknown=False):
    """
    Group table indices based on values of selected columns (`key_vars`).

    Return a dictionary mapping all unique value combinations (keys)
    into a list of indices in the table where they are present.

    :param Orange.data.Table table:
    :param list-of-Orange.data.FeatureDescriptor] key_vars:
    :param bool exclude_unknown:

    """
    groups = defaultdict(list)
    for i, inst in enumerate(table):
        key = [inst.id if a == INSTANCEID else
               i if a == INDEX else inst[a]
                   for a in key_vars]
        if exclude_unknown and any(math.isnan(k) for k in key):
            continue
        key = tuple([str(k) for k in key])
        groups[key].append(i)
    return groups


def left_join_indices(table1, table2, vars1, vars2):
    key_map1 = group_table_indices(table1, vars1)
    key_map2 = group_table_indices(table2, vars2)
    indices = []
    for i, inst in enumerate(table1):
        key = tuple([str(inst.id if v == INSTANCEID else
                         i if v == INDEX else inst[v])
                            for v in vars1])
        if key in key_map1 and key in key_map2:
            for j in key_map2[key]:
                indices.append((i, j))
        else:
            return (str(val) if val else np.nan for val in col)

    @classmethod
    def _get_keymap(cls, data, var, as_string):
        """Return a generator of pairs (key, index) by enumerating and
        switching the values for rows (method `_values`).
        """
        return ((val, i)
                for i, val in enumerate(cls._values(data, var, as_string)))

    def _augment_indices(self, var_data, extra_map, as_string):
        """Compute a two-row array of indices:
        - the first row contains indices for the primary table,
        - the second row contains the matching rows in the extra table or -1"""
        data = self.data
        extra_map = dict(extra_map)
        # Don't match nans. This is needed since numpy supports using nan as
        # keys. If numpy fixes this, the below conditions will always be false,
        # so we're OK again.
        if np.nan in extra_map:
            del extra_map[np.nan]
        keys = (extra_map.get(val, -1)
                for val in self._values(data, var_data, as_string))
        return np.vstack((np.arange(len(data), dtype=np.int64),
                          np.fromiter(keys, dtype=np.int64, count=len(data))))

    def _merge_indices(self, var_data, extra_map, as_string):
        """Use _augment_indices to compute the array of indices,
        then remove those with no match in the second table"""
        augmented = self._augment_indices(var_data, extra_map, as_string)
        return augmented[:, augmented[1] != -1]

    def _combine_indices(self, var_data, extra_map, as_string):
        """Use _augment_indices to compute the array of indices,
        then add rows in the second table without a match in the first"""
        to_add, extra_map = tee(extra_map)
        # dict instead of set because we have pairs; we'll need only keys
        key_map = dict(self._get_keymap(self.data, var_data, as_string))
        # _augment indices will skip rows where the key in the left table
        # is nan. See comment in `_augment_indices` wrt numpy and nan in dicts
        if np.nan in key_map:
            del key_map[np.nan]
        keys = np.fromiter((j for key, j in to_add if key not in key_map),
                           dtype=np.int64)
        right_indices = np.vstack((np.full(len(keys), -1, np.int64), keys))
        return np.hstack(
            (self._augment_indices(var_data, extra_map, as_string),
             right_indices))

    def _join_table_by_indices(self, reduced_extra, indices):
        """Join (horizontally) self.data and reduced_extra, taking the pairs
        of rows given in indices"""
        if not len(indices):
            return None
        domain = Orange.data.Domain(
            *(getattr(self.data.domain, x) + getattr(reduced_extra.domain, x)
              for x in ("attributes", "class_vars", "metas")))
        X = self._join_array_by_indices(self.data.X, reduced_extra.X, indices)
        Y = self._join_array_by_indices(
            np.c_[self.data.Y], np.c_[reduced_extra.Y], indices)
        string_cols = [i for i, var in enumerate(domain.metas) if var.is_string]
        metas = self._join_array_by_indices(
            self.data.metas, reduced_extra.metas, indices, string_cols)
        return Orange.data.Table.from_numpy(domain, X, Y, metas)

    @staticmethod
    def _join_array_by_indices(left, right, indices, string_cols=None):
        """Join (horizontally) two arrays, taking pairs of rows given in indices
        """
        def prepare(arr, inds, str_cols):
            try:
                newarr = arr[inds]
            except IndexError:
                newarr = np.full_like(arr, np.nan)
            else:
                empty = np.full(arr.shape[1], np.nan)
                if str_cols:
                    assert arr.dtype == object
                    empty = empty.astype(object)
                    empty[str_cols] = ''
                newarr[inds == -1] = empty
            return newarr

        left_width = left.shape[1]
        str_left = [i for i in string_cols or () if i < left_width]
        str_right = [i - left_width for i in string_cols or () if i >= left_width]
        res = hstack((prepare(left, indices[0], str_left),
                      prepare(right, indices[1], str_right)))
        return res


def main():
    app = QApplication([])

    w = OWMergeData()
    zoo = Orange.data.Table("zoo")
    A = zoo[:, [0, 1, 2, "type", -1]]
    B = zoo[:, [3, 4, 5, "type", -1]]
    w.setDataA(A)
    w.setDataB(B)
    w.handleNewSignals()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
